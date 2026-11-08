"""Write outputs: promos.json, promos.csv, post_first.md, quality_report.md, report.html."""
from __future__ import annotations

import csv
import html
import json
import re
from collections import Counter
from pathlib import Path

from .clips import Promo, Segment, transcript
from .parse import Show, fmt_time, parse_time


def seg_str(p: Promo) -> str:
    return " + ".join(f"Ep{s.episode} {fmt_time(s.start)}-{fmt_time(s.end)}" for s in p.segments)


def write_all(show: Show, res: dict, cfg: dict, out: Path, gold: list[dict] | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    ranked: list[Promo] = res["ranked"]
    by_id = {p.id: p for p in ranked}

    # JSON (everything, machine-readable)
    (out / "promos.json").write_text(json.dumps({
        "show": show.name, "mode": cfg.get("mode"),
        "model": f"{cfg['llm'].get('provider')}/{cfg['llm'].get('model')}" if cfg.get("mode") == "llm" else "rules-only",
        "post_first": res["post_first"],
        "promos": [dict(p.to_dict(), rank=p.scores.get("rank")) for p in ranked],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # CSV (for an editor / spreadsheet)
    with open(out / "promos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "id", "final_score", "kind", "source", "segments", "total_sec", "hook",
                    "cliffhanger", "why", "judge(h/c/s/i)", "verdict", "flags"])
        for p in ranked:
            j = p.scores.get("judge")
            w.writerow([p.scores.get("rank"), p.id, p.scores.get("final"), p.kind, p.source, seg_str(p),
                        p.duration, p.hook, p.cliffhanger, p.scores.get("judge_why") or p.why,
                        f"{j['hook']}/{j['cliffhanger']}/{j['standalone']}/{j['intensity']}" if j else "",
                        p.scores.get("judge_verdict", ""), " | ".join(p.flags)])

    # rejected (honesty about what failed validation)
    (out / "rejected.json").write_text(json.dumps(
        [{"source": p.source, "reason": p.reject_reason, "why": p.why,
          "segments": [s.to_dict() for s in p.segments]} for p in res["rejected"]], indent=2), encoding="utf-8")

    # post-first list
    lines = [f"# Post these first — {show.name}\n"]
    for item in res["post_first"]:
        p = by_id[item["id"]]
        lines += [f"## {item['post_order']}. {p.id} — {seg_str(p)} ({p.duration:.0f}s, {p.kind})",
                  f"- **Hook:** {p.hook}", f"- **Cliffhanger:** {p.cliffhanger}",
                  f"- **Why it works:** {p.scores.get('judge_why') or p.why}",
                  f"- **Why post it now:** {item['reason']}"]
        if p.flags:
            lines.append(f"- **Check:** {'; '.join(p.flags)}")
        lines.append("")
    (out / "post_first.md").write_text("\n".join(lines), encoding="utf-8")

    q = quality(show, res, cfg, gold)
    (out / "quality_report.md").write_text(q["markdown"], encoding="utf-8")
    meta = {"show": show.name, "episodes": len(show.episodes), "mode": cfg.get("mode"),
            "llm": cfg["llm"] if cfg.get("mode") == "llm" else None, "seconds": res["seconds"],
            "errors": res["errors"], "calls": q["calls"], "context": res["context"], "metrics": q["metrics"]}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.html").write_text(html_report(show, res, cfg, q), encoding="utf-8")
    return q


# --------------------------------------------------------------------- quality
def _gold_match(p: Promo, g: dict) -> float:
    """Fraction of the gold clip covered by the promo."""
    gsegs = g["segments"]
    total = sum(parse_time(s["end"]) - parse_time(s["start"]) for s in gsegs) or 1
    cov = 0.0
    for gs in gsegs:
        a, b = parse_time(gs["start"]), parse_time(gs["end"])
        for s in p.segments:
            if s.episode == int(gs["episode"]):
                cov += max(0.0, min(b, s.end) - max(a, s.start))
    return cov / total


def quality(show: Show, res: dict, cfg: dict, gold: list[dict] | None):
    ranked: list[Promo] = res["ranked"]
    md = [f"# Quality report — {show.name}\n"]
    metrics = {}
    n_eps = len(show.episodes)
    covered = {s.episode for p in ranked for s in p.segments}
    by_src = Counter(p.source for p in ranked)
    by_kind = Counter(p.kind for p in ranked)
    durs = sorted(p.duration for p in ranked)
    metrics.update({"valid_promos": len(ranked), "rejected": len(res["rejected"]),
                    "episodes_covered": len(covered), "by_source": dict(by_src), "by_kind": dict(by_kind)})
    md += ["## Volume",
           f"- {len(ranked)} valid promos (after de-duplication) from {n_eps} episodes; "
           f"{len(covered)}/{n_eps} episodes contribute at least one.",
           f"- By source: {dict(by_src)}; by kind: {dict(by_kind)}.",
           f"- Duration: min {durs[0]:.0f}s / median {durs[len(durs)//2]:.0f}s / max {durs[-1]:.0f}s." if durs else "",
           f"- {len(res['rejected'])} candidates rejected by the validator (see rejected.json).", ""]
    rej = Counter(p.reject_reason.split(" '")[0].split(" (")[0] for p in res["rejected"])
    if rej:
        md += ["## Why candidates were rejected"] + [f"- {k}: {v}" for k, v in rej.most_common()] + [""]
    fixes = Counter()
    for p in ranked:
        for n in p.notes:
            fixes["end moved to avoid mid-line cut" if "end moved" in n else
                  "lead-in trimmed for 3-second hook" if "lead-in" in n else
                  "trimmed to 60 s" if "fit 60" in n else
                  "extended to 20 s" if "reach 20" in n else "re-picked hook beat"] += 1
    if fixes:
        md += ["## Automatic fixes applied by code (not hidden)"] + [f"- {k}: {v}" for k, v in fixes.most_common()] + [""]
    flags = Counter()
    for p in ranked:
        for f in p.flags:
            flags[re.sub(r"\d+", "N", f.split(":")[0].split(" —")[0])] += 1
    top20 = ranked[:20]
    md += ["## Where the tool is likely picking badly (flags)"]
    md += [f"- {k}: {v} promos" for k, v in flags.most_common()] or ["- none"]
    md += [f"- In the top 20: {sum(1 for p in top20 if p.flags)} carry at least one flag.", ""]

    judged = [p for p in ranked if p.scores.get("judge")]
    if judged:
        import statistics as st
        xs = [p.scores["rules_total"] for p in judged]
        js = [sum(p.scores["judge"].values()) for p in judged]
        rho = _spearman(xs, js)
        verdicts = Counter(p.scores.get("judge_verdict", "") for p in judged)
        src_mean = {}
        for s in sorted({p.source for p in judged}):
            v = [sum(p.scores["judge"].values()) / 4 for p in judged if p.source == s]
            src_mean[s] = round(st.mean(v), 2)
        metrics.update({"judged": len(judged), "rules_vs_judge_spearman": rho, "verdicts": dict(verdicts),
                        "mean_judge_by_source": src_mean})
        md += ["## Rules vs LLM judge",
               f"- {len(judged)} promos judged blind (judge sees only the clip, no show context = the 'stranger' test).",
               f"- Verdicts: {dict(verdicts)}.",
               f"- Mean judge score (1-5) by source: {src_mean}.",
               f"- Spearman correlation rules-score vs judge-score: {rho} "
               "(low = the cheap rules and the LLM disagree; those disagreements are where to look first).", ""]
        worst = sorted(judged, key=lambda p: sum(p.scores["judge"].values()))[:5]
        md += ["### Lowest-judged promos (examples of bad picks)"]
        for p in worst:
            md.append(f"- {p.id} {seg_str(p)} — judge {p.scores['judge']} — {p.scores.get('judge_weakness', '')}")
        md.append("")

    if gold:
        hits, ranks, offsets = 0, [], []
        md += ["## Against hand-picked gold promos"]
        for g in gold:
            best = max(ranked, key=lambda p: _gold_match(p, g), default=None)
            cov = _gold_match(best, g) if best else 0
            ok = cov >= 0.5
            hits += ok
            if ok:
                r = best.scores.get("rank", 0)
                ranks.append(r)
                g0 = g["segments"][0]
                if best.segments[0].episode == int(g0["episode"]):
                    offsets.append(abs(best.segments[0].start - parse_time(g0["start"])))
            md.append(f"- {'HIT ' if ok else 'MISS'} {g.get('name', '')}: best match {best.id if best else '-'} "
                      f"covers {cov:.0%}" + (f", ranked #{best.scores.get('rank')}" if ok else ""))
        rec = hits / len(gold)
        in_top20 = sum(1 for r in ranks if r <= 20)
        metrics.update({"gold_recall": round(rec, 2), "gold_in_top20": in_top20,
                        "gold_mean_start_offset_sec": round(sum(offsets) / len(offsets), 1) if offsets else None})
        md += [f"- Recall: {hits}/{len(gold)} gold promos found (>=50% overlap); {in_top20} of them in the top 20.",
               f"- Mean start-point error on hits: {metrics['gold_mean_start_offset_sec']} s.", ""]

    calls = res.get("llm_log", [])
    tin = sum(c["usage"].get("in", 0) for c in calls)
    tout = sum(c["usage"].get("out", 0) for c in calls)
    callinfo = {"total": len(calls), "cached": sum(1 for c in calls if c["cached"]), "tokens_in": tin, "tokens_out": tout}
    if calls:
        md += ["## Cost of this run",
               f"- {len(calls)} LLM calls ({callinfo['cached']} from cache), {tin:,} input + {tout:,} output tokens.",
               ""]
    if res["errors"]:
        md += ["## Errors during the run"] + [f"- {e}" for e in res["errors"]] + [""]
    md += ["## Known blind spots (by design)",
           "- Works from the script, not the video: acting, faces and pacing are invisible; "
           "visual-only hooks are flagged rather than trusted.",
           "- Assumes script timestamps match the final cut; if the edit drifts, cut points drift.",
           "- 'Stands alone' is judged by a model pretending to be a stranger — a proxy, not real viewers.",
           "- Ranking weights (funnel bonus for early episodes, late-arc penalty) are assumptions to tune "
           "against real promo performance (CTR, 3-second hold, installs)."]
    return {"markdown": "\n".join(md), "metrics": metrics, "calls": callinfo}


def _spearman(a, b):
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    if len(a) < 3:
        return None
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return round(num / den, 2) if den else None


# ------------------------------------------------------------------------ html
def html_report(show: Show, res: dict, cfg: dict, q: dict) -> str:
    ranked: list[Promo] = res["ranked"]
    by_id = {p.id: p for p in ranked}
    post_ids = [i["id"] for i in res["post_first"]]
    reasons = {i["id"]: i["reason"] for i in res["post_first"]}
    e = html.escape

    def card(p: Promo, highlight=False):
        j = p.scores.get("judge")
        jtxt = (f"<span class='pill'>judge H{j['hook']} C{j['cliffhanger']} S{j['standalone']} I{j['intensity']}"
                f" · {e(str(p.scores.get('judge_verdict', '')))}</span>") if j else ""
        flags = "".join(f"<li>{e(f)}</li>" for f in p.flags)
        fixes = "".join(f"<li>{e(n)}</li>" for n in p.notes)
        why = p.scores.get("judge_why") or p.why
        return f"""
<details class="card{' hl' if highlight else ''}" {'open' if highlight and post_ids.index(p.id) < 3 else ''}>
<summary><b>#{p.scores.get('rank')} · {p.id}</b> <span class="seg">{e(seg_str(p))}</span>
 <span class="pill">{p.duration:.0f}s</span> <span class="pill">{e(p.kind)}</span> <span class="pill">{e(p.source)}</span>
 <span class="pill score">{p.scores.get('final')}</span> {jtxt}
 <div class="hook">⚡ {e(p.hook[:180])}</div></summary>
<div class="body">
<p><b>Cliffhanger:</b> {e(p.cliffhanger)}</p>
<p><b>Why it works:</b> {e(why)}</p>
{f"<p><b>Why post it now:</b> {e(reasons[p.id])}</p>" if p.id in reasons else ""}
{f"<p><b>Judge's main weakness:</b> {e(str(p.scores.get('judge_weakness')))}</p>" if j else ""}
{f"<p><b>Quality flags</b></p><ul>{flags}</ul>" if flags else ""}
{f"<p class='muted'><b>Auto-fixes</b></p><ul class='muted'>{fixes}</ul>" if fixes else ""}
<pre>{e(transcript(show, p))}</pre>
</div></details>"""

    model = f"{cfg['llm'].get('provider')}/{cfg['llm'].get('model')}" if cfg.get("mode") == "llm" else "rules only (no LLM)"
    m = q["metrics"]
    banner = ""
    if cfg.get("mode") == "llm" and res.get("errors"):
        n_ok = sum(1 for c in res.get("llm_log", []))
        banner = (f"<div class='warn'><b>Warning:</b> {len(res['errors'])} LLM call(s) failed "
                  f"({n_ok} succeeded). Parts of this report fall back to rules-only results. "
                  f"First error: {e(res['errors'][0][:300])}</div>")
    first = "".join(card(by_id[i], True) for i in post_ids)
    rest = "".join(card(p) for p in ranked if p.id not in post_ids)
    qhtml = e(q["markdown"])
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Promos — {e(show.name)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{{font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#1c1c1e;background:#fafafa}}
h1{{font-size:22px;margin-bottom:4px}} h2{{font-size:17px;margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:4px}}
.meta{{color:#666;font-size:13px}} .card{{background:#fff;border:1px solid #e3e3e6;border-radius:10px;margin:8px 0;padding:10px 14px}}
.card.hl{{border-left:4px solid #e4572e}} summary{{cursor:pointer;list-style:none}} .seg{{font-family:ui-monospace,monospace;font-size:13px}}
.pill{{display:inline-block;font-size:12px;background:#f0f0f3;border-radius:20px;padding:1px 8px;margin-left:4px}}
.score{{background:#1c1c1e;color:#fff}} .hook{{margin-top:4px;color:#333}}
pre{{white-space:pre-wrap;font-size:12.5px;background:#f6f6f8;border-radius:8px;padding:10px;max-height:420px;overflow:auto}}
.muted{{color:#777;font-size:13px}}
.warn{{background:#fdecea;border:1px solid #f5c2bd;color:#8a1c12;border-radius:8px;padding:10px 14px;margin:12px 0}} ul{{margin-top:0}}
</style></head><body>
<h1>Promo picks — {e(show.name)}</h1>
<div class="meta">{len(show.episodes)} episodes · model: {e(model)} · {m.get('valid_promos')} valid promos ·
{m.get('episodes_covered')}/{len(show.episodes)} episodes covered · run took {res['seconds']}s</div>
{banner}
<h2>Post these first ({len(post_ids)})</h2>{first}
<h2>All other promos, ranked</h2>{rest}
<h2>Quality report</h2><pre>{qhtml}</pre>
</body></html>"""
