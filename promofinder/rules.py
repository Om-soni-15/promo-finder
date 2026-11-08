"""No-API candidate finder + a rules score usable on ANY promo (incl. LLM ones)."""
from __future__ import annotations

from .parse import Show, Episode, fmt_time
from .clips import (Promo, Segment, beats_in, snap_start, clean_end_times,
                    overlap_ratio, HOOK_WINDOW, MIN_TOTAL, MAX_TOTAL)


def rules_score(show: Show, promo: Promo, sig: dict) -> dict:
    """Explainable 0-10ish components; also used as a tie-breaker in LLM mode."""
    first, last = promo.segments[0], promo.segments[-1]
    ep0, epL = show.ep(first.episode), show.ep(last.episode)
    early = beats_in(ep0, first.start, first.start + HOOK_WINDOW)
    hook = max((sig[b.id]["hook"] for b in early), default=0.0)
    ending = [b for b in epL.beats if last.end - 3.0 <= b.end <= last.end + 0.05 and b.start < last.end]
    cliff = max((sig[b.id]["cliff"] for b in ending), default=0.0)
    inside = [b for s in promo.segments for b in beats_in(show.ep(s.episode), s.start, s.end)]
    density = sum(sig[b.id]["hook"] for b in inside) / max(promo.duration, 1) * 10
    density = min(density, 4.0)
    dur_bonus = 0.5 if 25 <= promo.duration <= 45 else 0.0
    jump_pen = {"single": 0.0, "multi-segment": 0.5, "cross-episode": 1.5}.get(promo.kind, 0.0)
    total = hook + 1.2 * cliff + density + dur_bonus - jump_pen
    hook_why = sorted({w for b in early for w in sig[b.id]["hook_why"]})
    cliff_why = sorted({w for b in ending for w in sig[b.id]["cliff_why"]})
    return {
        "rules_total": round(total, 2), "rules_hook": round(hook, 2), "rules_cliff": round(cliff, 2),
        "rules_density": round(density, 2), "hook_signals": hook_why, "cliff_signals": cliff_why,
    }


def _why_line(sc: dict) -> str:
    h = ", ".join(sc["hook_signals"][:2]) or "a charged opening"
    c = ", ".join(sc["cliff_signals"][:2]) or "an unresolved beat"
    return f"Opens on {h}; cuts on {c}."


def _best_end(show, ep, start, sig, lo_len=MIN_TOTAL, hi_len=MAX_TOTAL, prefix=0.0):
    best = None
    for t in clean_end_times(ep, start + lo_len - prefix, min(ep.duration, start + hi_len - prefix)):
        p = Promo([Segment(ep.number, start, t)], "rules")
        sc = rules_score(show, p, sig)
        if best is None or sc["rules_total"] > best[0]:
            best = (sc["rules_total"], t)
    return best


def hook_burst(ep: Episode, beat, sig, min_len=5.0, max_len=15.0) -> Segment | None:
    """A short hook moment: from the hook beat to the end of the exchange."""
    s = snap_start(ep, beat.start, max_back=0.8)
    opts = clean_end_times(ep, s + min_len, min(ep.duration, s + max_len))
    talk = [t for t in opts if any(b.type in ("dialogue", "action") and abs(b.end - t) < 0.05 for b in ep.beats)]
    pick = (talk or opts)
    if not pick:
        return None
    return Segment(ep.number, s, pick[0])


def episode_candidates(show: Show, ep: Episode, sig: dict, per_episode: int = 3) -> list[Promo]:
    starts = [b for b in ep.beats if sig[b.id]["hook"] >= 2.0 and b.type != "camera"]
    start_times = sorted({snap_start(ep, b.start) for b in starts} | {0.0})
    found: list[Promo] = []
    for s in start_times:
        best = _best_end(show, ep, s, sig)
        if not best:
            continue
        p = Promo([Segment(ep.number, s, best[1])], "rules")
        p.scores = rules_score(show, p, sig)
        found.append(p)

    # multi-segment: strongest hook moment + the episode's ending cliffhanger
    if starts:
        hb = max(starts, key=lambda b: sig[b.id]["hook"])
        burst = hook_burst(ep, hb, sig)
        if burst:
            for tail_len in (20.0, 30.0, 40.0):
                tail_start_t = ep.duration - tail_len
                if tail_start_t <= burst.end + 5:
                    continue
                # start the tail where a line of dialogue begins
                cands = [b.start for b in ep.beats if b.type == "dialogue" and tail_start_t - 4 <= b.start <= tail_start_t + 4]
                if not cands:
                    continue
                ts = snap_start(ep, min(cands, key=lambda x: abs(x - tail_start_t)))
                p = Promo([burst, Segment(ep.number, ts, ep.duration)], "rules", kind="multi-segment")
                if MIN_TOTAL <= p.duration <= MAX_TOTAL:
                    p.scores = rules_score(show, p, sig)
                    found.append(p)

    strong = [p for p in found if p.scores["rules_hook"] >= 1.5]
    found = strong or found
    found.sort(key=lambda p: -p.scores["rules_total"])
    kept: list[Promo] = []
    for p in found:
        if all(overlap_ratio(p, k) < 0.35 for k in kept):
            kept.append(p)
        if len(kept) >= per_episode:
            break
    for p in kept:
        p.why = _why_line(p.scores)
    return kept


def cross_episode_candidates(show: Show, sig: dict, limit: int = 8) -> list[Promo]:
    """Hook moment from episode N + the ending cliffhanger of episode N+1."""
    out = []
    for ep in show.episodes:
        nxt = show.ep(ep.number + 1)
        if not nxt:
            continue
        starts = [b for b in ep.beats if sig[b.id]["hook"] >= 2.5 and b.type != "camera"]
        if not starts:
            continue
        hb = max(starts, key=lambda b: sig[b.id]["hook"])
        burst = hook_burst(ep, hb, sig)
        if not burst:
            continue
        for tail_len in (20.0, 30.0):
            target = nxt.duration - tail_len
            cands = [b.start for b in nxt.beats if b.type == "dialogue" and target - 4 <= b.start <= target + 4]
            if not cands:
                continue
            ts = snap_start(nxt, min(cands, key=lambda x: abs(x - target)))
            p = Promo([burst, Segment(nxt.number, ts, nxt.duration)], "rules", kind="cross-episode")
            if MIN_TOTAL <= p.duration <= MAX_TOTAL:
                p.scores = rules_score(show, p, sig)
                p.why = _why_line(p.scores) + f" Bridges Ep {ep.number}->{nxt.number}."
                out.append(p)
    out.sort(key=lambda p: -p.scores["rules_total"])
    kept = []
    for p in out:
        if all(overlap_ratio(p, k) < 0.35 for k in kept):
            kept.append(p)
        if len(kept) >= limit:
            break
    return kept


def moment_menu(show: Show, ep: Episode, sig: dict, n_hooks: int = 3) -> dict:
    """Compact list of an episode's best hook moments + its ending (for the cross-episode LLM pass)."""
    hooks = sorted((b for b in ep.beats if b.type != "camera"), key=lambda b: -sig[b.id]["hook"])[:n_hooks]
    tail = [b for b in ep.beats if b.start >= ep.duration - 25 and b.type != "camera"]
    return {
        "episode": ep.number,
        "logline": ep.logline,
        "hook_moments": [{"beat": b.id, "t": fmt_time(b.start), "text": b.label()[:160]} for b in hooks],
        "ending": [{"beat": b.id, "t": fmt_time(b.start), "text": b.label()[:160]} for b in tail[-6:]],
        "duration": fmt_time(ep.duration),
    }
