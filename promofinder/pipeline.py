"""Orchestration: rules mode (no API) and llm mode (rules + LLM finder + LLM judge)."""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import prompts as P
from .clips import Promo, Segment, build_and_validate, transcript, overlap_ratio, beats_in
from .llm import make_llm, extract_json, LLMError
from .parse import Show, fmt_time
from .rules import (episode_candidates, cross_episode_candidates, rules_score, hook_burst)
from .signals import annotate, CALM_EMOTIONS, _emotion_words
from .clips import snap_start


class Run:
    def __init__(self, show: Show, cfg: dict, out_dir: Path, log=print):
        self.show, self.cfg, self.out = show, cfg, out_dir
        self.log = log
        self.sig: dict[str, dict] = {}
        for e in show.episodes:
            self.sig.update(annotate(e))
        self.llm = None
        self.context = {"premise": "", "characters": []}
        self.raw_dir = out_dir / "raw_llm"
        self.errors: list[str] = []
        self.rejected: list[Promo] = []
        self.t0 = time.time()

    # ---------------------------------------------------------------- utils
    def _save_raw(self, name: str, system: str, user: str, reply: str):
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        (self.raw_dir / f"{name}.txt").write_text(
            f"### SYSTEM\n{system}\n\n### USER\n{user}\n\n### REPLY\n{reply}\n", encoding="utf-8")

    def _ask(self, name, system, user, hint=None):
        reply = self.llm.complete(system, user, task=name, mock_hint=hint)
        self._save_raw(name, system, user, reply)
        return extract_json(reply)

    def _validate(self, p: Promo) -> Promo | None:
        build_and_validate(self.show, p, self.sig)
        if not p.valid:
            self.rejected.append(p)
            return None
        p.scores.update(rules_score(self.show, p, self.sig))
        if not p.why:
            from .rules import _why_line
            p.why = _why_line(p.scores)
        return p

    # ---------------------------------------------------------------- stages
    def rules_candidates(self) -> list[Promo]:
        per_ep = int(self.cfg["rules"].get("per_episode", 3))
        out = []
        for ep in self.show.episodes:
            for p in episode_candidates(self.show, ep, self.sig, per_ep):
                if self._validate(p):
                    out.append(p)
        for p in cross_episode_candidates(self.show, self.sig, int(self.cfg["rules"].get("cross_limit", 8))):
            if self._validate(p):
                out.append(p)
        return out

    def show_context(self):
        loglines = "\n".join(f"Ep {e.number}: {e.logline}" for e in self.show.episodes)
        speakers = "\n".join(f"{k} ({v})" for k, v in self.show.speakers().items())
        hint = {"premise": self.show.episodes[0].logline,
                "characters": [{"name": k.title(), "aliases": [k], "who": ""} for k in list(self.show.speakers())[:12]]}
        try:
            self.context = self._ask("00_show_context", P.CONTEXT_SYSTEM,
                                     P.CONTEXT_USER.format(show=self.show.name, loglines=loglines, speakers=speakers), hint)
        except LLMError as e:
            self.errors.append(f"show context failed: {e}")

    def _characters_text(self):
        rows = []
        for c in self.context.get("characters", [])[:25]:
            al = ", ".join(c.get("aliases", []))
            rows.append(f"- {c.get('name')}: {c.get('who', '')} [script names: {al}]")
        return "\n".join(rows) or "(unknown)"

    def _beats_listing(self, ep):
        rows = []
        for b in ep.beats:
            txt = b.label()
            if b.type == "camera" and len(txt) > 150:
                txt = txt[:147] + "..."
            rows.append(f"{b.id} | {fmt_time(b.start)}-{fmt_time(b.end)} | {txt}")
        return "\n".join(rows)

    def llm_episode_candidates(self) -> list[Promo]:
        n_per = int(self.cfg["llm_pipeline"].get("episodes_per_call", 2))
        per_ep = int(self.cfg["llm_pipeline"].get("promos_per_episode", 3))
        eps = self.show.episodes
        out = []
        for i in range(0, len(eps), n_per):
            batch = eps[i:i + n_per]
            block = "\n\n".join(P.EPISODE_BLOCK.format(num=e.number, duration=fmt_time(e.duration),
                                                       logline=e.logline, beats=self._beats_listing(e)) for e in batch)
            user = P.EPISODE_USER.format(show=self.show.name, premise=self.context.get("premise", ""),
                                         characters=self._characters_text(), episodes_block=block, per_ep=per_ep)
            hint = {"promos": [self._promo_to_hint(p) for e in batch
                               for p in episode_candidates(self.show, e, self.sig, 2)]}
            name = "ep_" + "_".join(f"{e.number:02d}" for e in batch)
            self.log(f"  LLM finder: episodes {batch[0].number}-{batch[-1].number}")
            try:
                data = self._ask(name, P.EPISODE_SYSTEM, user, hint)
            except LLMError as e:
                self.errors.append(f"{name}: {e}")
                continue
            for item in (data.get("promos") or []):
                p = self._promo_from_llm(item, "llm-episode")
                if p and self._validate(p):
                    out.append(p)
        return out

    def _promo_to_hint(self, p: Promo) -> dict:
        segs = []
        for s in p.segments:
            bs = beats_in(self.show.ep(s.episode), s.start, s.end)
            segs.append({"start_beat": bs[0].id, "end_beat": bs[-1].id})
        return {"segments": segs, "hook_beat": "", "hook": "", "cliffhanger": "", "why": "[mock] " + p.why,
                "context_risk": "unknown"}

    def _promo_from_llm(self, item: dict, source: str) -> Promo | None:
        segs = []
        for s in item.get("segments") or []:
            sb, eb = str(s.get("start_beat", "")).strip(), str(s.get("end_beat", "")).strip()
            b = self.show.beat(sb)
            if b is None:
                self.rejected.append(Promo([], source, why=item.get("why", ""), valid=False,
                                           reject_reason=f"LLM used unknown beat id {sb!r}"))
                return None
            segs.append(Segment(b.episode, b.start, b.end, sb, eb))
        if not segs:
            return None
        p = Promo(segs, source, hook_beat=str(item.get("hook_beat", "")), hook=item.get("hook", ""),
                  cliffhanger=item.get("cliffhanger", ""), why=item.get("why", ""))
        risk = str(item.get("context_risk", "")).strip()
        if risk and risk.lower() not in ("none", "n/a", "unknown", ""):
            p.scores["llm_context_risk"] = risk
        return p

    def _menu(self):
        """Ready-made moments per episode for the cross-episode pass."""
        menu, items = [], {}
        for ep in self.show.episodes:
            lines = [f"EPISODE {ep.number}: {ep.logline}"]
            hooks = sorted((b for b in ep.beats if b.type != "camera" and self.sig[b.id]["hook"] >= 2),
                           key=lambda b: -self.sig[b.id]["hook"])
            used = []
            for b in hooks:
                if len(used) >= 2:
                    break
                seg = hook_burst(ep, b, self.sig)
                if not seg or any(abs(seg.start - u.start) < 8 for u in used):
                    continue
                used.append(seg)
                mid = f"M{ep.number:02d}-H{len(used)}"
                items[mid] = seg
                txt = " / ".join(x.label()[:110] for x in beats_in(ep, seg.start, seg.end) if x.type != "camera")[:260]
                lines.append(f"  {mid} ({seg.duration:.0f}s): {txt}")
            target = ep.duration - 20
            starts = [b.start for b in ep.beats if b.type == "dialogue" and target - 5 <= b.start <= target + 5]
            if starts:
                ts = snap_start(ep, min(starts, key=lambda x: abs(x - target)))
                seg = Segment(ep.number, ts, ep.duration)
                mid = f"M{ep.number:02d}-END"
                items[mid] = seg
                txt = " / ".join(x.label()[:110] for x in beats_in(ep, seg.start, seg.end) if x.type != "camera")[:300]
                lines.append(f"  {mid} ({seg.duration:.0f}s): {txt}")
            menu.append("\n".join(lines))
        return "\n\n".join(menu), items

    def llm_cross_candidates(self) -> list[Promo]:
        n = int(self.cfg["llm_pipeline"].get("cross_promos", 6))
        menu, items = self._menu()
        keys = list(items)
        hint = {"promos": [{"moments": [keys[0], keys[2] if len(keys) > 2 else keys[-1]],
                            "hook": "", "cliffhanger": "", "why": "[mock] cross pick"}]} if len(keys) > 2 else {"promos": []}
        self.log("  LLM finder: cross-episode")
        try:
            data = self._ask("cross_episode", P.CROSS_SYSTEM,
                             P.CROSS_USER.format(show=self.show.name, premise=self.context.get("premise", ""), menu=menu, n=n), hint)
        except LLMError as e:
            self.errors.append(f"cross_episode: {e}")
            return []
        out = []
        for item in data.get("promos") or []:
            ids = [str(m).strip() for m in item.get("moments") or []]
            bad = [m for m in ids if m not in items]
            if bad or len(ids) < 2:
                self.rejected.append(Promo([], "llm-cross", why=item.get("why", ""), valid=False,
                                           reject_reason=f"unknown/too few moment ids {bad or ids}"))
                continue
            segs = [Segment(items[m].episode, items[m].start, items[m].end) for m in ids]
            p = Promo(segs, "llm-cross", hook=item.get("hook", ""), cliffhanger=item.get("cliffhanger", ""),
                      why=item.get("why", ""))
            if self._validate(p):
                out.append(p)
        return out

    def judge(self, pool: list[Promo]):
        top_n = int(self.cfg["llm_pipeline"].get("judge_top_n", 40))
        bs = int(self.cfg["llm_pipeline"].get("judge_batch", 6))
        pre = sorted(pool, key=lambda p: -(p.scores["rules_total"] + (2.0 if p.source.startswith("llm") else 0)))[:top_n]
        for i in range(0, len(pre), bs):
            batch = pre[i:i + bs]
            clips = "\n\n".join(f"### CLIP {p.id} (total {p.duration:.0f}s)\n{transcript(self.show, p)}" for p in batch)
            hint = {"scores": [{"id": p.id, "hook": min(5, 1 + round(p.scores['rules_hook'])),
                                "cliffhanger": min(5, 1 + round(p.scores['rules_cliff'] / 1.5)),
                                "standalone": 3, "intensity": min(5, 1 + round(p.scores['rules_density'])),
                                "first_3_seconds": "[mock]", "why": "[mock]", "weakness": "[mock]", "verdict": "maybe"}
                               for p in batch]}
            name = f"judge_{i // bs + 1:02d}"
            self.log(f"  LLM judge: batch {i // bs + 1}/{(len(pre) + bs - 1) // bs}")
            try:
                data = self._ask(name, P.JUDGE_SYSTEM, P.JUDGE_USER.format(clips=clips), hint)
            except LLMError as e:
                self.errors.append(f"{name}: {e}")
                continue
            by_id = {p.id: p for p in batch}
            for s in data.get("scores") or []:
                p = by_id.get(str(s.get("id")))
                if not p:
                    continue
                try:
                    j = {k: max(1, min(5, int(s.get(k, 1)))) for k in ("hook", "cliffhanger", "standalone", "intensity")}
                except (TypeError, ValueError):
                    continue
                p.scores["judge"] = j
                p.scores["judge_verdict"] = s.get("verdict", "")
                p.scores["judge_first_3s"] = s.get("first_3_seconds", "")
                p.scores["judge_weakness"] = s.get("weakness", "")
                p.scores["judge_why"] = s.get("why", "")

    # ---------------------------------------------------------------- ranking
    def final_score(self, p: Promo) -> float:
        w = self.cfg["ranking"]
        rules_norm = max(0.0, min(100.0, p.scores["rules_total"] / 15.0 * 100))
        j = p.scores.get("judge")
        if j:
            jw = w["judge_weights"]
            js = sum(jw[k] * j[k] for k in jw) / sum(jw.values())
            base = 0.8 * ((js - 1) / 4 * 100) + 0.2 * rules_norm
        else:
            base = rules_norm * (1.0 if self.llm is None else 0.7)
        first_ep = p.segments[0].episode
        n_eps = max(e.number for e in self.show.episodes)
        adj = 0.0
        if first_ep <= w.get("early_episode_max", 5):
            adj += w.get("early_bonus", 6)
        elif first_ep <= w.get("mid_episode_max", 15):
            adj += w.get("mid_bonus", 3)
        if first_ep > n_eps * w.get("late_fraction", 0.8):
            adj -= w.get("late_penalty", 6)
        adj -= w.get("flag_penalty", 2) * len(p.flags)
        p.scores["funnel_and_flag_adjust"] = adj
        return round(base + adj, 1)

    def quality_flags(self, p: Promo):
        hb = self.show.beat(p.hook_beat)
        if hb is not None and hb.type in ("camera", "action", "sfx"):
            p.flags.append("visual-only hook: strength depends on the footage, not checkable from script")
        last = p.segments[-1]
        epL = self.show.ep(last.episode)
        tail = [b for b in beats_in(epL, last.end - 5, last.end) if b.type == "dialogue"]
        if tail and _emotion_words(tail[-1]) & CALM_EMOTIONS:
            p.flags.append("ends on a calm/tender line — may feel resolved")
        speakers = {b.speaker for s in p.segments for b in beats_in(self.show.ep(s.episode), s.start, s.end) if b.speaker}
        if len(speakers) >= 5:
            p.flags.append(f"{len(speakers)} speaking characters — may confuse a newcomer")
        j = p.scores.get("judge")
        if j and j["standalone"] <= 2:
            p.flags.append("judge: hard to follow without earlier episodes")
        if j and j["hook"] <= 2:
            p.flags.append("judge: weak first 3 seconds")
        n_eps = max(e.number for e in self.show.episodes)
        if p.segments[0].episode > n_eps * self.cfg["ranking"].get("late_fraction", 0.8):
            p.flags.append("late-arc clip — may spoil the payoff and lands viewers deep in the show")

    def rank(self, pool: list[Promo]) -> list[Promo]:
        for p in pool:
            self.quality_flags(p)
            p.scores["final"] = self.final_score(p)
        pool.sort(key=lambda p: -p.scores["final"])
        kept: list[Promo] = []
        for p in pool:  # drop near-duplicates (keep the better-scored one)
            dup = next((k for k in kept if overlap_ratio(p, k) >= 0.6), None)
            if dup is None:
                kept.append(p)
            else:
                dup.scores.setdefault("duplicates_merged", []).append(p.id)
        return kept

    def post_first(self, ranked: list[Promo], n: int = 10) -> list[dict]:
        chosen, per_ep = [], {}
        for p in ranked:
            if len(chosen) >= n:
                break
            if any(overlap_ratio(p, c) > 0.2 for c in chosen):
                continue
            e0 = p.segments[0].episode
            if per_ep.get(e0, 0) >= 2:
                continue
            per_ep[e0] = per_ep.get(e0, 0) + 1
            chosen.append(p)
        out = []
        for i, p in enumerate(chosen, 1):
            out.append({"post_order": i, "id": p.id, "final_score": p.scores["final"],
                        "reason": self._post_reason(p, i)})
        return out

    def _post_reason(self, p: Promo, i: int) -> str:
        bits = []
        j = p.scores.get("judge")
        if j:
            best = max(j, key=j.get)
            bits.append(f"judge {j['hook']}/{j['cliffhanger']}/{j['standalone']}/{j['intensity']} (hook/cliff/standalone/intensity), strongest on {best}")
        else:
            bits.append(f"rules hook {p.scores['rules_hook']}, cliff {p.scores['rules_cliff']}")
        e0 = p.segments[0].episode
        if e0 <= self.cfg["ranking"].get("early_episode_max", 5):
            bits.append(f"from Ep {e0}: a new viewer who taps through starts at the beginning")
        elif e0 <= self.cfg["ranking"].get("mid_episode_max", 15):
            bits.append(f"from Ep {e0}: early enough that little backstory is needed")
        if p.kind != "single":
            bits.append(p.kind)
        if p.flags:
            bits.append(f"{len(p.flags)} caution flag(s)")
        return "; ".join(bits)

    # ---------------------------------------------------------------- main
    def execute(self) -> dict:
        mode = self.cfg.get("mode", "rules")
        self.log(f"Parsed {len(self.show.episodes)} episodes; mode={mode}")
        pool = self.rules_candidates()
        self.log(f"  rules: {len(pool)} candidates")
        if mode == "llm":
            self.llm = make_llm(self.cfg["llm"])
            self.show_context()
            pool += self.llm_episode_candidates()
            pool += self.llm_cross_candidates()
        for i, p in enumerate(pool, 1):
            p.id = f"P{i:03d}"
        if mode == "llm":
            self.judge(pool)
        ranked = self.rank(pool)
        for r, p in enumerate(ranked, 1):
            p.scores["rank"] = r
        return {"ranked": ranked, "post_first": self.post_first(ranked, int(self.cfg["ranking"].get("post_first_n", 10))),
                "rejected": self.rejected, "errors": self.errors, "context": self.context,
                "seconds": round(time.time() - self.t0, 1),
                "llm_log": self.llm.log if self.llm else []}
