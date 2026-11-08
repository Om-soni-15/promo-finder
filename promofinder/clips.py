"""Promo data model + the deterministic part: snapping and validation.

LLMs (and rules) propose promos as beat IDs. This module turns them into real
timestamps, cuts cleanly (never mid-line), enforces the 20-60 s / 3-second-hook
rules, and records every automatic fix so nothing is hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from .parse import Show, Episode, Beat, fmt_time

MIN_TOTAL, MAX_TOTAL, HOOK_WINDOW = 20.0, 60.0, 3.0


@dataclass
class Segment:
    episode: int
    start: float
    end: float
    start_beat: str = ""
    end_beat: str = ""

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)

    def to_dict(self):
        return {
            "episode": self.episode, "start": fmt_time(self.start), "end": fmt_time(self.end),
            "start_sec": round(self.start, 2), "end_sec": round(self.end, 2),
            "duration_sec": self.duration, "start_beat": self.start_beat, "end_beat": self.end_beat,
        }


@dataclass
class Promo:
    segments: list[Segment]
    source: str                    # rules | llm-episode | llm-cross
    hook_beat: str = ""
    hook: str = ""
    cliffhanger: str = ""
    why: str = ""
    kind: str = "single"           # single | multi-segment | cross-episode
    scores: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)   # automatic fixes & warnings
    flags: list[str] = field(default_factory=list)   # quality-check flags
    valid: bool = True
    reject_reason: str = ""
    id: str = ""

    @property
    def duration(self) -> float:
        return round(sum(s.duration for s in self.segments), 2)

    @property
    def episodes(self) -> list[int]:
        out = []
        for s in self.segments:
            if s.episode not in out:
                out.append(s.episode)
        return out

    def to_dict(self):
        return {
            "id": self.id, "kind": self.kind, "source": self.source,
            "episodes": self.episodes, "total_duration_sec": self.duration,
            "segments": [s.to_dict() for s in self.segments],
            "hook": self.hook, "hook_beat": self.hook_beat, "cliffhanger": self.cliffhanger,
            "why": self.why, "scores": self.scores, "flags": self.flags, "auto_fixes": self.notes,
            "valid": self.valid, "reject_reason": self.reject_reason,
        }


# ---------------------------------------------------------------- helpers
def beats_in(ep: Episode, start: float, end: float) -> list[Beat]:
    return [b for b in ep.beats if b.start >= start - 0.05 and b.start < end - 0.05]


def snap_start(ep: Episode, t: float, max_back: float = 1.0) -> float:
    """Start on a camera-shot boundary if one begins just before t."""
    cands = [b.start for b in ep.beats if b.type == "camera" and t - max_back <= b.start <= t]
    return min(cands) if cands else t


def snap_end(ep: Episode, t: float, max_fwd: float = 1.0) -> float:
    """Never cut mid-dialogue; finish the shot if it ends within max_fwd."""
    changed = True
    while changed:
        changed = False
        for b in ep.beats:
            if b.type == "dialogue" and b.start < t - 0.05 and b.end > t + 0.05:
                t = b.end
                changed = True
    shots = [b.end for b in ep.beats if b.type == "camera" and b.start < t and t <= b.end <= t + max_fwd]
    if shots:
        t = max(shots)
    return min(t, ep.duration)


def clean_end_times(ep: Episode, lo: float, hi: float) -> list[float]:
    """Candidate cut points in [lo, hi] that don't split a line of dialogue."""
    ends = sorted({b.end for b in ep.beats if lo <= b.end <= hi})
    out = []
    for t in ends:
        if any(b.type == "dialogue" and b.start < t - 0.05 and b.end > t + 0.05 for b in ep.beats):
            continue
        out.append(t)
    return out


def transcript(show: Show, promo: Promo, include_camera: bool = True, relative: bool = True) -> str:
    """Readable script of exactly what is inside the promo."""
    lines, offset = [], 0.0
    for i, seg in enumerate(promo.segments):
        ep = show.ep(seg.episode)
        lines.append(f"--- segment {i+1}: Episode {seg.episode} {fmt_time(seg.start)}-{fmt_time(seg.end)} ---")
        for b in beats_in(ep, seg.start, seg.end):
            if b.type == "camera" and not include_camera:
                continue
            t = (b.start - seg.start + offset) if relative else b.start
            lines.append(f"[{fmt_time(t)}] {b.label()}")
        offset += seg.duration
    return "\n".join(lines)


# ------------------------------------------------------------- validation
def build_and_validate(show: Show, promo: Promo, sig: dict[str, dict]) -> Promo:
    """Snap, auto-fix within limits, and mark invalid promos with a reason.

    `sig` is the per-beat signal table from signals.annotate (all episodes merged).
    """
    if not promo.segments:
        promo.valid, promo.reject_reason = False, "no segments"
        return promo

    # 1. resolve beat IDs -> timestamps, snap cut points
    fixed: list[Segment] = []
    for seg in promo.segments:
        ep = show.ep(seg.episode)
        if ep is None:
            promo.valid, promo.reject_reason = False, f"episode {seg.episode} not in input"
            return promo
        if seg.start_beat:
            b0, b1 = show.beat(seg.start_beat), show.beat(seg.end_beat or seg.start_beat)
            if b0 is None or b1 is None:
                promo.valid, promo.reject_reason = False, f"unknown beat id {seg.start_beat}/{seg.end_beat}"
                return promo
            if b0.episode != seg.episode or b1.episode != seg.episode:
                seg.episode = b0.episode
                if b1.episode != b0.episode:
                    promo.valid, promo.reject_reason = False, "segment spans two episodes"
                    return promo
            seg.start, seg.end = b0.start, max(b1.end, b0.end)
        if seg.end <= seg.start:
            promo.valid, promo.reject_reason = False, "segment end is before start"
            return promo
        s2, e2 = snap_start(ep, seg.start), snap_end(ep, seg.end)
        if abs(e2 - seg.end) > 0.05:
            promo.notes.append(f"ep{seg.episode}: end moved {fmt_time(seg.end)}->{fmt_time(e2)} to avoid cutting mid-line/shot")
        seg.start, seg.end = s2, e2
        fixed.append(seg)
    promo.segments = fixed

    # 2. overlapping segments inside one promo
    for i in range(len(fixed)):
        for j in range(i + 1, len(fixed)):
            a, b = fixed[i], fixed[j]
            if a.episode == b.episode and a.start < b.end and b.start < a.end:
                promo.valid, promo.reject_reason = False, "segments overlap"
                return promo
    if len(fixed) > 1:
        promo.kind = "cross-episode" if len(promo.episodes) > 1 else "multi-segment"
        chrono = all((fixed[i].episode, fixed[i].start) < (fixed[i + 1].episode, fixed[i + 1].start)
                     for i in range(len(fixed) - 1))
        if not chrono:
            promo.flags.append("non-chronological order (flash-forward) — check it reads clearly")

    # 3. hook must land in the first 3 seconds
    first = fixed[0]
    ep0 = show.ep(first.episode)
    hb = show.beat(promo.hook_beat) if promo.hook_beat else None
    if hb is None or hb.episode != first.episode or not (first.start - 0.05 <= hb.start < first.end):
        # choose the strongest beat in the first 3 s
        early = beats_in(ep0, first.start, first.start + HOOK_WINDOW)
        if early:
            hb = max(early, key=lambda b: sig.get(b.id, {}).get("hook", 0))
            if promo.hook_beat:
                promo.notes.append("hook beat was outside the first segment; re-picked from first 3 s")
            promo.hook_beat = hb.id
    if hb is not None and hb.start - first.start > HOOK_WINDOW:
        lead = hb.start - first.start
        new_start = snap_start(ep0, hb.start, max_back=0.8)
        if promo.duration - (new_start - first.start) >= MIN_TOTAL:
            promo.notes.append(f"trimmed {lead:.1f}s lead-in so the hook lands in the first 3 s")
            first.start = new_start
        else:
            promo.flags.append(f"hook arrives at {lead:.1f}s (>3 s)")
    if hb is not None and not promo.hook:
        promo.hook = hb.label()

    # 4. duration window
    if promo.duration > 1.5 * MAX_TOTAL:
        promo.valid, promo.reject_reason = False, f"far too long ({promo.duration:.0f}s) — not a promo selection"
        return promo
    last = fixed[-1]
    epL = show.ep(last.episode)
    if promo.duration > MAX_TOTAL:
        excess = promo.duration - MAX_TOTAL
        lo = last.start + max(2.0, last.duration - excess - 15)
        hi = last.end - excess
        opts = [t for t in clean_end_times(epL, lo, hi) if t > last.start + 2]
        if opts:
            best = max(opts, key=lambda t: _cliff_at(epL, t, sig))
            promo.notes.append(f"trimmed end {fmt_time(last.end)}->{fmt_time(best)} to fit 60 s")
            last.end = best
        else:
            promo.valid, promo.reject_reason = False, f"too long ({promo.duration:.0f}s) and no clean trim point"
            return promo
    if promo.duration < MIN_TOTAL:
        need = MIN_TOTAL - promo.duration
        opts = clean_end_times(epL, last.end + need, min(epL.duration, last.end + need + 20))
        if opts:
            best = max(opts, key=lambda t: _cliff_at(epL, t, sig) - 0.02 * (t - last.end))
            promo.notes.append(f"extended end {fmt_time(last.end)}->{fmt_time(best)} to reach 20 s")
            last.end = best
        else:
            promo.valid, promo.reject_reason = False, f"too short ({promo.duration:.0f}s) and cannot extend"
            return promo

    if not promo.cliffhanger:
        window = beats_in(epL, max(last.start, last.end - 6), last.end)
        if not window:  # one long final shot: use the last beats that start inside the segment
            window = beats_in(epL, last.start, last.end)[-2:]
        tail = [b for b in window if b.type != "camera"] or window
        if tail:
            promo.cliffhanger = max(tail, key=lambda b: sig.get(b.id, {}).get("cliff", 0)).label()
    for s in promo.segments:
        ep = show.ep(s.episode)
        bs = beats_in(ep, s.start, s.end)
        if bs:
            s.start_beat, s.end_beat = bs[0].id, bs[-1].id
    return promo


def _cliff_at(ep: Episode, t: float, sig: dict) -> float:
    ending = [b for b in ep.beats if abs(b.end - t) <= 3.0 and b.start < t]
    return max((sig.get(b.id, {}).get("cliff", 0) for b in ending), default=0.0)


def overlap_ratio(a: Promo, b: Promo) -> float:
    """Shared screen time / shorter promo's duration."""
    shared = 0.0
    for sa in a.segments:
        for sb in b.segments:
            if sa.episode == sb.episode:
                shared += max(0.0, min(sa.end, sb.end) - max(sa.start, sb.start))
    d = min(a.duration, b.duration) or 1.0
    return shared / d
