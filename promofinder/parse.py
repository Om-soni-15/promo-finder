"""Parse Kuku-style timestamped screenplays into structured "beats".

A beat is one block of the script: a CAMERA shot, an ACTION, a SOUND FX cue or a
line of dialogue. Every beat gets a stable ID like ``E01-B023`` so that LLMs can
refer to moments by ID instead of inventing timestamps.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

TIME_RE = r"(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)"
HEADER_RE = re.compile(r"^\*\*(?P<body>.+?)\s*:\s*\*\*\s*$")
STARTEND_RE = re.compile(
    r"^\s*Start:\s*\(?\s*" + TIME_RE + r"\s*\)?\s*\|\s*End:\s*\(?\s*" + TIME_RE + r"\s*\)?\s*$",
    re.I,
)
META_RE = re.compile(r"^(EPISODE NUMBER|EPISODE LOGLINE|CONTINUITY NOTES)\s*:\s*(.*)$", re.I)
SCENE_RE = re.compile(r"^SCENE\s+(\d+)\s*$", re.I)
SCENE_FIELD_RE = re.compile(r"^SCENE\s+(HEADING|START|END)\s*:\s*(.*)$", re.I)

TYPE_MAP = {"CAMERA": "camera", "ACTION": "action", "SOUND FX": "sfx", "SFX": "sfx"}


def parse_time(s: str) -> float:
    """'02:18.50' -> 138.5 ; '1:02:03.5' -> 3723.5"""
    parts = s.strip().split(":")
    secs = 0.0
    for p in parts:
        secs = secs * 60 + float(p)
    return round(secs, 3)


def fmt_time(t: float) -> str:
    m, s = divmod(max(t, 0.0), 60)
    return f"{int(m):02d}:{s:05.2f}"


@dataclass
class Beat:
    id: str
    episode: int
    scene: int
    idx: int
    type: str            # camera | action | sfx | dialogue
    speaker: str | None  # for dialogue, raw header name e.g. "LADY JULIANA"
    emotion: str | None  # e.g. "Gasping, Devastated"
    start: float
    end: float
    text: str

    def label(self) -> str:
        if self.type == "dialogue":
            emo = f" ({self.emotion})" if self.emotion else ""
            return f"{self.speaker}{emo}: {self.text}"
        return f"[{self.type.upper()}] {self.text}"


@dataclass
class Scene:
    number: int
    heading: str
    start: float
    end: float


@dataclass
class Episode:
    number: int
    title: str
    logline: str
    continuity: str
    file: str
    scenes: list[Scene] = field(default_factory=list)
    beats: list[Beat] = field(default_factory=list)
    parse_warnings: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        ends = [s.end for s in self.scenes] + [b.end for b in self.beats]
        return max(ends) if ends else 0.0

    def beat(self, beat_id: str) -> Beat | None:
        return self._index().get(beat_id)

    def _index(self):
        if not hasattr(self, "_idx_cache"):
            self._idx_cache = {b.id: b for b in self.beats}
        return self._idx_cache


def _split_header(body: str) -> tuple[str, str | None]:
    m = re.match(r"^(?P<name>.*?)\s*(?:\((?P<emo>[^)]*)\))?\s*$", body)
    name = (m.group("name") if m else body).strip()
    emo = (m.group("emo") or "").strip() if m else ""
    return name, (emo or None)


def parse_episode(path: Path) -> Episode:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    ep_num, title, logline, continuity = None, "", "", ""
    scenes: list[Scene] = []
    raw_beats: list[dict] = []
    warnings: list[str] = []

    cur_scene = 0
    pending: dict | None = None  # beat being built

    def flush():
        nonlocal pending
        if pending is not None:
            if pending.get("start") is None:
                warnings.append(f"block '{pending['header']}' has no Start/End line; skipped")
            else:
                pending["text"] = " ".join(t.strip() for t in pending["text"] if t.strip())
                raw_beats.append(pending)
        pending = None

    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("--- SCRIPT"):
            continue
        m = META_RE.match(s)
        if m and not pending:
            key, val = m.group(1).upper(), m.group(2).strip()
            if key == "EPISODE NUMBER":
                title = val
                n = re.search(r"\d+", val)
                ep_num = int(n.group()) if n else None
            elif key == "EPISODE LOGLINE":
                logline = val
            else:
                continuity = val
            continue
        m = SCENE_RE.match(s)
        if m:
            flush()
            cur_scene = int(m.group(1))
            scenes.append(Scene(cur_scene, "", 0.0, 0.0))
            continue
        m = SCENE_FIELD_RE.match(s)
        if m and scenes:
            key, val = m.group(1).upper(), m.group(2).strip()
            if key == "HEADING":
                scenes[-1].heading = val
            elif key == "START":
                scenes[-1].start = parse_time(val)
            else:
                scenes[-1].end = parse_time(val)
            continue
        if re.match(r"^ACTIONS AND DIALOGUE\s*:\s*$", s, re.I):
            continue
        m = HEADER_RE.match(s)
        if m:
            flush()
            name, emo = _split_header(m.group("body"))
            btype = TYPE_MAP.get(name.upper(), "dialogue")
            pending = {
                "header": name, "type": btype,
                "speaker": name if btype == "dialogue" else None,
                "emotion": emo, "scene": cur_scene,
                "start": None, "end": None, "text": [],
            }
            continue
        m = STARTEND_RE.match(s)
        if m and pending is not None and pending["start"] is None:
            pending["start"], pending["end"] = parse_time(m.group(1)), parse_time(m.group(2))
            continue
        if pending is not None:
            pending["text"].append(s)
        else:
            warnings.append(f"orphan line ignored: {s[:60]}")
    flush()

    if ep_num is None:
        n = re.search(r"episode-(\d+)", path.name)
        ep_num = int(n.group(1)) if n else 0
        warnings.append("episode number not found in header; taken from filename")

    # stable ordering: by start time, then original order
    raw_beats = [dict(b, _o=i) for i, b in enumerate(raw_beats)]
    raw_beats.sort(key=lambda b: (b["start"], b["_o"]))
    beats = []
    for i, b in enumerate(raw_beats):
        text_ = b["text"].strip().strip('"').strip("“”").strip() if b["type"] == "dialogue" else b["text"]
        if b["end"] < b["start"]:
            warnings.append(f"beat {i} end<start; swapped")
            b["start"], b["end"] = b["end"], b["start"]
        beats.append(Beat(
            id=f"E{ep_num:02d}-B{i:03d}", episode=ep_num, scene=b["scene"], idx=i,
            type=b["type"], speaker=b["speaker"], emotion=b["emotion"],
            start=b["start"], end=b["end"], text=text_,
        ))
    return Episode(ep_num, title, logline, continuity, path.name, scenes, beats, warnings)


@dataclass
class Show:
    name: str
    episodes: list[Episode]

    def ep(self, n: int) -> Episode | None:
        for e in self.episodes:
            if e.number == n:
                return e
        return None

    def beat(self, beat_id: str) -> Beat | None:
        m = re.match(r"^E(\d+)-B\d+$", beat_id or "")
        if not m:
            return None
        e = self.ep(int(m.group(1)))
        return e.beat(beat_id) if e else None

    def speakers(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.episodes:
            for b in e.beats:
                if b.speaker:
                    counts[b.speaker] = counts.get(b.speaker, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def parse_show(folder: str | Path, episodes: set[int] | None = None) -> Show:
    folder = Path(folder)
    files = sorted(p for p in folder.rglob("*.txt") if p.is_file())
    if not files:
        raise FileNotFoundError(f"no .txt screenplay files found under {folder}")
    eps = []
    for f in files:
        e = parse_episode(f)
        if episodes is None or e.number in episodes:
            eps.append(e)
    eps.sort(key=lambda e: e.number)
    name = folder.name if folder.name else "show"
    # if the zip had a single subfolder, use that as the show name
    subdirs = {f.parent for f in files}
    if len(subdirs) == 1:
        name = next(iter(subdirs)).name
    return Show(name, eps)


def show_to_dict(show: Show) -> dict:
    return {
        "show": show.name,
        "episodes": [
            {
                "number": e.number, "title": e.title, "logline": e.logline,
                "continuity": e.continuity, "file": e.file, "duration": e.duration,
                "warnings": e.parse_warnings,
                "scenes": [asdict(s) for s in e.scenes],
                "beats": [asdict(b) for b in e.beats],
            }
            for e in show.episodes
        ],
    }
