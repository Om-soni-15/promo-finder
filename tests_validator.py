"""Feed the validator deliberately bad LLM answers; every one must be fixed or rejected with a reason."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from promofinder.parse import parse_show
from promofinder.signals import annotate
from promofinder.pipeline import Run
import yaml

show = parse_show(sys.argv[1], {1, 2, 3})
cfg = yaml.safe_load(open(Path(__file__).parent / "config.yaml", encoding="utf-8"))
run = Run(show, cfg, Path("/tmp/vt"))
cases = {
    "invented beat id":            {"segments": [{"start_beat": "E01-B999", "end_beat": "E01-B050"}]},
    "segment spans 2 episodes":    {"segments": [{"start_beat": "E01-B080", "end_beat": "E02-B010"}]},
    "way too long (whole ep)":     {"segments": [{"start_beat": "E01-B000", "end_beat": "E01-B094"}]},
    "too short (4 s)":             {"segments": [{"start_beat": "E01-B027", "end_beat": "E01-B030"}]},
    "hook 20 s after start":       {"segments": [{"start_beat": "E01-B010", "end_beat": "E01-B060"}], "hook_beat": "E01-B045"},
    "end mid-dialogue (camera id)":{"segments": [{"start_beat": "E01-B030", "end_beat": "E01-B055"}]},
    "overlapping segments":        {"segments": [{"start_beat": "E01-B020", "end_beat": "E01-B040"}, {"start_beat": "E01-B030", "end_beat": "E01-B050"}]},
    "flash-forward order":         {"segments": [{"start_beat": "E03-B060", "end_beat": "E03-B066"}, {"start_beat": "E03-B010", "end_beat": "E03-B030"}]},
    "good single":                 {"segments": [{"start_beat": "E01-B027", "end_beat": "E01-B060"}], "hook_beat": "E01-B029"},
}
for name, item in cases.items():
    p = run._promo_from_llm(item, "llm-episode")
    if p is None:
        print(f"{name:30} -> REJECTED: {run.rejected[-1].reject_reason}")
        continue
    ok = run._validate(p)
    if not ok:
        print(f"{name:30} -> REJECTED: {p.reject_reason}")
    else:
        segs = " + ".join(f"E{s.episode} {s.start:.1f}-{s.end:.1f}" for s in p.segments)
        print(f"{name:30} -> OK {p.duration:.1f}s [{segs}] fixes={p.notes} flags={p.flags}")
