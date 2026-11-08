#!/usr/bin/env python3
"""Find promo clips in a folder (or .zip) of timestamped microdrama screenplays.

Examples
  python run.py --input scripts.zip                         # rules only, no API key
  python run.py --input scripts/ --mode llm                 # Gemini (needs GEMINI_API_KEY)
  python run.py --input scripts/ --mode llm --provider openai \
      --base-url https://api.groq.com/openai/v1 --model llama-3.3-70b-versatile --api-key-env GROQ_API_KEY
  python run.py --input scripts/ --episodes 1-10 --gold gold.yaml
"""
from __future__ import annotations

import argparse
import copy
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

from promofinder.parse import parse_show, show_to_dict  # noqa: E402
from promofinder.pipeline import Run  # noqa: E402
from promofinder.report import write_all  # noqa: E402


def load_dotenv(*paths: Path) -> list[str]:
    """Minimal .env loader (no extra dependency). Real environment variables win.
    Tolerates Windows quirks: BOM from Notepad, CRLF, 'set KEY=' / 'export KEY=' prefixes,
    and a file accidentally saved as '.env.txt'. Returns the files it loaded."""
    import os
    loaded = []
    seen = set()
    for p in paths:
        for cand in (p, p.with_name(".env.txt")):
            rp = cand.resolve()
            if rp in seen or not cand.is_file():
                continue
            seen.add(rp)
            loaded.append(str(cand))
            raw = cand.read_bytes()
            enc = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
            for line in raw.decode(enc, errors="replace").splitlines():
                line = line.strip().lstrip("\ufeff")
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                for pre in ("export ", "set "):
                    if k.lower().startswith(pre):
                        k = k[len(pre):].strip()
                v = v.strip().strip('"').strip("'").strip()
                if k and v and not os.environ.get(k):
                    os.environ[k] = v
    return loaded


def list_gemini_models(key: str, llm_cfg: dict) -> str:
    """Ask Google which generateContent models this key can use (free call)."""
    import requests
    base = llm_cfg.get("base_url", "https://generativelanguage.googleapis.com/v1beta")
    try:
        r = requests.get(f"{base}/models", headers={"x-goog-api-key": key}, params={"pageSize": 200}, timeout=30)
        models = [m["name"].split("/")[-1] for m in r.json().get("models", [])
                  if "generateContent" in m.get("supportedGenerationMethods", [])]
    except Exception as ex:  # noqa: BLE001
        return f"(could not list models: {ex})"
    flash = [m for m in models if "flash" in m and "image" not in m and "tts" not in m and "live" not in m]
    show = flash or models
    return ("Models your key can use: " + ", ".join(show[:12]) +
            f"\n  Re-run with e.g.  --model {show[0]}" if show else "No models returned for this key.")


def preflight(cfg: dict, env_files: list[str]) -> None:
    """In llm mode, stop BEFORE the run if the key is missing or the API rejects it."""
    import os
    from promofinder.llm import make_llm, LLMError
    llm_cfg = cfg["llm"]
    if llm_cfg.get("provider") == "mock":
        return
    key_env = llm_cfg.get("api_key_env") or {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY",
                                             "anthropic": "ANTHROPIC_API_KEY"}.get(llm_cfg.get("provider"), "")
    base = str(llm_cfg.get("base_url", ""))
    local = "localhost" in base or "127.0.0.1" in base
    if key_env and not os.environ.get(key_env) and not local:
        where = ", ".join(env_files) if env_files else "none found"
        sys.exit(
            f"\nERROR: {key_env} is not set, so the LLM cannot be called.\n"
            f"  .env files loaded: {where}\n"
            f"  Fix: create a file named exactly .env (not .env.txt) in {Path.cwd()} containing one line:\n"
            f"      {key_env}=your-key-here\n"
            f"  Windows Command Prompt:  echo {key_env}=your-key-here> .env\n"
            f"  Windows PowerShell:      Set-Content .env \"{key_env}=your-key-here\"\n"
            f"  Mac/Linux:               echo '{key_env}=your-key-here' > .env\n"
            f"  (or skip .env and set it for this session: PowerShell  $env:{key_env}=\"your-key\" ;"
            f"  cmd  set {key_env}=your-key ; Mac/Linux  export {key_env}=your-key)")
    print(f"Checking {llm_cfg.get('provider')}/{llm_cfg.get('model')} ...", end=" ", flush=True)
    test = make_llm(dict(llm_cfg, cache=False, retries=1, min_interval_sec=0))
    try:
        test.complete("Reply with JSON only.", 'Return {"ok": true}', task="preflight")
    except LLMError as e:
        hint = ""
        if llm_cfg.get("provider") == "gemini" and ("404" in str(e) or "NOT_FOUND" in str(e)):
            hint = "\n  " + list_gemini_models(os.environ.get(key_env, ""), llm_cfg)
        sys.exit(f"failed.\nERROR: the API call did not work, so the run was stopped.\n  {str(e)[:500]}{hint}\n"
                 "  Check the key is correct and the model name exists for your account "
                 "(e.g. --model gemini-3.8-flash or gemini-3.5-flash-lite).")
    print("ok")


def parse_range(s: str | None):
    if not s:
        return None
    out = set()
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out |= set(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def main():
    env_files = load_dotenv(Path.cwd() / ".env", Path(__file__).resolve().parent / ".env")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="folder or .zip of episode .txt screenplays")
    ap.add_argument("--out", default="outputs/latest", help="output folder")
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--mode", choices=["rules", "llm"])
    ap.add_argument("--provider", choices=["gemini", "openai", "anthropic", "mock"])
    ap.add_argument("--model")
    ap.add_argument("--base-url")
    ap.add_argument("--api-key-env")
    ap.add_argument("--min-interval", type=float, help="seconds between LLM calls (rate limit)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--episodes", help="subset, e.g. 1-10 or 1,3,5")
    ap.add_argument("--gold", help="YAML of hand-picked promos to evaluate against")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    if a.mode: cfg["mode"] = a.mode
    for k, v in (("provider", a.provider), ("model", a.model), ("base_url", a.base_url),
                 ("api_key_env", a.api_key_env), ("min_interval_sec", a.min_interval)):
        if v is not None:
            cfg["llm"][k] = v
    if a.provider == "mock" and a.min_interval is None:
        cfg["llm"]["min_interval_sec"] = 0
    if a.no_cache:
        cfg["llm"]["cache"] = False

    if cfg.get("mode") == "llm":
        preflight(cfg, env_files)

    src = Path(a.input)
    if not src.exists():
        sys.exit(f"ERROR: --input not found: {src.resolve()}\n  Tip: put the zip next to run.py as scripts.zip, "
                 "or wrap a path containing spaces in double quotes.")
    tmp = None
    if src.suffix.lower() == ".zip":
        tmp = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(src) as z:
            for m in z.infolist():
                if m.filename.lower().endswith(".txt") and ".." not in m.filename and not m.filename.startswith("/"):
                    z.extract(m, tmp.name)
        src = Path(tmp.name)

    show = parse_show(src, parse_range(a.episodes))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import json
    (out / "parsed_show.json").write_text(json.dumps(show_to_dict(show), indent=1, ensure_ascii=False), encoding="utf-8")
    warn = [w for e in show.episodes for w in e.parse_warnings]
    if warn:
        print(f"parser warnings: {len(warn)} (see parsed_show.json)")

    gold = None
    if a.gold:
        gold = yaml.safe_load(open(a.gold, encoding="utf-8")).get("promos", [])
        if a.episodes:
            keep = parse_range(a.episodes)
            gold = [g for g in gold if all(int(s["episode"]) in keep for s in g["segments"])]

    res = Run(show, cfg, out).execute()
    q = write_all(show, res, cfg, out, gold)
    m = q["metrics"]
    print(f"\nDone in {res['seconds']}s -> {out}/")
    print(f"  {m['valid_promos']} promos, {m['episodes_covered']}/{len(show.episodes)} episodes covered, "
          f"{m['rejected']} rejected by validator")
    if "gold_recall" in m:
        print(f"  gold recall {m['gold_recall']:.0%}, {m['gold_in_top20']} gold promos in top 20")
    if res["errors"]:
        print(f"  WARNING: {len(res['errors'])} LLM calls failed (see quality_report.md); "
              "those parts fell back to rules-only results.")
    print("  open report.html, or read post_first.md")
    if tmp:
        tmp.cleanup()


if __name__ == "__main__":
    main()
