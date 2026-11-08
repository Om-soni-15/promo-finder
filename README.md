# promo-finder — find promo clips in microdrama screenplays

Reads a folder (or .zip) of timestamped episode screenplays and returns **ranked promo clips**
(20–60 s, hook in the first 3 s, cliffhanger at the cut, understandable to a stranger), each with
episode + start/end timestamps of every segment, the hook, the cliffhanger, a one-line "why",
a "post these first" list and a quality report that says where the picks are weak.

## Quick start

```bash
pip install -r requirements.txt

# 1) No API key: rules-only baseline (runs in <1 s)
python run.py --input path/to/scripts.zip --out outputs/rules_run --gold gold.yaml

# 2) With an LLM (recommended). Free Gemini key: https://aistudio.google.com/apikey
cp .env.example .env        # then paste your key into .env  (or: export GEMINI_API_KEY=...)
python run.py --input path/to/scripts.zip --out outputs/gemini_run --mode llm --gold gold.yaml
```

**Windows** (Command Prompt, inside the `promo-finder` folder):

```bat
pip install -r requirements.txt
echo GEMINI_API_KEY=your-key-here> .env
python run.py --input scripts.zip --out outputs\gemini_run --mode llm --gold gold.yaml
```

In LLM mode the tool first checks that the key exists and makes one test call. If either fails, it
stops with an explanation instead of quietly falling back to rules-only results. Notepad files saved
as `.env.txt`, and UTF-16 files from PowerShell's `echo >`, are also read.

Swap the model with flags only, no code changes:

```bash
# any OpenAI-compatible endpoint (Groq free tier, OpenRouter, Ollama, an internal gateway)
export GROQ_API_KEY=...
python run.py --input scripts.zip --mode llm --provider openai \
  --base-url https://api.groq.com/openai/v1 --model llama-3.3-70b-versatile --api-key-env GROQ_API_KEY

# Anthropic
python run.py --input scripts.zip --mode llm --provider anthropic --model <model-id>

# paid / internal APIs: remove the free-tier pause between calls
python run.py ... --min-interval 0
```

Useful options: `--episodes 1-10` (subset), `--no-cache`, `--config my.yaml`.
Every LLM reply is cached in `.cache/llm/`, so a re-run is instant and a run interrupted by a
rate limit resumes where it stopped.

## Outputs (`--out` folder)

| File | What it is |
|---|---|
| `post_first.md` | The 10 promos to post first, with hook, cliffhanger, why it works, why post it now, caution flags |
| `report.html` | Same, plus every promo with its exact script transcript. Open in a browser |
| `promos.csv` | One row per promo, for an editor or a spreadsheet |
| `promos.json` | Everything, machine-readable (segments in `mm:ss.ss` and seconds, scores, flags, auto-fixes) |
| `quality_report.md` | Volume, rejections, auto-fixes, flags, rules-vs-judge agreement, gold-set recall, cost |
| `rejected.json` | Every candidate the validator threw out, and why |
| `raw_llm/` | The exact prompt and raw reply of every LLM call (unedited) |
| `parsed_show.json`, `run_meta.json` | Parsed script and run settings, token use and errors |

## How it works

```
scripts ─▶ 1 Parse ─▶ 2 Show context ─▶ 3 Find candidates ─────────▶ 4 Validate & snap ─▶ 5 Judge ─▶ 6 Rank
           code       LLM, 1 call        rules + LLM per 2 episodes     code                 LLM, blind   code
                                         + LLM cross-episode pass
```

1. **Parse (code).** Each script block becomes a *beat* with a stable ID (`E12-B034`), timestamps,
   type (camera/action/sfx/dialogue), speaker and emotion tag. Handles format drift (e.g. Ep 6
   uses upper-case scene headers).
2. **Show context (LLM, once).** Premise plus character list with aliases, because the scripts call
   one person by several names ("MYSTERIOUS BEARDED MAN" = Alexander, "LADY IN GOLD GOWN" = Clarissa).
3. **Find candidates.**
   - *Rules* score every beat for hook signals (emotion tags such as Outraged or Gasping; slaps, strikes,
     fire, chains; reveal words; impact sound effects) and cliffhanger signals (sound cues that swell to
     a cliffhanger, hard cuts, questions, threats, episode endings). Then they search 20–60 s windows.
   - *LLM, per episode:* sees the full beat listing and proposes promos **by beat ID only.
     It never writes timestamps**, so it cannot invent them.
   - *LLM, cross-episode:* gets a menu of ready-made moments (hook bursts and episode endings) and joins
     them across episodes.
4. **Validate & snap (code).** Converts IDs to timestamps. Never cuts mid-line, cuts on shot
   boundaries, enforces 20–60 s, and checks that the hook lands in the first 3 s (trims the lead-in
   if not). Rejects invented IDs, overlaps and anything far too long. Every fix is logged.
5. **Judge (LLM, blind).** The top candidates are scored 1–5 on hook, cliffhanger, standalone and
   intensity by a model that sees **only the clip**, no show context. That is the "stranger
   scrolling TikTok" test.
6. **Rank (code).** 80% judge, 20% rules, then adjustments:
   - +bonus for clips from early episodes (a new viewer who taps through starts at Ep 1);
   - −penalty for late-arc spoilers and for each quality flag;
   - near-duplicates merged;
   - "post first" list made diverse (max 2 per episode, no overlaps).

**Where a human stays in the loop:** an editor reviews `post_first.md` / `report.html`, cuts the
approved ones, and fixes `gold.yaml` over time. Real performance data (3-second hold, completion,
installs per promo) should replace the hand-set ranking weights in `config.yaml`.

## Quality checking

- `gold.yaml`: hand-picked promos. The run reports recall (found with ≥50% overlap), how many land
  in the top 20, and the start-point error in seconds. *The current gold set is a draft picked from the
  scripts. Review it.*
- `python tests_validator.py <scripts folder>`: feeds the validator deliberately broken LLM answers
  (invented IDs, cross-episode spans, 138 s clips, 4 s clips, late hooks, overlaps, flash-forwards)
  and shows each one being fixed or rejected with a reason.
- `quality_report.md` lists the flags where the tool is likely wrong: visual-only hooks it cannot
  verify from text, endings on calm lines, too many characters for a newcomer, late-arc spoilers,
  and judge scores ≤2 on standalone or hook.

**Baseline result (rules only, 50 episodes):** 99 valid promos, 50/50 episodes covered, 7 of 8 draft
gold promos found, but only 2 of them ranked in the top 20, and cut points ~10 s off on average.
The rules find the moments but cannot judge them, which is why the LLM finder and judge exist.

## Scale and cost

One 50-episode show in LLM mode = **34 calls, ~170k input tokens** (largest call ~11k).
That's cents per show on a Flash-class model and $0 on the Gemini free tier, where it takes about
8 minutes at 5 requests/min. Episodes are independent, so many shows can run in parallel on a paid
tier. Rules mode costs nothing and takes <1 s, so it can pre-screen every new show.

## Known limits

- Works from the script, not the footage. Acting, faces and pacing are invisible.
- Assumes script timestamps match the final edit.
- "Stands alone" is a model imitating a stranger, not real viewers.
- Ranking weights are assumptions until tuned on real promo performance.
- The Gemini free tier may use prompts to improve Google's models. Fine for this assignment's data per
  the brief, but use a paid or internal endpoint for unreleased shows.

## Confidentiality

The input pack is Kuku's content. Do **not** commit scripts, `outputs/` or `.cache/` to a public repo
(`.gitignore` excludes them). Share outputs through a restricted link.
