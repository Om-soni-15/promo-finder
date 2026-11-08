"""Cheap, explainable text signals for hooks and cliffhangers.

These rules never decide alone in LLM mode — they (a) pre-select moments so the
LLM sees a short menu, (b) act as a no-API fallback, and (c) break ties.
Every score comes with the reasons that produced it, so a human can audit it.
"""
from __future__ import annotations

import re
from .parse import Beat, Episode

# --- lexicons (show-agnostic; tuned for melodrama microdramas) -------------
STRONG_EMOTIONS = {
    "outraged", "furious", "enraged", "livid", "screaming", "shrieking", "roaring",
    "gasping", "horrified", "terrified", "panicked", "hysterical", "sadistic",
    "murderous", "venomous", "explosive", "savage", "seething", "unhinged",
    "shocked", "stunned", "devastated", "agonized", "anguished", "frantic",
}
MEDIUM_EMOTIONS = {
    "cold", "icy", "menacing", "threatening", "cruel", "gloating", "smug", "mocking",
    "sneering", "accusing", "commanding", "sinister", "defiant", "desperate",
    "contemptuous", "disdainful", "triumphant", "dangerous", "lethal", "ominous",
    "vicious", "merciless", "taunting", "condescending", "arrogant", "fierce",
    "hostile", "bitter", "scornful", "imperious", "chilling", "deadly", "steely",
    "dismissive", "humiliated", "heartbroken", "pleading", "unbelieving", "shaken",
}
CALM_EMOTIONS = {
    "soft", "softly", "tender", "gentle", "gently", "warm", "relieved", "peaceful",
    "loving", "smiling", "content", "fond", "affectionate", "grateful", "calm",
    "reassuring", "soothing", "playful", "amused", "happy", "joyful",
}

ACTION_STRONG = [
    r"\bslap", r"\bstrik", r"\bstruck\b", r"\bpunch", r"\bbackhand", r"\briding crop",
    r"\bwhip", r"\blash", r"\bstab", r"\bblade\b", r"\bsword", r"\bdagger", r"\bknife",
    r"\bblood", r"\bbleed", r"\bflames?\b", r"\bblaze", r"\binferno", r"\bon fire\b",
    r"\bexplo", r"\bshatter", r"\bsmash", r"\bseiz", r"\bgrabs?\b", r"\bgrabbed",
    r"\bdrag", r"\bshove", r"\bhurl", r"\bthrows?\b", r"\bslams?\b", r"\bcollaps",
    r"\bchok", r"\bstrangl", r"\bchain", r"\bshackl", r"\bnoose", r"\bgallows",
    r"\bone knee\b", r"\bkneels?\b", r"\bkiss", r"\brips?\b", r"\btears? (?:off|open)",
    r"\bred-hot\b", r"\bbrand(?:ing|ed)?\b", r"\bpoison", r"\barsenic", r"\bpistol",
    r"\bbursts? (?:in|through|open)", r"\bstorms? in", r"\bfaints?\b", r"\bunconscious",
    r"\bscar(?:s|red)?\b", r"\bweapon",
]
REVEAL_WORDS = [
    r"\bduke\b", r"\bduchess\b", r"\bmarry", r"\bmarried\b", r"\bwife\b", r"\bhusband\b",
    r"\bpregnan", r"\balive\b", r"\bdead\b", r"\bdie\b", r"\bkill", r"\bhang",
    r"\bpoison", r"\bwitch", r"\bburn", r"\bbetray", r"\bliar\b", r"\bthief\b",
    r"\bmy son\b", r"\bhow dare\b", r"\byou knew\b", r"\bwho are you\b", r"\bdisown",
    r"\bworthless\b", r"\bbeneath\b", r"\bplain(?:ness)?\b", r"\bmine\b", r"\barrest",
    r"\bguilty\b", r"\bexecut", r"\binvalid\b", r"\bnever\b", r"\btruth\b", r"\blie\b",
    r"\bheir\b", r"\btitle\b", r"\bdungeon", r"\bmurder",
]
THREAT_WORDS = [
    r"\bor (?:else|watch|you)\b", r"\bunless\b", r"\bwill pay\b", r"\byou will\b",
    r"\byou'll\b", r"\bwatch (?:him|her|them|you)\b", r"\bhang\b", r"\bdie\b",
    r"\btomorrow\b", r"\buntil\b", r"\bbefore (?:dawn|sunrise|morning)\b", r"\bend you\b",
    r"\bregret\b", r"\bnext\b", r"\bfinished\b", r"\bburn\b",
]
SFX_HOOK = [
    r"gasp", r"crack", r"slap", r"thud", r"smash", r"shatter", r"boom", r"explo",
    r"bang", r"sting", r"scream", r"clang", r"roar", r"gunshot", r"thunder", r"crash",
    r"whoosh", r"impact", r"hit\b", r"jolt",
]
SFX_CLIFF = [
    r"cliffhanger", r"ominous", r"swell", r"peak", r"sting", r"suspense", r"heartbeat",
    r"drone", r"cut to black", r"silence", r"rumbl", r"tension",
]
SFX_RESOLVE = [
    r"triumphant", r"warm", r"gentle", r"tender", r"peaceful", r"resolv", r"hopeful",
    r"uplifting", r"romantic", r"soft", r"soothing", r"serene", r"joyful", r"sweet",
    r"fad(?:e|es|ing) (?:out|away|to black)", r"melanchol", r"gently",
]
CAMERA_HOOK = [r"snap zoom", r"crash zoom", r"whip pan", r"slow[- ]motion", r"extreme close"]
CAMERA_CLIFF = [r"cut to black", r"freeze", r"smash cut", r"fade to black", r"hard cut"]


def _hits(patterns, text):
    t = text.lower()
    return [p for p in patterns if re.search(p, t)]


def _emotion_words(beat: Beat) -> set[str]:
    if not beat.emotion:
        return set()
    return {w.strip().lower() for w in re.split(r"[,/;]| and ", beat.emotion) if w.strip()} | {
        w.lower() for w in re.findall(r"[A-Za-z]+", beat.emotion)
    }


def hook_score(beat: Beat) -> tuple[float, list[str]]:
    """How likely is this beat to stop a scrolling viewer?"""
    score, why = 0.0, []
    t = beat.text
    if beat.type == "dialogue":
        emo = _emotion_words(beat)
        s = emo & STRONG_EMOTIONS
        m = emo & MEDIUM_EMOTIONS
        if s:
            score += 2.0; why.append(f"strong emotion ({', '.join(sorted(s))})")
        elif m:
            score += 1.0; why.append(f"charged delivery ({', '.join(sorted(m))})")
        rv = _hits(REVEAL_WORDS, t)
        if rv:
            score += min(1.5, 0.75 * len(rv)); why.append("reveal/stakes words")
        if t.rstrip().endswith("!"):
            score += 0.5; why.append("exclamation")
        nwords = len(t.split())
        if nwords <= 7 and (s or m or rv):
            score += 0.5; why.append("short punchy line")
    elif beat.type == "action":
        a = _hits(ACTION_STRONG, t)
        if a:
            score += min(3.0, 1.5 + 0.5 * (len(a) - 1)); why.append("physical/visual shock")
    elif beat.type == "sfx":
        if _hits(SFX_HOOK, t):
            score += 1.5; why.append("impact sound cue")
    elif beat.type == "camera":
        a = _hits(ACTION_STRONG, t)
        if a:
            score += 1.0; why.append("shocking image in shot")
        if _hits(CAMERA_HOOK, t):
            score += 0.5; why.append("punch-in camera move")
    return score, why


def cliff_score(beat: Beat, episode: Episode) -> tuple[float, list[str]]:
    """How good is it to CUT right after this beat?"""
    score, why = 0.0, []
    t = beat.text
    if beat.type == "sfx" and _hits(SFX_RESOLVE, t):
        score -= 1.5; why.append("music resolves (warm/triumphant) — not a cliffhanger")
    elif beat.type == "sfx" and _hits(SFX_CLIFF, t):
        score += 2.5; why.append("score/sound swells to a cliffhanger")
    if beat.type == "camera" and _hits(CAMERA_CLIFF, t):
        score += 2.0; why.append("scripted hard cut")
    if beat.type == "dialogue":
        tt = t.rstrip()
        if tt.endswith("?"):
            score += 1.0; why.append("ends on a question")
        if tt.endswith("...") or tt.endswith("…"):
            score += 1.0; why.append("line trails off")
        if tt.endswith("!"):
            score += 0.5
        if _hits(THREAT_WORDS, t):
            score += 1.0; why.append("threat/ultimatum")
        emo = _emotion_words(beat)
        if emo & (STRONG_EMOTIONS | MEDIUM_EMOTIONS):
            score += 0.5
        if emo & CALM_EMOTIONS:
            score -= 1.0; why.append("calm delivery (feels resolved)")
    if beat.type == "action" and _hits(ACTION_STRONG, t):
        score += 1.0; why.append("cuts on an action peak")
    # Microdrama episodes are written to end on a cliffhanger.
    dist_to_end = episode.duration - beat.end
    if dist_to_end <= 6.0:
        score += 2.0; why.append("episode's own ending cliffhanger")
    elif dist_to_end <= 15.0:
        score += 0.75
    return score, why


def annotate(episode: Episode) -> dict[str, dict]:
    """Per-beat {hook, hook_why, cliff, cliff_why}."""
    out = {}
    for b in episode.beats:
        h, hw = hook_score(b)
        c, cw = cliff_score(b, episode)
        out[b.id] = {"hook": round(h, 2), "hook_why": hw, "cliff": round(c, 2), "cliff_why": cw}
    return out
