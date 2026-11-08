"""Rule-based promo finder: candidate generation + hook / cliffhanger / standalone scoring.

This is the free, offline, deterministic layer. It runs on any show with the same
screenplay format and needs no API key. The optional LLM layer (llm.py) re-judges
its shortlist.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from .parse import Episode, Event, fmt

# ---------------------------------------------------------------- lexicons
# Built from the vocabulary the writers actually use in this pack (emotion tags,
# ACTION / SOUND FX verbs). Weights are hand-set, 0..1.
HOT_EMOTIONS = {
    "screaming": 1.0, "shouting": .9, "hysterical": 1.0, "furious": .9, "enraged": .9, "outraged": .8,
    "venomous": .9, "threatening": .9, "menacing": .9, "chilling": .8, "sinister": .8, "cruel": .8,
    "malicious": .8, "vindictive": .8, "spiteful": .7, "mocking": .7, "taunting": .7, "gloating": .7,
    "scornful": .6, "contemptuous": .6, "derisive": .6, "accusatory": .8, "accusing": .8,
    "desperate": .7, "sobbing": .7, "pleading": .6, "terrified": .8, "panicked": .8, "frantic": .8,
    "shocked": .8, "stunned": .7, "incredulous": .7, "gasping": .7, "heartbroken": .6, "weeping": .6,
    "defiant": .6, "commanding": .5, "cold": .5, "booming": .6, "fierce": .6, "harsh": .6,
    "dismissive": .5, "arrogant": .5, "condescending": .5, "smug": .5, "calculating": .5,
    "callous": .6, "steely": .5, "unyielding": .5, "final": .5, "ominous": .7, "dark": .5,
    "warning": .6, "breathless": .5, "urgent": .6, "demanding": .5, "bitter": .5, "piercing": .6,
}
SOFT_EMOTIONS = {"gentle", "soft", "tender", "warm", "loving", "calm", "reassuring", "curious",
                 "amused", "sincere", "earnest", "reverent", "affectionate", "playful", "content",
                 "relieved", "inquiring", "measured", "thoughtful"}

ACTION_HOT = {
    r"\bslap": 1.0, r"\bpunch": .9, r"\bstrik(e|es|ing)\b": .9, r"whip crack|(strikes?|lashes?|whips?)\b.*\b(crop|whip)": .9,
    r"\bshov": .8, r"\bgrab": .5, r"\bseiz": .8, r"\bdrag": .7, r"\bchain": .7, r"\bshackle": .7,
    r"\bblood": .9, r"\bwound": .6, r"\bbleed": .9, r"\bburn": .7, r"\bblaze|inferno|flames": .8,
    r"explo": 1.0, r"\bcrash": .7, r"kick(s|ed)? (open|down)|bursts? (open|in)\b|flies open|fly open": .9,
    r"\bcollaps": .8, r"\bgasp": .6, r"\brecoil": .6, r"(drops|falls|sinks) to (one|her|his) knee": .9,
    r"\bkneel": .7, r"\bkiss": .7, r"\bpoison|arsenic": .8, r"dagger|knife|sword|blade": .8,
    r"red-hot|hot iron|branding": 1.0, r"\bhurl|\bthrows?\b": .7, r"\btear(s|ing)? (up|apart)|\brips?\b": .7,
    r"\bscream": .8, r"\bchok": .8, r"\bstrangl": .9, r"\bthroat": .7, r"\bgun|pistol|musket": .8,
    r"\bthorn|briar": .6, r"\bdungeon|cell\b": .5, r"\bgallows|noose|execution": .9, r"\bguards? seize": .8,
    r"\bshatter": .7, r"\bsmash": .7, r"\bslam": .6, r"\bfist": .5, r"\bsnap zoom|crash zoom|whip pan": .6,
}
SHOCK_WORDS = {
    r"\bmarr(y|ied|iage)\b": .6, r"\bwife\b|\bhusband\b": .5, r"\bduke\b|\bduchess\b": .5,
    r"\bdead\b|\bdie\b|\bdeath\b": .8, r"\bkill|murder": .9, r"\bhang(ed)?\b|gallows": .9,
    r"\bwitch": .9, r"\bpoison": .9, r"\bthief\b|\bsteal|stole": .7, r"\bliar\b|\blie\b|\blied\b": .6,
    r"\bbetray": .8, r"\bdisown": .9, r"\bnever\b": .4, r"\bburn": .7, r"\barrest": .8,
    r"\bguilty\b": .7, r"\bexecut": .9, r"\bworthless|plain(ness)?\b|\bnobody\b|\bbeggar": .7,
    r"\bpregnan": .9, r"\bmistress\b|\bharlot|whore": .8, r"\btraitor|treason": .8,
    r"\bsister\b": .4, r"\bmother\b": .3, r"\bhow dare": .8, r"\bget out\b": .7, r"\bon your knees\b|\bkneel\b": .8,
    r"\bwho are you\b": .7, r"\bmine\b": .4, r"\bproof\b": .5, r"\bblood\b": .7,
}
CLIFF_SFX = r"cliffhanger|ominous|drone|sting|cut to black|abrupt|sudden silence|dread|tension"
CLIFF_ACTION = (r"bursts? (open|in)\b|door .*?(opens|creaks|bangs)|footsteps|freezes|eyes widen|turns? (slowly )?to face|"
                r"reveal|unmask|stares? in (shock|horror|disbelief)|gasps|falls|collapses|draws? (a|his|her) (sword|blade|dagger)|"
                r"raises (the|his|her) (hand|crop|iron)|lunges|seiz|grabs|hand (hovers|closes)|recognizes|interrupt")
RESOLVE = (r"\bsmil(e|es|ing)\b|embrace|\bhugs?\b|relief|\bwarm(ly|th)?\b|tender|peaceful|together|"
           r"thank you|laugh|sigh(s)? (in|with) relief|fades? (out|to black) .*?(warm|gentle|soft)|gentle (string|music)")
DANGLING = r"^(and|but|so|then|that|it|he|she|they|this|those|these|because|or)\b"
ESTABLISH = r"establishing|wide (aerial|exterior)|exterior of|drone shot|landscape|sky\b"


@dataclass
class Segment:
    episode: int
    start: float
    end: float
    first_idx: int
    last_idx: int

    @property
    def duration(self):
        return self.end - self.start

    def to_dict(self):
        return {"episode": self.episode, "start": fmt(self.start), "end": fmt(self.end),
                "start_sec": round(self.start, 2), "end_sec": round(self.end, 2),
                "duration_sec": round(self.duration, 2)}


@dataclass
class Promo:
    segments: list[Segment]
    hook_score: float
    cliff_score: float
    standalone_score: float
    hook: str
    cliffhanger: str
    why: str
    kind: str = "continuous"
    reasons: dict = field(default_factory=dict)
    source: str = "rules"
    total: float = 0.0
    flags: list[str] = field(default_factory=list)

    @property
    def duration(self):
        return sum(s.duration for s in self.segments)

    @property
    def episodes(self):
        return sorted({s.episode for s in self.segments})

    def to_dict(self):
        return {"episodes": self.episodes, "kind": self.kind, "duration_sec": round(self.duration, 2),
                "segments": [s.to_dict() for s in self.segments], "hook": self.hook,
                "cliffhanger": self.cliffhanger, "why": self.why,
                "scores": {"hook": round(self.hook_score, 3), "cliffhanger": round(self.cliff_score, 3),
                           "standalone": round(self.standalone_score, 3), "total": round(self.total, 3)},
                "source": self.source, "flags": self.flags, "signals": self.reasons}


# ---------------------------------------------------------------- event intensity
def _rx_max(table: dict, text: str) -> tuple[float, str]:
    best, hit = 0.0, ""
    for pat, w in table.items():
        m = re.search(pat, text, re.I)
        if m and w > best:
            best, hit = w, m.group(0)
    return best, hit


MILD_CONTEXT = r"\b(tapping|holding|carrying|cradling|gently|softly|keep (her|him) moving|resting|reassuring(ly)?)\b"


def intensity(ev: Event) -> tuple[float, str]:
    """0..1 'would a scroller stop for this?' score for one event, plus the reason."""
    t = ev.text
    if ev.kind == "dialogue":
        emos = [e.strip().lower() for e in ev.emotion.split(",") if e.strip()]
        emo = max((HOT_EMOTIONS.get(e, 0) for e in emos), default=0)
        soft = any(e in SOFT_EMOTIONS for e in emos)
        word, w = _rx_max(SHOCK_WORDS, t)
        s = 0.55 * emo + 0.45 * word
        if "!" in t:
            s += 0.1
        if re.match(r"^you\b", t, re.I) and emo >= .5:
            s += 0.1                      # direct accusation
        if len(t.split()) <= 9 and emo >= .6:
            s += 0.05                     # short, punchy line
        if soft:
            s -= 0.2
        why = f'{ev.speaker} ({ev.emotion}) "{t}"'
        return max(0.0, min(1.0, s)), why
    if ev.kind in ("action", "sfx"):
        s, hit = _rx_max(ACTION_HOT, t)
        if s and re.search(MILD_CONTEXT, t, re.I):
            s *= 0.5                      # "tapping her riding crop" is not a strike
        if ev.kind == "sfx" and re.search(r"gasp|slap|crack|scream|crash|explosion|whip", t, re.I):
            s = max(s, .8)
        return s * (0.9 if ev.kind == "sfx" else 1.0), t
    # camera: only aggressive camera moves count, establishing shots are negative
    s, _ = _rx_max({r"snap zoom|crash zoom|whip pan|smash cut": .5}, t)
    if re.search(ESTABLISH, t, re.I):
        s = -0.3
    return s, t


# ---------------------------------------------------------------- cut points
def cut_points(ep: Episode) -> tuple[list[float], list[float]]:
    """Legal starts/ends = camera cuts that don't slice through a spoken line."""
    cams = [e for e in ep.events if e.kind == "camera"]
    dlg = [e for e in ep.events if e.kind == "dialogue"]

    def clean(t):
        return not any(d.start + 0.15 < t < d.end - 0.15 for d in dlg)

    starts = sorted({c.start for c in cams if clean(c.start)})
    ends = sorted({c.end for c in cams if clean(c.end)})
    return starts, ends


def events_in(ep: Episode, a: float, b: float) -> list[Event]:
    return [e for e in ep.events if e.start >= a - 0.05 and e.start < b - 0.05]


# ---------------------------------------------------------------- component scores
def hook_score(ep: Episode, a: float, hook_window=3.0):
    evs = [e for e in ep.events if a - 0.05 <= e.start < a + hook_window]
    if not evs:
        return 0.0, "", {}
    scored = [(intensity(e), e) for e in evs]
    (best, why), bev = max(scored, key=lambda x: x[0][0])
    s = best
    if any(e.kind == "dialogue" for e in evs):
        s += 0.08                         # a spoken line in the first 3s
    if all(e.kind == "camera" for e in evs) or any(re.search(ESTABLISH, e.text, re.I) for e in evs if e.kind == "camera"):
        s -= 0.15
    first_dlg = next((e for e in evs if e.kind == "dialogue"), None)
    if first_dlg and re.match(DANGLING, first_dlg.text, re.I):
        s -= 0.1
    return max(0.0, min(1.0, s)), why, {"hook_event": bev.idx, "hook_t": fmt(bev.start)}


def cliff_score(ep: Episode, b: float, window=5.0):
    evs = [e for e in ep.events if b - window <= e.end <= b + 0.05 or (e.start < b and e.end > b - window)]
    evs = [e for e in evs if e.start < b]
    if not evs:
        return 0.0, "", {}
    s, sig = 0.0, []
    text = " ".join(e.text for e in evs)
    dl = [e for e in evs if e.kind == "dialogue"]
    last_line = dl[-1] if dl else None
    if re.search(CLIFF_SFX, " ".join(e.text for e in evs if e.kind in ("sfx", "camera")), re.I):
        s += 0.35; sig.append("ominous/cliffhanger cue")
    if re.search(CLIFF_ACTION, " ".join(e.text for e in evs if e.kind in ("action", "camera", "sfx")), re.I):
        s += 0.25; sig.append("unresolved action")
    if last_line:
        lt = last_line.text
        if lt.rstrip().endswith("?"):
            s += 0.25; sig.append("unanswered question")
        if re.search(r"(\.\.\.|…|—|-)$", lt.rstrip('"” ')):
            s += 0.2; sig.append("line cut off")
        li, _ = intensity(last_line)
        s += 0.3 * li
        if li > .6:
            sig.append("threat/shock line")
    peak = max(intensity(e)[0] for e in evs)
    s += 0.2 * peak
    if re.search(RESOLVE, text, re.I):
        s -= 0.3; sig.append("PENALTY resolution/tender beat")
    if abs(b - ep.duration) < 1.5:
        s += 0.15; sig.append("episode-ending cliffhanger")
    desc = (f'{last_line.speaker}: "{last_line.text}"' if last_line and last_line.end >= b - 4
            else max(evs, key=lambda e: intensity(e)[0]).text)
    return max(0.0, min(1.0, s)), desc, {"cliff_signals": sig}


def standalone_score(ep: Episode, a: float, b: float, episode_index: int):
    evs = events_in(ep, a, b)
    dl = [e for e in evs if e.kind == "dialogue"]
    speakers = {e.speaker.replace("LADY ", "").replace("LORD ", "") for e in dl}
    s, sig = 0.6, []
    if len(dl) < 3:
        s -= 0.3; sig.append("PENALTY too little dialogue")
    if len(speakers) >= 2:
        s += 0.15; sig.append("clear two-sided conflict")
    if len(speakers) > 4:
        s -= 0.1 * (len(speakers) - 4); sig.append("PENALTY many characters")
    if dl and re.match(DANGLING, dl[0].text, re.I):
        s -= 0.2; sig.append("PENALTY opens mid-thought")
    alltext = " ".join(e.text for e in dl)
    if re.search(r"\b(sister|husband|wife|mother|father|fianc|duke|duchess|maid|stepmother|son)\b", alltext, re.I):
        s += 0.1; sig.append("relationship stated in dialogue")
    if re.search(r"\b(remember|again|earlier|as I said|last night|yesterday|the ring|that book)\b", alltext, re.I):
        s -= 0.1; sig.append("PENALTY leans on earlier context")
    hot = sum(1 for e in dl if intensity(e)[0] > .55)
    s += min(0.15, 0.05 * hot)
    return max(0.0, min(1.0, s)), {"standalone_signals": sig, "speakers": sorted(speakers)}


# ---------------------------------------------------------------- generation
W_HOOK, W_CLIFF, W_STAND = 0.40, 0.35, 0.25
MIN_LEN, MAX_LEN, IDEAL = 20.0, 60.0, (28.0, 50.0)


def length_penalty(d):
    lo, hi = IDEAL
    if d < lo:
        return 0.02 * (lo - d) / 2
    if d > hi:
        return 0.02 * (d - hi) / 2
    return 0.0


def _clip(txt, n=140):
    return txt if len(txt) <= n else txt[: n - 1].rsplit(" ", 1)[0] + "…"


def make_why(h, c, st_sig):
    hs = "a shock line" if '"' in h else "a physical shock beat"
    cs = c.get("cliff_signals") or ["a tense beat"]
    extra = "two-sided conflict a stranger can read" if "clear two-sided conflict" in st_sig.get("standalone_signals", []) else "a self-contained beat"
    return f"Opens on {hs}, ends on {cs[0]} with {extra}."


def score_window(ep: Episode, a: float, b: float, ei: int) -> Promo | None:
    d = b - a
    if not (MIN_LEN <= d <= MAX_LEN):
        return None
    hs, hdesc, hsig = hook_score(ep, a)
    cs, cdesc, csig = cliff_score(ep, b)
    ss, ssig = standalone_score(ep, a, b, ei)
    evs = events_in(ep, a, b)
    seg = Segment(ep.number, a, b, evs[0].idx if evs else 0, evs[-1].idx if evs else 0)
    p = Promo([seg], hs, cs, ss, _clip(hdesc), _clip(cdesc), make_why(hdesc, csig, ssig),
              reasons={**hsig, **csig, **ssig})
    p.total = W_HOOK * hs + W_CLIFF * cs + W_STAND * ss - length_penalty(d)
    return p


def overlap(p: Promo, q: Promo) -> float:
    ov = 0.0
    for s in p.segments:
        for t in q.segments:
            if s.episode == t.episode:
                ov += max(0.0, min(s.end, t.end) - max(s.start, t.start))
    return ov / min(p.duration, q.duration)


def nms(cands: list[Promo], k: int, max_overlap=0.25) -> list[Promo]:
    keep: list[Promo] = []
    for c in sorted(cands, key=lambda p: -p.total):
        if all(overlap(c, kp) <= max_overlap for kp in keep):
            keep.append(c)
        if len(keep) >= k:
            break
    return keep


def cold_open_stitch(ep: Episode, conts: list[Promo], ei: int) -> list[Promo]:
    """Flash-forward format common in ReelShort/DramaBox ads: show the episode's most
    explosive 2-8s moment first, then cut back to the build-up and stop *before* it
    resolves. Segment 1 = the peak, segment 2 = an earlier continuous run."""
    starts, ends = cut_points(ep)
    peak = max(ep.events, key=lambda e: intensity(e)[0])
    pi, _ = intensity(peak)
    if pi < 0.75:
        return []
    a1 = max([s for s in starts if s <= peak.start + 0.05] or [None]) if starts else None
    if a1 is None:
        return []
    reply = [e.start for e in ep.events if e.kind == "dialogue" and e.start > peak.start + 0.3
             and (peak.kind != "dialogue" or e.speaker != peak.speaker)]
    # A spoken peak ("You are disowned!") must not include the reply that defuses it.
    # A physical peak (a punch) can carry the shout that follows it.
    stop = reply[0] + 0.15 if (reply and peak.kind == "dialogue") else 1e9
    b1s = [e for e in ends if 2.0 <= e - a1 <= 8.0 and e <= stop]
    if not b1s:
        return []
    b1 = b1s[0] if len(b1s) == 1 else min(b1s, key=lambda e: abs((e - a1) - 5))
    out = []
    for a2 in starts:
        for b2 in ends:
            if b2 > a1 + 0.05 or b2 <= a2:
                continue
            tot = (b1 - a1) + (b2 - a2)
            if not (MIN_LEN <= tot <= MAX_LEN) or b2 - a2 < 12:
                continue
            hs, hdesc, hsig = hook_score(ep, a1)
            cs, cdesc, csig = cliff_score(ep, b2)
            ss, ssig = standalone_score(ep, a2, b2, ei)
            cs = min(1.0, cs + 0.1)       # viewer saw the payoff but not how it lands
            e1, e2 = events_in(ep, a1, b1), events_in(ep, a2, b2)
            if not e1 or not e2:
                continue
            segs = [Segment(ep.number, a1, b1, e1[0].idx, e1[-1].idx),
                    Segment(ep.number, a2, b2, e2[0].idx, e2[-1].idx)]
            p = Promo(segs, hs, cs, ss, _clip(hdesc), _clip(cdesc),
                      "Flash-forward: opens on the episode's biggest beat, then rewinds and cuts before it lands.",
                      kind="cold-open stitch", reasons={**hsig, **csig, **ssig})
            p.total = W_HOOK * hs + W_CLIFF * cs + W_STAND * ss - length_penalty(tot) - 0.03  # stitch risk
            out.append(p)
    return nms(out, 1)


MIN_TOTAL = 0.50   # below this a clip is not worth an editor's time; say so instead


def find_promos_in_episode(ep: Episode, ei: int, per_episode=3, pool=12, min_total=MIN_TOTAL) -> tuple[list[Promo], list[Promo]]:
    """Returns (final picks above the quality floor, wider candidate pool for the LLM)."""
    starts, ends = cut_points(ep)
    cands = []
    for a in starts:
        for b in ends:
            if b - a < MIN_LEN:
                continue
            if b - a > MAX_LEN:
                break
            p = score_window(ep, a, b, ei)
            if p:
                cands.append(p)
    cands += cold_open_stitch(ep, cands, ei)
    picks = [p for p in nms(cands, per_episode) if p.total >= min_total]
    return picks, nms(cands, pool)
