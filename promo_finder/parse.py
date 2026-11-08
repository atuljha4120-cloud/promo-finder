"""Parse Kuku microdrama screenplays (.txt) into timestamped events.

Each screenplay block looks like:

    **CAMERA:** / **ACTION:** / **SOUND FX:** / **NAME (Emotion, Tone):**
    Start: (MM:SS.ss) | End: (MM:SS.ss)
    text (dialogue is in quotes)

Timestamps are continuous across the whole episode.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

BLOCK_RE = re.compile(r"^\*\*(?P<label>[^*]+?):?\*\*:?\s*$")
TIME_RE = re.compile(
    r"Start:\s*\(?(?P<s>\d{1,2}:\d{2}(?:\.\d+)?)\)?\s*\|\s*End:\s*\(?(?P<e>\d{1,2}:\d{2}(?:\.\d+)?)\)?",
    re.I,
)
SCENE_RE = re.compile(r"^SCENE\s+(\d+)", re.I)
HEADING_RE = re.compile(r"^Scene heading:\s*(.*)", re.I)
EP_NUM_RE = re.compile(r"^EPISODE NUMBER:\s*Episode\s*(\d+)", re.I)
LOGLINE_RE = re.compile(r"^EPISODE LOGLINE:\s*(.*)", re.I)
CONT_RE = re.compile(r"^CONTINUITY NOTES:\s*(.*)", re.I)

NON_SPEAKER = {"CAMERA": "camera", "ACTION": "action", "SOUND FX": "sfx", "SFX": "sfx",
               "SOUND": "sfx", "MUSIC": "sfx", "VO": "dialogue"}


def to_sec(ts: str) -> float:
    m, s = ts.split(":")
    return int(m) * 60 + float(s)


def fmt(sec: float) -> str:
    m = int(sec // 60)
    return f"{m:02d}:{sec - 60 * m:05.2f}"


@dataclass
class Event:
    idx: int            # position within episode (0-based)
    kind: str           # camera | action | sfx | dialogue
    start: float
    end: float
    text: str
    speaker: str = ""
    emotion: str = ""
    scene: int = 1
    heading: str = ""


@dataclass
class Episode:
    number: int
    file: str
    logline: str = ""
    continuity: str = ""
    events: list[Event] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max((e.end for e in self.events), default=0.0)

    def to_dict(self):
        d = asdict(self)
        d["duration"] = self.duration
        return d


def parse_label(label: str) -> tuple[str, str, str]:
    """'LADY MARA (Confused, Breathless)' -> ('dialogue', 'LADY MARA', 'Confused, Breathless')."""
    label = label.strip().rstrip(":").strip()
    emo = ""
    m = re.match(r"^(.*?)\s*\((.*)\)\s*$", label)
    if m:
        label, emo = m.group(1).strip(), m.group(2).strip()
    key = label.upper()
    if key in NON_SPEAKER and not emo:
        return NON_SPEAKER[key], "", ""
    if key in ("CAMERA", "ACTION", "SOUND FX"):
        return NON_SPEAKER[key], "", emo
    return "dialogue", label, emo


def parse_episode(path: Path) -> Episode:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    num_m = re.search(r"episode-(\d+)", path.name)
    ep = Episode(number=int(num_m.group(1)) if num_m else 0, file=path.name)
    scene, heading = 1, ""
    cur = None  # (kind, speaker, emo)
    pending: dict | None = None

    def flush():
        nonlocal pending
        if pending and pending.get("start") is not None:
            text = " ".join(pending["text"]).strip()
            ep.events.append(Event(len(ep.events), pending["kind"], pending["start"], pending["end"],
                                   text.strip('"“” '), pending["speaker"], pending["emo"], scene, heading))
        pending = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if m := EP_NUM_RE.match(line):
            ep.number = int(m.group(1)); continue
        if m := LOGLINE_RE.match(line):
            ep.logline = m.group(1).strip(); continue
        if m := CONT_RE.match(line):
            ep.continuity = m.group(1).strip(); continue
        if m := SCENE_RE.match(line):
            flush(); scene = int(m.group(1)); continue
        if m := HEADING_RE.match(line):
            heading = m.group(1).strip(); continue
        if line.startswith("---") or re.match(r"^(Scene (start|end)|Actions and dialogue)", line, re.I):
            continue
        if m := BLOCK_RE.match(line):
            flush()
            kind, spk, emo = parse_label(m.group("label"))
            pending = {"kind": kind, "speaker": spk, "emo": emo, "start": None, "end": None, "text": []}
            continue
        if pending is not None:
            if pending["start"] is None and (m := TIME_RE.search(line)):
                pending["start"], pending["end"] = to_sec(m.group("s")), to_sec(m.group("e"))
                continue
            pending["text"].append(line)
    flush()
    ep.events.sort(key=lambda e: (e.start, e.idx))
    for i, e in enumerate(ep.events):
        e.idx = i
    return ep


def load_show(folder: str | Path) -> list[Episode]:
    folder = Path(folder)
    files = sorted(p for p in folder.rglob("*.txt") if not p.name.startswith("."))
    eps = [parse_episode(p) for p in files]
    eps.sort(key=lambda e: e.number)
    return eps
