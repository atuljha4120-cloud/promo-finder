"""Optional LLM judge. Works with ANY OpenAI-compatible chat endpoint, so the model
is swapped with config only:

  provider   base_url                                                   example model
  gemini     https://generativelanguage.googleapis.com/v1beta/openai/   gemini-3.5-flash
  groq       https://api.groq.com/openai/v1                             llama-3.3-70b-versatile
  ollama     http://localhost:11434/v1                                  qwen2.5:7b
  openai     https://api.openai.com/v1                                  gpt-4o-mini
  custom     $LLM_BASE_URL  (e.g. an API Kuku provides)                 $LLM_MODEL
  mock       offline stand-in used for tests (no network)

Design: the model never writes timestamps. It answers with line IDs from the
numbered script; code maps IDs -> timestamps, snaps them to real camera cuts and
rejects anything that breaks the 20-60s rule. So a model can be wrong about
*taste*, but it can't invent time codes.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.request
import urllib.error
from pathlib import Path

from .parse import Episode, fmt
from .score import (Promo, Segment, cut_points, events_in, hook_score, cliff_score,
                    standalone_score, MIN_LEN, MAX_LEN)

PROVIDERS = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-3.5-flash", "GEMINI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", "GROQ_API_KEY"),
    "ollama": ("http://localhost:11434/v1", "qwen2.5:7b", ""),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini", "OPENAI_API_KEY"),
    "custom": (os.environ.get("LLM_BASE_URL", ""), os.environ.get("LLM_MODEL", ""), "LLM_API_KEY"),
}

SYSTEM = """You are a senior promo editor for vertical microdrama apps (ReelShort, DramaBox style).
You pick 20-60 second clips from a show to post on TikTok/Instagram/YouTube to pull in NEW viewers.
A good promo:
1. HOOK: the first 3 seconds make a scrolling viewer stop (slap, reveal, shocking line, humiliation, threat).
2. CLIFFHANGER: it cuts at an unresolved moment so the viewer must know what happens next. Never end on a resolution, hug, smile or calm beat.
3. STANDALONE: someone who has never seen the show understands who is fighting whom and what is at stake.
It may be one continuous stretch, or join 2-3 moments in order (e.g. flash-forward: show the explosive beat first, then the build-up, cut before it lands).
Only choose moments, start and end points. No editing, music or captions.
Answer with JSON only."""

USER_TMPL = """SHOW CONTEXT (for you only; viewers of the promo will NOT know this):
{context}

SCRIPT, one line per event: ID [start-end] TYPE SPEAKER (emotion): text
{script}

SHORTLIST from a rule-based scorer (you may keep, adjust or ignore them):
{cands}

Return up to {k} promos for this episode, best first, as JSON:
{{"promos": [{{
  "segments": [{{"from": "<first line ID>", "to": "<last line ID>"}}],
  "hook": "<what happens in the first 3 seconds>",
  "cliffhanger": "<the unresolved moment it cuts on>",
  "why": "<one line on why it works for a stranger>",
  "hook_score": <1-10>, "cliffhanger_score": <1-10>, "standalone_score": <1-10>,
  "spoiler": <true if it gives away a major twist that would hurt the show>
}}]}}
Rules: each segment's "from" should be a CAMERA line (a shot start); total length of all segments 20-60s;
the hook must happen within the first 3 seconds of the first segment; use only IDs that exist above."""


def _short(t, n=22):
    w = t.split()
    return " ".join(w[:n]) + ("…" if len(w) > n else "")


def script_lines(ep: Episode) -> str:
    out = []
    for e in ep.events:
        lid = f"E{ep.number}-L{e.idx}"
        tag = {"camera": "CAMERA", "action": "ACTION", "sfx": "SFX"}.get(e.kind, "")
        if e.kind == "dialogue":
            out.append(f"{lid} [{fmt(e.start)}-{fmt(e.end)}] {e.speaker} ({e.emotion}): {e.text}")
        else:
            out.append(f"{lid} [{fmt(e.start)}-{fmt(e.end)}] {tag}: {_short(e.text) if e.kind == 'camera' else e.text}")
    return "\n".join(out)


def build_prompt(eps: list[Episode], pool: list[Promo], k: int, show_title: str) -> str:
    ctx = [f"Show: {show_title}"] + [f"Episode {e.number} logline: {e.logline}" for e in eps]
    cands = []
    for i, p in enumerate(pool, 1):
        segs = " + ".join(f"E{s.episode}-L{s.first_idx}..E{s.episode}-L{s.last_idx} ({fmt(s.start)}-{fmt(s.end)})"
                          for s in p.segments)
        cands.append(f"C{i}: {segs} = {p.duration:.1f}s, kind={p.kind}, rule score={p.total:.2f}; hook: {p.hook}")
    return USER_TMPL.format(context="\n".join(ctx), script="\n\n".join(script_lines(e) for e in eps),
                            cands="\n".join(cands) or "(none)", k=k)


class LLMClient:
    def __init__(self, provider="gemini", model=None, base_url=None, api_key=None,
                 cache_dir=".llm_cache", temperature=0.2, timeout=None, rpm=None):
        self.provider = provider
        # free tiers are rate-limited (Gemini Flash free ~10-15 req/min): pace calls proactively
        self.min_gap = 60.0 / (rpm or float(os.environ.get("LLM_RPM", 8 if provider == "gemini" else 30)))
        self._last = 0.0
        dflt_url, dflt_model, key_env = PROVIDERS.get(provider, ("", "", "LLM_API_KEY"))
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or dflt_url).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL") or dflt_model
        self.api_key = api_key or (os.environ.get(key_env) if key_env else "") or os.environ.get("LLM_API_KEY", "")
        self.temperature = temperature
        self.timeout = timeout or float(os.environ.get("LLM_TIMEOUT", 300))
        self.cache = Path(cache_dir); self.cache.mkdir(parents=True, exist_ok=True)
        self.usage = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0, "errors": 0}

    def chat(self, system: str, user: str) -> str:
        key = hashlib.sha256(f"{self.model}|{self.temperature}|{system}|{user}".encode()).hexdigest()[:24]
        cf = self.cache / f"{key}.json"
        if cf.exists():
            self.usage["cached"] += 1
            return json.loads(cf.read_text())["content"]
        if self.provider == "mock":
            content = _mock_answer(user)
        else:
            content = self._post(system, user)
        cf.write_text(json.dumps({"model": self.model, "content": content}))
        return content

    def _post(self, system, user, retries=4):
        body = {"model": self.model, "temperature": self.temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_object"}}
        for attempt in range(retries):
            wait = self.min_gap - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": f"Bearer {self.api_key}"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read())
                self.usage["calls"] += 1
                u = data.get("usage") or {}
                self.usage["prompt_tokens"] += u.get("prompt_tokens", 0)
                self.usage["completion_tokens"] += u.get("completion_tokens", 0)
                return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")[:300]
                if e.code == 400 and "response_format" in msg and "response_format" in body:
                    body.pop("response_format"); continue      # provider doesn't support JSON mode
                if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                    time.sleep(min(60, 8 * 2 ** attempt)); continue   # free-tier rate limits
                self.usage["errors"] += 1
                raise RuntimeError(f"LLM HTTP {e.code}: {msg}")
            except (TimeoutError, OSError) as e:
                if isinstance(e, urllib.error.HTTPError):
                    raise
                if attempt < retries - 1:
                    time.sleep(5); continue
                self.usage["errors"] += 1
                raise RuntimeError(f"LLM timeout/connection error after {retries} tries: {e}")
            except urllib.error.URLError as e:
                if attempt < retries - 1:
                    time.sleep(5); continue
                self.usage["errors"] += 1
                raise RuntimeError(f"LLM connection error: {e}")
        raise RuntimeError("LLM failed after retries")


def _mock_answer(user: str) -> str:
    """Offline stand-in: echoes the rule shortlist back in the LLM's JSON format,
    with one deliberately broken item so the validator path is exercised."""
    promos = []
    for m in re.finditer(r"C\d+: (.*?) = ([\d.]+)s.*?hook: (.*)", user):
        segs = [{"from": a, "to": b} for a, b in re.findall(r"(E\d+-L\d+)\.\.(E\d+-L\d+)", m.group(1))]
        promos.append({"segments": segs, "hook": m.group(3)[:100], "cliffhanger": "(mock)", "why": "(mock)",
                       "hook_score": 7, "cliffhanger_score": 6, "standalone_score": 7, "spoiler": False})
    promos = promos[:2] + [{"segments": [{"from": "E999-L1", "to": "E999-L2"}], "hook": "hallucinated",
                            "cliffhanger": "", "why": "", "hook_score": 9, "cliffhanger_score": 9,
                            "standalone_score": 9, "spoiler": False}]
    return json.dumps({"promos": promos})


def parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def _snap_start(t, starts):
    prev = [s for s in starts if s <= t + 0.05]
    return prev[-1] if prev and t - prev[-1] <= 2.0 else min(starts, key=lambda s: abs(s - t), default=t)


def _snap_end(t, ends):
    nxt = [e for e in ends if e >= t - 0.05]
    return nxt[0] if nxt and nxt[0] - t <= 2.0 else min(ends, key=lambda e: abs(e - t), default=t)


def validate(item: dict, eps_by_num: dict[int, Episode]) -> tuple[Promo | None, list[str]]:
    """Map line IDs to real, snapped timestamps and enforce the brief's rules."""
    problems, segs = [], []
    for s in item.get("segments", []):
        try:
            e1, l1 = map(int, re.match(r"E(\d+)-L(\d+)", s["from"]).groups())
            e2, l2 = map(int, re.match(r"E(\d+)-L(\d+)", s["to"]).groups())
        except Exception:
            problems.append(f"bad id {s}"); continue
        if e1 != e2 or e1 not in eps_by_num:
            problems.append(f"unknown/cross-episode id {s}"); continue
        ep = eps_by_num[e1]
        if not (0 <= l1 < len(ep.events) and 0 <= l2 < len(ep.events)):
            problems.append(f"id out of range {s}"); continue
        if l2 < l1:
            l1, l2 = l2, l1; problems.append("reversed segment fixed")
        starts, ends = cut_points(ep)
        a = _snap_start(ep.events[l1].start, starts)
        b = _snap_end(max(e.end for e in ep.events[l1:l2 + 1]), ends)
        ev = events_in(ep, a, b)
        segs.append(Segment(e1, a, b, ev[0].idx if ev else l1, ev[-1].idx if ev else l2))
    if not segs:
        return None, problems + ["no valid segments"]
    tot = sum(s.duration for s in segs)
    if not (MIN_LEN - 1 <= tot <= MAX_LEN + 1):
        return None, problems + [f"duration {tot:.1f}s outside 20-60s"]
    # independent rule-based check of the LLM's choice
    first, last = segs[0], segs[-1]
    rh, _, _ = hook_score(eps_by_num[first.episode], first.start)
    rc, _, _ = cliff_score(eps_by_num[last.episode], last.end)
    rs, _ = standalone_score(eps_by_num[last.episode], last.start, last.end, last.episode)
    g = lambda k: max(1, min(10, float(item.get(k, 5)))) / 10
    p = Promo(segs, g("hook_score"), g("cliffhanger_score"), g("standalone_score"),
              str(item.get("hook", ""))[:200], str(item.get("cliffhanger", ""))[:200], str(item.get("why", ""))[:240],
              kind="continuous" if len(segs) == 1 else ("cross-episode stitch" if len({s.episode for s in segs}) > 1 else "stitch"),
              source="llm", reasons={"rule_check": {"hook": round(rh, 2), "cliffhanger": round(rc, 2), "standalone": round(rs, 2)},
                                     "validator_notes": problems})
    p.total = 0.40 * p.hook_score + 0.35 * p.cliff_score + 0.25 * p.standalone_score
    if item.get("spoiler"):
        p.flags.append("spoiler")
    if rh < 0.3:
        p.flags.append("rule-check: weak hook in first 3s")
    if rc < 0.3:
        p.flags.append("rule-check: ending may resolve")
    return p, problems


def judge_episode(client: LLMClient, eps: list[Episode], pool: list[Promo], k: int, show_title: str):
    prompt = build_prompt(eps, pool, k, show_title)
    raw = client.chat(SYSTEM, prompt)
    data = parse_json(raw)
    by_num = {e.number: e for e in eps}
    good, rejected = [], []
    for item in data.get("promos", [])[: k + 2]:
        p, probs = validate(item, by_num)
        if p:
            good.append(p)
        else:
            rejected.append({"item": item, "problems": probs})
    return good[:k], rejected, raw, prompt
