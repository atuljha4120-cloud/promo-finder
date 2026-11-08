"""End-to-end run: screenplays -> promos -> ranking -> quality checks -> output files."""
from __future__ import annotations

import csv
import json
import re
import tempfile
import time
import zipfile
from pathlib import Path

from .parse import load_show, fmt, Episode
from .score import find_promos_in_episode, RESOLVE, cut_points, MIN_LEN, MAX_LEN, Promo, nms
from .rank import rank


def resolve_input(path: str) -> tuple[Path, str]:
    """Accepts a folder of .txt episodes, a .zip of them, or a single episode .txt."""
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".txt":
        tmp = Path(tempfile.mkdtemp(prefix="promo_in_")) / p.stem
        tmp.mkdir(parents=True)
        (tmp / p.name).write_bytes(p.read_bytes())
        return tmp, p.stem
    if p.suffix.lower() == ".zip":
        tmp = Path(tempfile.mkdtemp(prefix="promo_in_"))
        with zipfile.ZipFile(p) as z:
            for m in z.namelist():                     # safe extract: .txt only, no path escape
                if m.endswith(".txt") and ".." not in m and not m.startswith("/"):
                    z.extract(m, tmp)
        p = tmp
    dirs = [d for d in p.rglob("*") if d.is_dir() and any(d.glob("*.txt"))]
    title = dirs[0].name if dirs else p.name
    return p, title


# ---------------------------------------------------------------- automatic QA
def qa_checks(promo: dict, eps: dict[int, Episode]) -> list[str]:
    """Hard checks every promo must pass. Returns a list of failures (empty = pass)."""
    fails = []
    if not (MIN_LEN - 1 <= promo["duration_sec"] <= MAX_LEN + 1):
        fails.append(f"duration {promo['duration_sec']}s outside 20-60s")
    for i, s in enumerate(promo["segments"]):
        ep = eps[s["episode"]]
        starts, ends = cut_points(ep)
        if not any(abs(s["start_sec"] - x) < 0.06 for x in starts):
            fails.append(f"seg{i+1} start {s['start']} is not on a clean camera cut")
        if not any(abs(s["end_sec"] - x) < 0.06 for x in ends):
            fails.append(f"seg{i+1} end {s['end']} is not on a clean camera cut")
        for e in ep.events:
            if e.kind == "dialogue" and (e.start + .15 < s["start_sec"] < e.end - .15 or e.start + .15 < s["end_sec"] < e.end - .15):
                fails.append(f"seg{i+1} cuts through a line: \"{e.text[:40]}\"")
    s0 = promo["segments"][0]
    ep0 = eps[s0["episode"]]
    first3 = [e for e in ep0.events if s0["start_sec"] - .05 <= e.start < s0["start_sec"] + 3]
    if not any(e.kind in ("dialogue", "action", "sfx") for e in first3):
        fails.append("nothing but camera description in the first 3s (no hook beat)")
    sl = promo["segments"][-1]
    epl = eps[sl["episode"]]
    last5 = " ".join(e.text for e in epl.events if sl["end_sec"] - 5 <= e.start < sl["end_sec"])
    if re.search(RESOLVE, last5, re.I):
        fails.append("last 5s contain a resolution/tender beat (weak cliffhanger)")
    return fails


def overlaps(ranked: list[dict]) -> list[str]:
    out = []
    for i, a in enumerate(ranked):
        for b in ranked[i + 1:]:
            for s in a["segments"]:
                for t in b["segments"]:
                    if s["episode"] == t["episode"]:
                        ov = min(s["end_sec"], t["end_sec"]) - max(s["start_sec"], t["start_sec"])
                        if ov > 0.5 * min(a["duration_sec"], b["duration_sec"]):
                            out.append(f"rank {a['rank']} and rank {b['rank']} overlap {ov:.1f}s in E{s['episode']}")
    return out


# ---------------------------------------------------------------- run
def run(input_path: str, out_dir: str, mode="rules", provider="gemini", model=None, base_url=None,
        per_episode=3, top_n=15, window=1, limit=None, log=print, api_key=None) -> dict:
    t0 = time.time()
    folder, title = resolve_input(input_path)
    eps = load_show(folder)
    if limit:
        eps = eps[:limit]
    if not eps:
        raise SystemExit(f"No screenplay .txt files found in {input_path}")
    by_num = {e.number: e for e in eps}
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "llm_raw").mkdir(exist_ok=True) if mode == "llm" else None

    client = None
    if mode == "llm":
        from .llm import LLMClient, judge_episode
        client = LLMClient(provider, model, base_url, api_key=api_key, cache_dir=str(out / ".llm_cache"))
        log(f"LLM judge: provider={client.provider} model={client.model}")

    all_promos: list[Promo] = []
    per_ep_log, rejected_all = [], []
    for i, ep in enumerate(eps):
        picks, pool = find_promos_in_episode(ep, ep.number, per_episode)
        used = "rules"
        if client:
            win = eps[i:i + window]
            try:
                llm_picks, rejected, raw, prompt = judge_episode(client, win, pool, per_episode, title)
                (out / "llm_raw" / f"episode_{ep.number:03d}.json").write_text(json.dumps(
                    {"model": client.model, "prompt": prompt, "response": raw, "rejected": rejected}, indent=2))
                rejected_all += [{"episode": ep.number, **r} for r in rejected]
                if llm_picks:
                    # the model sometimes returns 2-3 cuts of the same scene: keep the best, drop overlaps
                    for p in llm_picks:          # a cut that fails auto-QA loses ties to one that passes
                        if qa_checks(p.to_dict(), by_num):
                            p.total *= 0.85
                    picks, used = nms(llm_picks, per_episode), "llm"
                else:
                    used = "rules (llm returned nothing valid)"
            except Exception as e:                       # never lose an episode to an API hiccup
                used = f"rules (llm error: {str(e)[:120]})"
        for p in picks:
            if used != "llm" and client:
                p.flags.append("llm-fallback")
        all_promos += picks
        entry = {"episode": ep.number, "duration": fmt(ep.duration), "events": len(ep.events),
                 "promos": len(picks), "picked_by": used,
                 "note": "" if picks else "no clip cleared the quality floor (quiet/tender episode)"}
        if not picks and pool:          # still show the best we found, clearly marked
            b = max(pool, key=lambda p: p.total).to_dict()
            entry["best_below_floor"] = {"segments": b["segments"], "hook": b["hook"], "cliffhanger": b["cliffhanger"],
                                         "score": b["scores"]["total"], "status": "below quality floor - not recommended"}
        per_ep_log.append(entry)
        log(f"E{ep.number:03d}: {len(picks)} promos via {used}")

    for p in all_promos:                       # QA before ranking so failures sink
        p.qa_failed = bool(qa_checks(p.to_dict(), by_num))
    ranked = rank(all_promos, len(eps), top_n=top_n)
    qa = []
    for r in ranked:
        r["qa_failures"] = qa_checks(r, by_num)
        qa.append({"rank": r["rank"], "episodes": r["episodes"], "pass": not r["qa_failures"], "failures": r["qa_failures"]})
    ov = overlaps(ranked)

    meta = {"show": title, "episodes": len(eps), "mode": mode, "promos": len(ranked),
            "llm": ({"provider": client.provider, "model": client.model, **client.usage} if client else None),
            "runtime_sec": round(time.time() - t0, 2), "per_episode": per_ep_log,
            "qa_summary": {"passed": sum(q["pass"] for q in qa), "failed": sum(not q["pass"] for q in qa),
                           "overlaps": ov, "llm_rejected_items": len(rejected_all)}}
    (out / "promos.json").write_text(json.dumps({"meta": meta, "promos": ranked}, indent=2, ensure_ascii=False))
    (out / "qa.json").write_text(json.dumps({"checks": qa, "overlaps": ov, "llm_rejected": rejected_all}, indent=2, ensure_ascii=False))
    write_csv(ranked, out / "promos.csv")
    (out / "report.md").write_text(report_md(ranked, meta, top_n), encoding="utf-8")
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"Done: {len(ranked)} promos, {meta['qa_summary']['passed']} pass QA, {meta['runtime_sec']}s -> {out}")
    return {"meta": meta, "promos": ranked}


def seg_str(r):
    return " + ".join(f"E{s['episode']} {s['start']}–{s['end']}" for s in r["segments"])


def write_csv(ranked, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "episodes", "kind", "segments", "duration_sec", "hook", "cliffhanger", "why",
                    "hook_score", "cliff_score", "standalone_score", "total", "rank_score", "source", "flags", "qa_failures"])
        for r in ranked:
            sc = r["scores"]
            w.writerow([r["rank"], " ".join(map(str, r["episodes"])), r["kind"], seg_str(r), r["duration_sec"],
                        r["hook"], r["cliffhanger"], r["why"], sc["hook"], sc["cliffhanger"], sc["standalone"],
                        sc["total"], r["rank_score"], r["source"], "; ".join(r["flags"]), "; ".join(r["qa_failures"])])


def report_md(ranked, meta, top_n):
    L = [f"# Promo picks: {meta['show']}", "",
         f"{meta['episodes']} episodes · {meta['promos']} promos · mode: **{meta['mode']}**"
         + (f" ({meta['llm']['provider']} / {meta['llm']['model']})" if meta.get("llm") else "")
         + f" · QA pass {meta['qa_summary']['passed']}/{meta['promos']} · {meta['runtime_sec']}s", "",
         f"## Post these first (top {top_n})", "",
         "| # | Episode · segments | Len | Hook (first 3s) | Cliffhanger (cut point) | Why it works | Score |",
         "|---|---|---|---|---|---|---|"]
    esc = lambda s: str(s).replace("|", "/").replace("\n", " ")
    for r in ranked[:top_n]:
        L.append(f"| {r['rank']} | {seg_str(r)}{' ⚠' if r['qa_failures'] or r['flags'] else ''} | {r['duration_sec']:.0f}s | "
                 f"{esc(r['hook'])} | {esc(r['cliffhanger'])} | {esc(r['why'])} | {r['rank_score']:.2f} |")
    empty = [str(e["episode"]) for e in meta["per_episode"] if not e["promos"]]
    if empty:
        L += ["", f"**No promo recommended for episodes {', '.join(empty)}**: no clip cleared the quality floor."]
    L += ["", "## Coverage: all episodes", "",
          "| Ep | Promos | Best clip | Note |", "|---|---|---|---|"]
    best_by_ep = {}
    for r in ranked:
        best_by_ep.setdefault(r["episodes"][0], r)
    for e in meta["per_episode"]:
        if e["promos"]:
            r = best_by_ep.get(e["episode"])
            L.append(f"| {e['episode']} | {e['promos']} | {seg_str(r) if r else ''} (rank {r['rank'] if r else '-'}) | via {e['picked_by']} |")
        else:
            b = e.get("best_below_floor")
            clip = " + ".join(f"E{s['episode']} {s['start']}–{s['end']}" for s in b["segments"]) if b else "-"
            L.append(f"| {e['episode']} | 0 | {clip} (score {b['score'] if b else '-'}) | below quality floor, not recommended |")
    L += ["", "## All promos by episode", ""]
    for r in sorted(ranked, key=lambda r: (r["episodes"][0], r["rank"])):
        sc = r["scores"]
        L.append(f"- **E{r['episodes'][0]} · rank {r['rank']}** · {r['kind']} · {seg_str(r)} ({r['duration_sec']:.0f}s) · "
                 f"hook {sc['hook']:.2f} / cliff {sc['cliffhanger']:.2f} / standalone {sc['standalone']:.2f}")
        L.append(f"  - Hook: {r['hook']}")
        L.append(f"  - Cliffhanger: {r['cliffhanger']}")
        L.append(f"  - Why: {r['why']}")
        if r["flags"] or r["qa_failures"]:
            L.append(f"  - ⚠ {'; '.join(r['flags'] + r['qa_failures'])}")
    return "\n".join(L) + "\n"
