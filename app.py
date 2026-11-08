"""Streamlit UI for the promo finder.  Run:  streamlit run app.py"""
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from promo_finder.parse import load_show, fmt
from promo_finder.pipeline import run, resolve_input

DEMO = Path(__file__).parent / "sample_input" / "demo_show"

st.set_page_config(page_title="Promo Finder", page_icon="🎬", layout="wide")
st.title("🎬 Promo Finder")
st.markdown("Reads microdrama screenplays and picks **20–60 second promo clips**: exact timestamps, "
            "the **hook** (first 3 s), the **cliffhanger** (where it cuts) and a **posting order**.")
with st.expander("How it works", expanded=False):
    st.markdown(
        "1. **Parse** every camera, action, sound and dialogue line with its timestamps.\n"
        "2. **Generate** every 20–60 s window that starts and ends on a camera cut and never slices a spoken line, "
        "plus *flash-forward* promos (the peak moment first, then the build-up).\n"
        "3. **Score** each clip on hook (0.40), cliffhanger (0.35) and standalone clarity (0.25) with rules built from the "
        "script's emotion tags, action words and shock words. Episodes with nothing good get **no** promo.\n"
        "4. **Optional LLM judge**: any OpenAI-compatible model re-picks from the shortlist. It answers with line IDs, "
        "never timestamps, so it can't invent time codes; invalid picks are rejected.\n"
        "5. **Rank and check**: overlap removal, automatic quality checks, one promo per episode in the top list.\n\n"
        "Input format: one `.txt` screenplay per episode, with `Start: (MM:SS.ss) | End: (MM:SS.ss)` on every block.")

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["rules", "llm"], format_func=lambda m: {"rules": "Rules (free, instant)",
                                                                     "llm": "Rules + LLM judge"}[m])
    provider = model = base_url = key = None
    if mode == "llm":
        provider = st.selectbox("Provider", ["gemini", "groq", "openai", "ollama", "custom"])
        model = st.text_input("Model (blank = default)", "", placeholder="gemini-3.5-flash")
        if provider == "custom":
            base_url = st.text_input("Base URL (any OpenAI-compatible endpoint)", "")
        key = st.text_input("API key", type="password", help="Used only for this run. Not stored or shared.")
        st.caption("Free Gemini keys: aistudio.google.com. One call per episode; free tiers have daily limits.")
    per_ep = st.slider("Max promos per episode", 1, 5, 3)
    top_n = st.slider("Size of the 'post first' list", 5, 30, 15)
    limit = st.number_input("Only the first N episodes (0 = all)", 0, 500, 0)

left, right = st.columns([3, 1])
with left:
    ups = st.file_uploader("Upload screenplays: a .zip of episodes, or one or more episode .txt files",
                           type=["zip", "txt"], accept_multiple_files=True)
with right:
    st.write("")
    st.write("")
    go = st.button("Find promos", type="primary", disabled=not ups, width='stretch')
    demo = st.button("Try the demo episode", width='stretch',
                     help="A short synthetic episode, so you can see the output without uploading anything.")


def do_run(src: Path, work: Path):
    out = work / "out"
    prog = st.empty()
    lines = []

    def log(m):
        lines.append(m)
        prog.code("\n".join(lines[-6:]))

    if mode == "llm" and not key and provider != "ollama":
        st.error("LLM mode needs an API key (sidebar). Or switch to Rules mode.")
        st.stop()
    with st.spinner("Reading scripts and scoring every 20–60 s window…"):
        try:
            res = run(str(src), str(out), mode, provider or "gemini", model or None, base_url or None,
                      per_ep, top_n, 1, limit or None, log=log, api_key=key or None)
        except SystemExit as e:
            st.error(str(e)); st.stop()
    prog.empty()
    folder, _ = resolve_input(str(src))
    st.session_state.update(res=res, out=str(out), eps={e.number: e for e in load_show(folder)})


if go and ups:
    work = Path(tempfile.mkdtemp())
    zips = [u for u in ups if u.name.lower().endswith(".zip")]
    if zips:
        src = work / "show.zip"
        src.write_bytes(zips[0].getvalue())
    else:
        src = work / "show"
        src.mkdir()
        for u in ups:
            (src / Path(u.name).name).write_bytes(u.getvalue())
    do_run(src, work)
elif demo:
    do_run(DEMO, Path(tempfile.mkdtemp()))


def seg_str(p):
    return " + ".join(f"E{s['episode']} {s['start']}–{s['end']}" for s in p["segments"])


if "res" in st.session_state:
    res, out, eps = st.session_state["res"], Path(st.session_state["out"]), st.session_state["eps"]
    meta, promos = res["meta"], res["promos"]
    st.divider()
    c = st.columns(5)
    c[0].metric("Episodes", meta["episodes"])
    c[1].metric("Promos found", meta["promos"])
    c[2].metric("Pass auto checks", f"{meta['qa_summary']['passed']}/{meta['promos']}")
    c[3].metric("Mode", meta["mode"] if not meta.get("llm") else f"llm · {meta['llm']['model']}")
    c[4].metric("Runtime", f"{meta['runtime_sec']}s")
    fallback = [e["episode"] for e in meta["per_episode"] if e["picked_by"].startswith("rules (")]
    if fallback:
        st.warning(f"The LLM failed for episodes {fallback}; the tool used rule-based picks there. "
                   f"First error: {meta['per_episode'][fallback[0] - meta['per_episode'][0]['episode']]['picked_by'][:160]}")
    if not promos:
        st.info("No clip cleared the quality floor in this input.")
        st.stop()

    tab1, tab2, tab3 = st.tabs([f"Post these first (top {min(top_n, len(promos))})", "All promos", "Coverage by episode"])

    def table(ps):
        return pd.DataFrame([{"Rank": p["rank"], "Segments": seg_str(p), "Len (s)": round(p["duration_sec"]),
                              "Hook (first 3 s)": p["hook"], "Cuts on": p["cliffhanger"], "Why it works": p["why"],
                              "Score": p["rank_score"], "Flags": "; ".join(p["flags"] + p["qa_failures"])} for p in ps])

    with tab1:
        st.dataframe(table(promos[:top_n]), hide_index=True, width='stretch')
    with tab2:
        st.dataframe(table(promos), hide_index=True, width='stretch')
    with tab3:
        best = {}
        for p in promos:
            best.setdefault(p["episodes"][0], p)
        rows = []
        for e in meta["per_episode"]:
            b = best.get(e["episode"])
            if b:
                rows.append({"Episode": e["episode"], "Promos": e["promos"], "Best clip": seg_str(b),
                             "Rank": b["rank"], "Status": f"via {e['picked_by'].split(' (')[0]}"})
            else:
                bb = e.get("best_below_floor")
                rows.append({"Episode": e["episode"], "Promos": 0,
                             "Best clip": " + ".join(f"E{s['episode']} {s['start']}–{s['end']}" for s in bb["segments"]) if bb else "",
                             "Rank": None, "Status": "below quality floor, not recommended"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

    st.subheader("Inspect a promo")
    pick = st.selectbox("Rank", [p["rank"] for p in promos],
                        format_func=lambda r: f"#{r}  ·  " + seg_str(next(x for x in promos if x["rank"] == r)))
    p = next(x for x in promos if x["rank"] == pick)
    k = st.columns(4)
    k[0].metric("Length", f"{p['duration_sec']:.0f} s")
    k[1].metric("Hook", f"{p['scores']['hook']:.2f}")
    k[2].metric("Cliffhanger", f"{p['scores']['cliffhanger']:.2f}")
    k[3].metric("Standalone", f"{p['scores']['standalone']:.2f}")
    st.markdown(f"**Why it works:** {p['why']}  \n**Type:** {p['kind']} · picked by **{p['source']}**")
    if p["flags"] or p["qa_failures"]:
        st.warning("; ".join(p["flags"] + p["qa_failures"]))
    for i, s in enumerate(p["segments"], 1):
        st.markdown(f"**Segment {i} · Episode {s['episode']} · {s['start']} → {s['end']}** ({s['duration_sec']:.1f} s)")
        txt = []
        for e in eps[s["episode"]].events:
            if s["start_sec"] - .05 <= e.start < s["end_sec"] - .05 and e.kind != "camera":
                txt.append(f"{fmt(e.start)}  " + (f"{e.speaker}: \"{e.text}\"" if e.kind == "dialogue" else f"[{e.text}]"))
        st.code("\n".join(txt) or "(camera only)", language=None)
    with st.expander("Why the tool scored it this way (signals)"):
        st.json(p["signals"])

    st.subheader("Download raw outputs")
    d = st.columns(4)
    d[0].download_button("report.md", (out / "report.md").read_bytes(), "report.md", width='stretch')
    d[1].download_button("promos.csv", (out / "promos.csv").read_bytes(), "promos.csv", width='stretch')
    d[2].download_button("promos.json", (out / "promos.json").read_bytes(), "promos.json", width='stretch')
    d[3].download_button("qa.json", (out / "qa.json").read_bytes(), "qa.json", width='stretch')
