# Promo Finder: find 20–60s promos in microdrama screenplays

Reads a show's episode screenplays and returns promo clips, each with the episode,
start and end timestamps of every segment, the hook, the cliffhanger, one line on why it
works, and a posting order for the whole show.

Kuku AI Generalist assignment, Problem 1.

```
screenplays (.txt / .zip)
   │  parse.py      every CAMERA / ACTION / SOUND / DIALOGUE line with its timestamps
   ▼
candidates          every window of 20–60s that starts and ends on a camera cut
   │  score.py      and never slices a spoken line in half (12,570 windows for the sample show)
   ▼                + "flash-forward" stitches (peak moment first, then the build-up)
rule scores         hook (first 3s) · cliffhanger (last 5s) · standalone · length
   │                → top 3 per episode, overlap removed, quality floor (an episode can get 0)
   ▼
[optional] LLM judge  llm.py: any OpenAI-compatible model re-picks from the shortlist.
   │                It answers with LINE IDs only; code maps IDs → timestamps, snaps to cuts,
   ▼                rejects anything invalid, falls back to rules if the API fails
rank.py             score × funnel weight (early episodes = what a new viewer sees first)
   │                one promo per episode in the top block, so the feed spans the story
   ▼
qa_checks           length, clean cuts, hook present, no resolution in last 5s, overlaps
   ▼
outputs/            promos.json · promos.csv · report.md · qa.json · run_meta.json (+ llm_raw/)
```

## Run it

```bash
pip install -r requirements.txt

# 1) Free, offline, ~10s for 50 episodes. --input can be a .zip, a folder, or one episode .txt
python -m promo_finder --input path/to/show.zip --out outputs/rules_full

# 2) With an LLM judge (free Gemini key from aistudio.google.com; default model gemini-3.5-flash)
export GEMINI_API_KEY=...
python -m promo_finder --input path/to/show.zip --out outputs/llm_gemini --mode llm --provider gemini
#    Gemini models get retired; if you see HTTP 404, list current ones and pass --model <name>

# Swap the model with config only:
python -m promo_finder ... --mode llm --provider groq    # GROQ_API_KEY
python -m promo_finder ... --mode llm --provider ollama --model qwen2.5:7b   # local, free
LLM_BASE_URL=https://<any-openai-compatible-endpoint>/v1 LLM_MODEL=<name> LLM_API_KEY=... \
python -m promo_finder ... --mode llm --provider custom

# Web UI
streamlit run app.py

# Tests (synthetic demo episode, no Kuku content)
python -m pytest -q
```

Useful flags: `--per-episode 3`, `--top 15`, `--limit 5` (quick test), `--window 2` (the LLM sees
two consecutive episodes, so it can propose cross-episode stitches).

Any OpenAI-compatible endpoint works through `--provider custom` (Anthropic, OpenAI, Groq, Together,
vLLM and others offer one), so a stronger model is a config change, not a code change.

LLM responses are cached in `<out>/.llm_cache`, so re-running is free. Calls are paced
(`LLM_RPM`, default 8/min for Gemini) to stay inside free-tier limits.

## Live app (Streamlit Community Cloud)

1. Push this folder to a public GitHub repo (files at the repo root).
2. Go to share.streamlit.io → sign in with GitHub → **Create app** → pick the repo, branch `main`,
   main file `app.py` → **Deploy**.
3. The app runs in Rules mode with no key. Click **Try the demo episode** for an instant example,
   or upload a show's `.zip` / episode `.txt` files. For LLM mode, paste a key in the sidebar
   (used only for that run).

## Input format

One `.txt` per episode, in the Kuku screenplay format: `**CAMERA:**` / `**ACTION:**` /
`**SOUND FX:**` / `**NAME (Emotion):**` blocks, each followed by
`Start: (MM:SS.ss) | End: (MM:SS.ss)`. Upper- and lower-case headers are both handled.

## Data

The Kuku input pack is confidential and is **not** in this repo. `sample_input/demo_show/` is a
synthetic episode written for testing.
