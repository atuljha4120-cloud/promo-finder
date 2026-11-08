"""CLI:  python -m promo_finder --input <folder|zip> --out outputs/ [--mode rules|llm] [--provider gemini]"""
import argparse

from .pipeline import run


def main():
    ap = argparse.ArgumentParser(description="Find 20-60s promos in microdrama screenplays.")
    ap.add_argument("--input", required=True, help="folder of episode .txt files, or a .zip of them")
    ap.add_argument("--out", default="outputs/run", help="output folder")
    ap.add_argument("--mode", choices=["rules", "llm"], default="rules",
                    help="rules = free/offline; llm = rules shortlist re-judged by an LLM")
    ap.add_argument("--provider", default="gemini", help="gemini | groq | ollama | openai | custom | mock")
    ap.add_argument("--model", default=None, help="override model name")
    ap.add_argument("--base-url", default=None, help="override OpenAI-compatible base URL")
    ap.add_argument("--per-episode", type=int, default=3, help="max promos per episode")
    ap.add_argument("--top", type=int, default=15, help="size of the 'post first' list")
    ap.add_argument("--window", type=int, default=1, help="LLM sees this many consecutive episodes (2 = allow cross-episode stitches)")
    ap.add_argument("--limit", type=int, default=None, help="only first N episodes (quick test)")
    a = ap.parse_args()
    run(a.input, a.out, a.mode, a.provider, a.model, a.base_url, a.per_episode, a.top, a.window, a.limit)


if __name__ == "__main__":
    main()
