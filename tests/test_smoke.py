"""Smoke tests on the synthetic demo show (no Kuku content).  Run:  python -m pytest -q"""
import json
from pathlib import Path

from promo_finder.parse import load_show
from promo_finder.pipeline import run

DEMO = Path(__file__).resolve().parents[1] / "sample_input" / "demo_show"


def test_parse():
    eps = load_show(DEMO)
    assert len(eps) == 1 and eps[0].duration == 40.0
    kinds = {e.kind for e in eps[0].events}
    assert kinds == {"camera", "action", "dialogue", "sfx"}


def test_rules_run(tmp_path):
    res = run(str(DEMO), str(tmp_path), log=lambda m: None)
    p = res["promos"][0]
    assert 20 <= p["duration_sec"] <= 60
    assert p["segments"][0]["start"] == "00:00.00"          # opens on the slap
    assert not p["qa_failures"]
    assert (tmp_path / "promos.csv").exists()


def test_llm_path_rejects_hallucinated_ids(tmp_path):
    res = run(str(DEMO), str(tmp_path), mode="llm", provider="mock", log=lambda m: None)
    qa = json.loads((tmp_path / "qa.json").read_text())
    assert qa["llm_rejected"], "mock returns a fake E999 id; validator must reject it"
    assert all(p["source"] == "llm" for p in res["promos"])
