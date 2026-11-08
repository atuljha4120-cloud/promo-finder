"""Turn per-episode picks into a posting order for the whole show."""
from __future__ import annotations

from .score import Promo

# Assumption: a new viewer who taps a promo lands on Episode 1. Promos from early
# episodes match what they'll watch first (ad-to-show continuity) and don't spoil
# later twists, so they get a mild boost. Big later set-pieces can still win if
# they score much higher.
def funnel_weight(first_ep: int, n_eps: int) -> float:
    frac = (first_ep - 1) / max(1, n_eps - 1)
    return 1.0 - 0.12 * frac          # Ep1 = 1.00, last ep = 0.88


def rank(promos: list[Promo], n_eps: int, max_per_episode_in_top=1, top_n=15) -> list[dict]:
    scored = []
    for p in promos:
        fw = funnel_weight(min(p.episodes), n_eps)
        spoiler = 0.9 if "spoiler" in p.flags else 1.0
        qa = 0.85 if getattr(p, "qa_failed", False) else 1.0
        scored.append((p.total * fw * spoiler * qa, fw, p))
    scored.sort(key=lambda x: -x[0])

    # Greedy posting order: spread across the story, no episode twice in the top block.
    order, used = [], {}
    pending = list(scored)
    while pending:
        for i, (s, fw, p) in enumerate(pending):
            ep = min(p.episodes)
            if len(order) >= top_n or used.get(ep, 0) < max_per_episode_in_top:
                order.append((s, fw, p)); used[ep] = used.get(ep, 0) + 1
                pending.pop(i)
                break
        else:
            order.append(pending.pop(0))
    out = []
    for r, (s, fw, p) in enumerate(order, 1):
        d = p.to_dict()
        d["rank"] = r
        d["rank_score"] = round(s, 3)
        d["funnel_weight"] = round(fw, 3)
        out.append(d)
    return out
