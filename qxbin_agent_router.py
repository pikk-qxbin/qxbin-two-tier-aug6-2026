#!/usr/bin/env python3
"""
QxBin Application Tier — Uncertainty-Aware Multi-Path Agent Router
==================================================================
By Rupesh Malpani | pikk.company | QxBin Framework

Maintains an ensemble of Binary Probability Matrices, one per candidate
path / agent / microservice route. Evolves them with fractional exponents,
then collapses to a routing decision under live uncertainty.

Use cases:
- Multi-agent fleets (robots, drones, software agents)
- Adaptive microservice / edge-cloud path selection
- Resilient workload routing when success probability is fuzzy

Classical hardware only. The coin keeps spinning until you measure.
"""

import numpy as np
from typing import List, Dict, Optional


class PathCubit:
    """Single path / route represented as a QxBin Binary Probability Matrix."""

    def __init__(self, path_id: str, grid_size: int = 4):
        self.path_id = path_id
        self.grid_size = grid_size
        self.state = np.random.rand(grid_size, grid_size).astype(np.float64)
        self._normalize()
        self.success_history = []

    def _normalize(self):
        s = self.state.sum()
        if s > 0:
            self.state /= s

    def evolve(self, estimated_success: float, n: int = 2, m: int = 1):
        """
        Steer the probability cloud toward the estimated success bias
        using QxBin fractional directed contributions.
        """
        bias = np.clip(estimated_success, 0.05, 0.95)
        frac = bias ** n
        tail = (1.0 - bias) ** m
        vec = np.linspace(frac, tail, self.grid_size)
        new_m = np.outer(vec, vec)
        self.state = (self.state * 0.6 + new_m * 0.4)   # soft memory blend
        self._normalize()
        return self.state.mean()

    def collapse_score(self) -> float:
        """Probabilistic measurement → scalar success likelihood."""
        # Weighted average of the matrix, heavier on high-value region
        weights = np.linspace(0.3, 1.0, self.grid_size)
        w = np.outer(weights, weights)
        score = (self.state * w).sum() / (w.sum() + 1e-12)
        return float(np.clip(score, 0.0, 1.0))


class QxBinAgentRouter:
    """
    Ensemble router. Keeps one PathCubit per candidate route.
    Feedback loop adjusts biases from observed outcomes.
    """

    def __init__(self, path_ids: List[str], grid_size: int = 4):
        self.paths = {pid: PathCubit(pid, grid_size) for pid in path_ids}
        self.routing_log = []

    def update_estimates(self, success_estimates: Dict[str, float]):
        """Evolve every path with its current estimated success probability."""
        for pid, est in success_estimates.items():
            if pid in self.paths:
                # Adaptive exponents: more decisive when confidence is extreme
                n = 3 if est > 0.75 or est < 0.25 else 2
                m = 1 if est > 0.5 else 2
                self.paths[pid].evolve(est, n=n, m=m)

    def route(self, success_estimates: Dict[str, float], temperature: float = 1.0) -> Dict:
        """
        Main routing decision.
        temperature < 1.0 → sharper (more greedy)
        temperature > 1.0 → more exploratory
        """
        self.update_estimates(success_estimates)

        scores = {}
        for pid, cubit in self.paths.items():
            scores[pid] = cubit.collapse_score()

        # Softmax with temperature for probabilistic selection
        ids = list(scores.keys())
        logits = np.array([scores[i] for i in ids]) / max(temperature, 1e-6)
        logits = logits - logits.max()          # numerical stability
        exp = np.exp(logits)
        probs = exp / exp.sum()

        chosen_idx = np.random.choice(len(ids), p=probs)
        chosen = ids[chosen_idx]

        result = {
            "chosen_path": chosen,
            "path_scores": {k: round(v, 4) for k, v in scores.items()},
            "selection_probs": {ids[i]: round(float(probs[i]), 4) for i in range(len(ids))},
            "temperature": temperature,
        }
        self.routing_log.append(result)
        return result

    def feedback(self, path_id: str, observed_success: float):
        """Close the loop: observed outcome becomes next estimate seed."""
        if path_id in self.paths:
            self.paths[path_id].success_history.append(observed_success)
            # Light pull toward the observed reality
            self.paths[path_id].evolve(observed_success, n=2, m=1)


def demo():
    print("=" * 64)
    print("QxBin Application Tier — Uncertainty-Aware Multi-Path Agent Router")
    print("Rupesh Malpani | pikk.company | Binary Probability Matrices")
    print("=" * 64)

    router = QxBinAgentRouter(
        path_ids=["edge-fast", "cloud-reliable", "hybrid-balanced", "local-fallback"],
        grid_size=4,
    )

    # Simulated live estimates (could come from monitoring, models, etc.)
    rounds = [
        {
            "edge-fast": 0.82,
            "cloud-reliable": 0.71,
            "hybrid-balanced": 0.65,
            "local-fallback": 0.40,
            "temp": 0.9,
            "label": "High confidence on edge path",
        },
        {
            "edge-fast": 0.35,
            "cloud-reliable": 0.88,
            "hybrid-balanced": 0.72,
            "local-fallback": 0.55,
            "temp": 1.1,
            "label": "Edge degraded, cloud looks strong",
        },
        {
            "edge-fast": 0.60,
            "cloud-reliable": 0.58,
            "hybrid-balanced": 0.79,
            "local-fallback": 0.48,
            "temp": 1.0,
            "label": "Ambiguous — explore hybrid",
        },
    ]

    for i, r in enumerate(rounds, 1):
        print(f"\n--- Round {i}: {r['label']} ---")
        estimates = {k: v for k, v in r.items() if k not in ("temp", "label")}
        decision = router.route(estimates, temperature=r["temp"])
        print(f"  Path scores      : {decision['path_scores']}")
        print(f"  Selection probs  : {decision['selection_probs']}")
        print(f"  → ROUTED TO      : {decision['chosen_path'].upper()}")

        # Simulate feedback (random outcome biased by score)
        observed = np.clip(decision["path_scores"][decision["chosen_path"]] + np.random.normal(0, 0.08), 0, 1)
        router.feedback(decision["chosen_path"], observed)
        print(f"  Observed success : {observed:.3f}  (fed back)")

    print("\nTotal routing decisions logged:", len(router.routing_log))
    print("Router ready for production agent fleets.")


if __name__ == "__main__":
    demo()
