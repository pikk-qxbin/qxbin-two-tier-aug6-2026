#!/usr/bin/env python3
"""
QxBin Edge Tier — Adaptive Battery-Aware Sensor Decision Cubit
=============================================================
By Rupesh Malpani | pikk.company | QxBin Framework

Room-temperature probabilistic decision engine for battery-powered
IoT / edge nodes. Fuses noisy sensor readings + remaining energy
into a live Binary Probability Matrix. Decides sample rate /
transmit / deep-sleep via fractional-exponent superposition and
probabilistic collapse.

No cryogenics. No special hardware. Runs on a $5 MCU or a laptop.
"""

import numpy as np
from typing import Tuple, Dict


class QxBinBatterySensorCubit:
    """
    Personal cubit for adaptive sensing under energy uncertainty.

    Core QxBin primitives:
    - Binary Probability Matrix (spatial grid of fractional probs)
    - Directed contributions via bias**n and (1-bias)**m
    - Soft blend (superposition) then hard measurement (collapse)
    """

    def __init__(self, grid_size: int = 5):
        self.grid_size = grid_size
        self.state = np.random.rand(grid_size, grid_size).astype(np.float64)
        self._normalize()
        self.history = []

    def _normalize(self):
        s = self.state.sum()
        if s > 0:
            self.state /= s

    def apply_superposition(self, bias: float, n: int = 2, m: int = 1):
        """
        QxBin fractional lean.
        bias**n  →  strong direction
        (1-bias)**m → opposing contribution
        Outer product builds the coordinate grid, then soft-blend.
        """
        frac = max(bias, 1e-9) ** n
        tail = max(1.0 - bias, 1e-9) ** m
        vec = np.linspace(frac, tail, self.grid_size)
        new_matrix = np.outer(vec, vec)
        self.state = (self.state + new_matrix) * 0.5
        self._normalize()
        return self.state

    def sense_and_decide(
        self,
        sensor_confidence: float,
        battery_fraction: float,
        urgency: float = 0.5,
    ) -> Dict:
        """
        Main decision loop for an edge node.

        Inputs (all in [0,1]):
          sensor_confidence : how trustworthy is the latest reading
          battery_fraction  : remaining energy (1.0 = full)
          urgency           : external pressure to act now

        Returns action probabilities + chosen action after collapse.
        """
        # Composite bias: high confidence + high battery → aggressive sensing
        # low battery → strong bias toward conservation
        energy_bias = 0.35 + 0.55 * battery_fraction
        conf_bias = 0.4 + 0.5 * sensor_confidence
        bias = np.clip(0.55 * energy_bias + 0.35 * conf_bias + 0.1 * urgency, 0.15, 0.92)

        # Stronger exponents when battery is critical (more decisive conservation)
        n = 3 if battery_fraction < 0.25 else 2
        m = 1 if battery_fraction > 0.4 else 2

        self.apply_superposition(bias=bias, n=n, m=m)

        # Map matrix regions to actions (simple spatial encoding)
        # Top-left quadrant  → sample aggressively
        # Center            → transmit summary
        # Bottom-right      → deep sleep / conserve
        g = self.grid_size
        sample_p = self.state[: g // 2, : g // 2].sum()
        transmit_p = self.state[g // 4 : 3 * g // 4, g // 4 : 3 * g // 4].sum()
        sleep_p = self.state[g // 2 :, g // 2 :].sum()
        total = sample_p + transmit_p + sleep_p + 1e-12
        sample_p /= total
        transmit_p /= total
        sleep_p /= total

        # Soft measurement → hard action
        actions = ["aggressive_sample", "transmit_summary", "deep_sleep"]
        probs = np.array([sample_p, transmit_p, sleep_p])
        probs /= probs.sum()
        chosen_idx = np.random.choice(3, p=probs)
        chosen = actions[chosen_idx]

        result = {
            "bias_used": float(bias),
            "n_m": (n, m),
            "action_probs": {
                "aggressive_sample": float(sample_p),
                "transmit_summary": float(transmit_p),
                "deep_sleep": float(sleep_p),
            },
            "chosen_action": chosen,
            "battery_fraction": battery_fraction,
            "sensor_confidence": sensor_confidence,
            "matrix_mean": float(self.state.mean()),
        }
        self.history.append(result)
        return result

    def measure(self) -> np.ndarray:
        """Classic probabilistic collapse of the entire matrix."""
        flat = self.state.flatten()
        idx = np.random.choice(len(flat), p=flat)
        out = np.zeros_like(flat)
        out[idx] = 1.0
        return out.reshape(self.state.shape)


def demo():
    print("=" * 60)
    print("QxBin Edge — Adaptive Battery-Aware Sensor Decision Cubit")
    print("Rupesh Malpani | pikk.company | Room-temperature logic")
    print("=" * 60)

    cubit = QxBinBatterySensorCubit(grid_size=5)

    scenarios = [
        {"sensor_confidence": 0.92, "battery_fraction": 0.85, "urgency": 0.3, "label": "Healthy node, high confidence"},
        {"sensor_confidence": 0.55, "battery_fraction": 0.22, "urgency": 0.7, "label": "Low battery, noisy sensor, urgent"},
        {"sensor_confidence": 0.78, "battery_fraction": 0.45, "urgency": 0.5, "label": "Mid energy, decent signal"},
        {"sensor_confidence": 0.30, "battery_fraction": 0.12, "urgency": 0.2, "label": "Critical battery, low trust"},
    ]

    for sc in scenarios:
        print(f"\n--- {sc['label']} ---")
        decision = cubit.sense_and_decide(
            sensor_confidence=sc["sensor_confidence"],
            battery_fraction=sc["battery_fraction"],
            urgency=sc["urgency"],
        )
        print(f"  Bias / (n,m)     : {decision['bias_used']:.3f}  {decision['n_m']}")
        print(f"  Action probs     : { {k: round(v,3) for k,v in decision['action_probs'].items()} }")
        print(f"  → CHOSEN ACTION  : {decision['chosen_action'].upper()}")

    print("\nHistory length:", len(cubit.history))
    print("Done. Ship it to the field.")


if __name__ == "__main__":
    demo()
