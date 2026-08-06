> **Note (Aug 2026 cleanup)**  
> This was a daily experiment. Going forward, all new sketches live in **[qxbin-experiments](https://github.com/pikk-qxbin/qxbin-experiments)**.  
> Production code lives in the core **[qxbin](https://github.com/pikk-qxbin/qxbin)** repository.

# QxBin Two-Tier Ideas — 6 Aug 2026

**By Rupesh Malpani** | pikk.company | QxBin Framework

Two new, fully implemented QxBin Logic ideas at different computing tiers for distinct real-world use cases.

Room-temperature. Classical hardware only. Binary Probability Matrices + fractional exponents (biasⁿ / (1-bias)ᵐ). The coin keeps spinning until you measure.

---

## 1. Edge Tier — Adaptive Battery-Aware Sensor Decision Cubit

**File:** `qxbin_edge_battery_sensor.py`

**Use case:** Battery-powered IoT / drone / industrial sensor nodes.

- Fuses live sensor confidence + remaining battery + urgency into a single Binary Probability Matrix.
- Applies QxBin directed superposition with adaptive exponents (more decisive when energy is critical).
- Spatially encodes three actions: aggressive sample / transmit summary / deep sleep.
- Soft probabilities → hard collapse for the next action.

Perfect for keeping edge nodes alive longer under noisy real-world conditions.

```bash
python qxbin_edge_battery_sensor.py
```

---

## 2. Application / Mid Tier — Uncertainty-Aware Multi-Path Agent Router

**File:** `qxbin_agent_router.py`

**Use case:** Multi-agent systems, microservice meshes, edge-cloud continuum, resilient AI agent fleets.

- One PathCubit (Binary Probability Matrix) per candidate route / agent / path.
- Evolves each matrix from live success estimates using fractional exponents.
- Temperature-controlled probabilistic selection (explore ↔ exploit).
- Closed feedback loop: observed outcomes pull the probability clouds.

```bash
python qxbin_agent_router.py
```

---

## Core QxBin Math (shared)

- **Binary Probability Matrix** — spatial 2-D grid of fractional probabilities instead of flat 0/1 bits.
- **Directed contributions** — `bias**n` and `(1-bias)**m` create steerable “leaning coin” behavior.
- **Soft blend** then **probabilistic collapse** — superposition-like evolution until measurement.
- Fully classical. Runs on any CPU. Trivial to port to C, CUDA, or microcontrollers.

---

## License

**QxBin Source-Available License** (Apache-2.0 OR MIT + commercial terms).  
See the main [qxbin](https://github.com/pikk-qxbin/qxbin) repository for full terms.

---

Part of the pikk-qxbin vision: Democratize advanced probabilistic compute. Ship fast. Keep the incentives aligned.

Rupesh Malpani  
pikk.company
