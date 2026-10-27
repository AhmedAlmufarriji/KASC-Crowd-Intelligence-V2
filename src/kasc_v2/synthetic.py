"""Clearly synthetic, reproducible event/gate queue simulation.

This is *not* a simulation calibrated on real stadium operations, nor a
representation of King Abdullah Sports City's actual physical layout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GATES, HORIZON_STEPS, SEED, STEP_MINUTES


def congestion_label(wait_min: float) -> str:
    """Illustrative operational thresholds, not venue safety standards."""
    if wait_min < 4.0:
        return "Low"
    if wait_min < 9.0:
        return "Medium"
    return "High"


def generate_events(n_events: int = 42, bins: int = 48, seed: int = SEED) -> pd.DataFrame:
    """Simulate gates for independent matches and label future 15-min outcomes.

    Features at t are visible at t. Targets use queue(t+3), kept outside X.
    A separate per-event random generator makes results deterministic.
    Each event has its own attendance intensity, gate staffing and incident pattern.
    """
    if n_events < 5:
        raise ValueError("Need at least five independent events for held-out evaluation")
    if bins <= HORIZON_STEPS + 2:
        raise ValueError("Time series too short for future target")
    master = np.random.default_rng(seed)
    rows: list[dict] = []
    for event_num in range(1, n_events + 1):
        event_id = f"SIM-E{event_num:03d}"
        match_type = str(master.choice(["League", "Cup", "Friendly"], p=[0.62, 0.23, 0.15]))
        total_attendance = int(master.integers(18000, 59000))
        temperature_c = float(master.uniform(22, 39))
        global_intensity = float(master.uniform(0.60, 1.32))
        incident_start_low = min(8, max(2, bins // 3))
        incident_start_high = max(incident_start_low + 1, bins - 4)
        incident_window = int(master.integers(incident_start_low, incident_start_high))
        incident_gate = str(master.choice(GATES))
        for gate in GATES:
            local_seed = int(master.integers(0, 2**32 - 1))
            rng = np.random.default_rng(local_seed)
            share = float(rng.uniform(0.75, 1.27))
            lanes = int(rng.integers(2, 6))
            service_per_lane = float(rng.uniform(2.0, 3.8)) # people / min
            queue = float(rng.uniform(5, 28))
            gate_rows: list[dict] = []
            for t in range(bins):
                peak = float(np.exp(-0.5 * ((t - 25) / 9.5) ** 2))
                arrival_per_min = max(0.1, 1.25 + peak * (9.5 * global_intensity * share) + rng.normal(0, 1.0))
                incident = int(gate == incident_gate and incident_window <= t < incident_window + 5)
                service_per_min = lanes * service_per_lane * (0.48 if incident else 1.0)
                arrival_count = int(rng.poisson(arrival_per_min * STEP_MINUTES))
                processed = min(queue + arrival_count, service_per_min * STEP_MINUTES)
                queue = max(0.0, queue + arrival_count - processed)
                # Measurement noise simulates imperfect real-world sensors.
                observed_queue = max(0.0, queue + rng.normal(0, 1.25))
                observed_rate = max(0.0, arrival_per_min + rng.normal(0, 0.3))
                waiting_min = min(40.0, queue / max(service_per_min, 0.5))
                gate_rows.append({
                    "event_id": event_id,
                    "gate": gate,
                    "minute_index": t,
                    "time": f"{16 + ((t*STEP_MINUTES)//60):02d}:{((t*STEP_MINUTES)%60):02d}",
                    "match_type": match_type,
                    "attendance": total_attendance,
                    "temperature_c": round(temperature_c, 2),
                    "lanes_open": lanes,
                    "service_per_lane": round(service_per_lane, 3),
                    "arrival_rate": round(observed_rate, 3),
                    "queue_length": round(observed_queue, 2),
                    "incident_flag": incident,
                    "current_wait_min": round(waiting_min, 3),
                    "future_wait_min": round(waiting_min, 3),
                })
            for t in range(bins - HORIZON_STEPS):
                row = gate_rows[t].copy()
                target = gate_rows[t + HORIZON_STEPS]
                row["future_wait_min"] = target["current_wait_min"]
                row["future_level"] = congestion_label(target["current_wait_min"])
                rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.sort_values(["event_id", "gate", "minute_index"]).reset_index(drop=True)


def export_csv(path, n_events: int = 42, seed: int = SEED) -> pd.DataFrame:
    """Export reproducible demo dataset."""
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = generate_events(n_events=n_events, seed=seed)
    df.to_csv(p, index=False)
    return df
