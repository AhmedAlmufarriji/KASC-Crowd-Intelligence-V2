"""FastAPI endpoints for a transparently synthetic stadium crowd-control demo."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import ARTIFACT_DIR, DATA_DIR, STATIC_DIR, GATES
from .routing import SECTIONS, edge_id, scenario_estimate, shortest_route, topology
from .synthetic import generate_events
from .training import FEATURES, train_and_evaluate

app = FastAPI(
    title="KASC Crowd Intelligence V2",
    description="Independent, synthetic-data stadium queue forecasting and illustrative routing",
    version="0.1.0",
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@lru_cache(maxsize=1)
def state():
    path = DATA_DIR / "synthetic_gate_flow.csv"
    frame = pd.read_csv(path) if path.exists() else generate_events()
    m = ARTIFACT_DIR / "models.joblib"
    j = ARTIFACT_DIR / "metrics.json"
    if m.exists() and j.exists():
        import json
        models = joblib.load(m)  # only self-trained trusted local artifact; never user uploads
        metrics = json.loads(j.read_text(encoding="utf-8"))
        if models.get("features") != FEATURES:
            raise RuntimeError("Saved model feature schema mismatch; retrain with kasc-train")
    else:
        out = train_and_evaluate(frame=frame, persist=False)
        models = out["models"]
        metrics = out["metrics"]
    return frame, models, metrics


def event_rows(event: str, minute: int) -> pd.DataFrame:
    frame = state()[0]
    if event not in set(frame["event_id"]):
        raise HTTPException(status_code=404, detail="Unknown synthetic event")
    if minute < 0 or minute > int(frame["minute_index"].max()):
        raise HTTPException(status_code=422, detail="Time bin out of range")
    current = frame[(frame.event_id == event) & (frame.minute_index == minute)]
    if current.empty:
        raise HTTPException(status_code=404, detail="No sample for selected time")
    return current


def infer(rows: pd.DataFrame) -> pd.DataFrame:
    _, models, _ = state()
    x = rows[FEATURES]
    results = rows.copy()
    results["predicted_level"] = models["classifier"].predict(x)
    results["predicted_wait_min"] = np.clip(models["regressor"].predict(x), 0, 40).round(2)
    return results


def serial(row) -> dict:
    return {
        "gate": row.gate,
        "queue": round(float(row.queue_length)),
        "arrival_rate": round(float(row.arrival_rate), 1),
        "current_wait_min": round(float(row.current_wait_min), 2),
        "predicted_wait_min": round(float(row.predicted_wait_min), 2),
        "predicted_level": str(row.predicted_level),
        "incident": bool(row.incident_flag),
        "lanes": int(row.lanes_open),
    }


@app.get("/")
def homepage():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/about")
def about():
    frame, _, metrics = state()
    return {
        "title": "KASC Crowd Intelligence V2",
        "status": "SYNTHETIC SIMULATION ONLY — NOT LIVE — NOT VENUE-CALIBRATED",
        "horizon_min": metrics["horizon_minutes"],
        "n_events": int(frame.event_id.nunique()),
        "n_rows": len(frame),
        "events": sorted(frame.event_id.unique().tolist()),
        "gates": GATES,
        "sections": SECTIONS,
        "max_minute_index": int(frame.minute_index.max()),
        "metrics": metrics,
    }


@app.get("/api/overview")
def overview(event_id: str = "SIM-E001", minute_index: int = Query(22, ge=0, le=44)):
    frame = infer(event_rows(event_id, minute_index))
    gates = [serial(row) for row in frame.itertuples(index=False)]
    return {
        "simulation": True,
        "event_id": event_id,
        "minute_index": minute_index,
        "time": str(frame.iloc[0]["time"]),
        "horizon_min": 15,
        "gate_cards": gates,
        "summary": {
            "predicted_high": sum(g["predicted_level"] == "High" for g in gates),
            "predicted_medium": sum(g["predicted_level"] == "Medium" for g in gates),
            "total_observed_queue": sum(g["queue"] for g in gates),
            "average_predicted_wait": round(sum(g["predicted_wait_min"] for g in gates) / len(gates), 2),
            "active_incidents": sum(g["incident"] for g in gates),
        },
    }


@app.get("/api/forecast")
def forecast(event_id: str = "SIM-E001", gate: str = "G1"):
    if gate not in GATES:
        raise HTTPException(status_code=422, detail="Unknown gate")
    frame, _, _ = state()
    if event_id not in set(frame.event_id):
        raise HTTPException(status_code=404, detail="Unknown synthetic event")
    rows = frame[(frame.event_id == event_id) & (frame.gate == gate)].iloc[::3]
    results = infer(rows)
    return {"event_id": event_id, "gate": gate, "horizon_min": 15,
            "simulation": True, "points": [{
                "time": row.time,
                "predicted_wait_min": float(row.predicted_wait_min),
                "synthetic_future_wait_min": float(row.future_wait_min),
                "predicted_level": str(row.predicted_level),
            } for row in results.itertuples(index=False)]}


@app.get("/api/topology")
def graph():
    return topology()


@app.get("/api/route")
def route(event_id: str = "SIM-E001", minute_index: int = Query(22, ge=0, le=44),
          start: str = "G1", destination: str = "S105", mode: str = "balanced",
          blocked_edges: str = ""):
    if start not in GATES or destination not in SECTIONS:
        raise HTTPException(status_code=422, detail="Start must be a gate and destination a section")
    valid_edges = {e["id"] for e in topology()["edges"]}
    blocked = set(filter(None, blocked_edges.split(",")))
    if not blocked.issubset(valid_edges):
        raise HTTPException(status_code=422, detail="Unknown blocked edge")
    current = infer(event_rows(event_id, minute_index))
    wait_by_gate = {str(row.gate): float(row.predicted_wait_min)
                    for row in current.itertuples(index=False)}
    crowd = {**wait_by_gate, **{f"CP{g[1:]}": w for g, w in wait_by_gate.items()}}
    # General concourse penalty approximates crowd on adjacent gates.
    crowd.update({"N": (wait_by_gate["G1"] + wait_by_gate["G2"]) / 2,
                  "E": (wait_by_gate["G3"] + wait_by_gate["G4"]) / 2,
                  "S": (wait_by_gate["G4"] + wait_by_gate["G5"]) / 2,
                  "W": (wait_by_gate["G1"] + wait_by_gate["G6"]) / 2})
    try:
        result = shortest_route(start, destination, crowd=crowd, blocked=blocked, mode=mode)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"simulation": True, "event_id": event_id,
            "origin": start, "destination": destination, "blocked_edges": sorted(blocked),
            **result}


@app.get("/api/scenario")
def scenario(event_id: str = "SIM-E001", minute_index: int = Query(22, ge=0, le=44),
             gate: str = "G1", extra_lanes: int = Query(1, ge=0, le=3)):
    current = event_rows(event_id, minute_index)
    if gate not in GATES:
        raise HTTPException(status_code=422, detail="Unknown gate")
    row = current[current.gate == gate]
    if row.empty:
        raise HTTPException(status_code=404, detail="No sample for gate")
    r = row.iloc[0]
    base = scenario_estimate(r.queue_length, r.arrival_rate, int(r.lanes_open), r.service_per_lane, 0)
    modified = scenario_estimate(r.queue_length, r.arrival_rate, int(r.lanes_open), r.service_per_lane, extra_lanes)
    return {"simulation": True, "event_id": event_id, "gate": gate,
            "observed_queue": round(float(r.queue_length), 2),
            "base_lanes": int(r.lanes_open),
            "baseline": base, "scenario": modified,
            "estimated_wait_change_min": round(modified["scenario_wait_min"] - base["scenario_wait_min"], 2),
            "warning": "Simplified deterministic queue equation — not a causal or real-world policy estimate"}


@app.post("/api/vision/analyze")
async def vision_analyze(request: Request):
    """Optional CV detection on supplied image bytes; data is not persisted."""
    content_type = request.headers.get("content-type", "").split(";")[0].strip()
    if content_type not in {"image/png", "image/jpeg", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload JPEG, PNG or WEBP")
    length = request.headers.get("content-length")
    if length:
        try:
            parsed = int(length)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid content length") from exc
        if parsed > 8 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image must be 8 MB or smaller")
    body = await request.body()
    if len(body) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be 8 MB or smaller")
    from .vision import analyze_photo
    try:
        return analyze_photo(body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
