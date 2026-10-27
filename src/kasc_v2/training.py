"""Leakage-aware train/evaluate pipeline; all inputs are observable at prediction time."""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import ARTIFACT_DIR, DATA_DIR, HORIZON_STEPS, SEED, STEP_MINUTES
from .synthetic import export_csv

NUMERIC = [
    "minute_index", "attendance", "temperature_c", "lanes_open", "service_per_lane",
    "arrival_rate", "queue_length", "incident_flag", "current_wait_min",
]
CATEGORICAL = ["gate", "match_type"]
FEATURES = NUMERIC + CATEGORICAL


def make_pipeline(estimator):
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocess = ColumnTransformer([
        ("numbers", numeric, NUMERIC),
        ("categories", categorical, CATEGORICAL),
    ], remainder="drop")
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def train_and_evaluate(frame: pd.DataFrame | None = None, persist: bool = False) -> dict:
    if frame is None:
        frame = export_csv(DATA_DIR / "synthetic_gate_flow.csv")
    required = {*FEATURES, "event_id", "future_level", "future_wait_min"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    groups = frame["event_id"]
    train_ix, test_ix = next(GroupShuffleSplit(test_size=0.24, n_splits=1, random_state=SEED).split(frame, groups=groups))
    train_events = set(frame.iloc[train_ix]["event_id"])
    test_events = set(frame.iloc[test_ix]["event_id"])
    assert not train_events & test_events, "Event leakage in holdout split"
    x_train = frame.iloc[train_ix][FEATURES]
    x_test = frame.iloc[test_ix][FEATURES]
    cls_train = frame.iloc[train_ix]["future_level"]
    cls_test = frame.iloc[test_ix]["future_level"]
    reg_train = frame.iloc[train_ix]["future_wait_min"]
    reg_test = frame.iloc[test_ix]["future_wait_min"]

    classifier = make_pipeline(RandomForestClassifier(
        n_estimators=90, max_depth=13, min_samples_leaf=3,
        class_weight="balanced_subsample", n_jobs=-1, random_state=SEED,
    ))
    regressor = make_pipeline(HistGradientBoostingRegressor(
        max_iter=100, max_leaf_nodes=23, l2_regularization=2.0,
        learning_rate=0.075, random_state=SEED,
    ))
    classifier.fit(x_train, cls_train)
    regressor.fit(x_train, reg_train)
    pred_cls = classifier.predict(x_test)
    pred_reg = np.clip(regressor.predict(x_test), 0, 40)
    base_cls = DummyClassifier(strategy="most_frequent").fit(x_train, cls_train)
    base_reg = DummyRegressor(strategy="mean").fit(x_train, reg_train)

    def regression_metrics(actual, predicted):
        return {
            "mae_min": round(float(mean_absolute_error(actual, predicted)), 4),
            "rmse_min": round(float(np.sqrt(mean_squared_error(actual, predicted))), 4),
        }

    report = {
        "project": "KASC Crowd Intelligence V2 — SYNTHETIC RESEARCH DEMO",
        "dataset_kind": "generated / synthetic — not stadium measurements",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "n_rows": int(len(frame)),
        "n_events": int(groups.nunique()),
        "n_train_events": len(train_events),
        "n_test_events": len(test_events),
        "n_test_rows": int(len(test_ix)),
        "train_test_event_overlap": len(train_events & test_events),
        "horizon_minutes": STEP_MINUTES * HORIZON_STEPS,
        "features": FEATURES,
        "classifier": {
            "accuracy": round(float(accuracy_score(cls_test, pred_cls)), 4),
            "macro_f1": round(float(f1_score(cls_test, pred_cls, average="macro", zero_division=0)), 4),
            "baseline_accuracy": round(float(accuracy_score(cls_test, base_cls.predict(x_test))), 4),
            "baseline_macro_f1": round(float(f1_score(cls_test, base_cls.predict(x_test), average="macro", zero_division=0)), 4),
        },
        "regressor": {
            **regression_metrics(reg_test, pred_reg),
            "baseline": regression_metrics(reg_test, base_reg.predict(x_test)),
        },
        "labels": sorted(frame["future_level"].unique().tolist()),
        "notes": [
            "Held-out events are unseen at training time; no rows from these events enter model fitting.",
            "Reported metrics describe synthetic distribution generalization, not real-world safety performance.",
            "Targets represent t+15min; features represent observations at t.",
            "No hyperparameter selection was performed on the held-out event test split.",
        ],
        "python": platform.python_version(),
    }
    if persist:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"classifier": classifier, "regressor": regressor, "features": FEATURES}, ARTIFACT_DIR / "models.joblib")
        (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"models": {"classifier": classifier, "regressor": regressor}, "metrics": report,
            "train_event_ids": train_events, "test_event_ids": test_events}


def main():
    bundle = train_and_evaluate(persist=True)
    print(json.dumps(bundle["metrics"], indent=2))
    print("\nSaved trained models to artifacts/models.joblib (excluded from Git).")


if __name__ == "__main__":
    main()
