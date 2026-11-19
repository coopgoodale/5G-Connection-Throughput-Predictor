"""Train and evaluate a next-window 5G throughput classifier.

The input is the Lumos5G-v1.0.csv file linked by the course's reference
repository. Predictions at row t use measurements through row t only and
target the throughput recorded at row t+1 in the same run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


LABELS = ("Low", "Medium", "High")
NUMERIC = (
    "Throughput",
    "previous_throughput_1",
    "previous_throughput_2",
    "abstractSignalStr",
    "movingSpeed",
    "compassDirection",
    "lte_rssi",
    "lte_rsrp",
    "lte_rsrq",
    "nr_ssRsrp",
    "nr_ssRsrq",
    "nr_ssSinr",
)
CATEGORICAL = ("nrStatus", "mobility_mode", "trajectory_direction", "tower_id")
FEATURES = list(NUMERIC + CATEGORICAL)
REQUIRED = {"run_num", "seq_num", "Throughput", *NUMERIC[3:], *CATEGORICAL}


def load_windows(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    missing = REQUIRED.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("The CSV is empty.")
    if frame.duplicated(["run_num", "seq_num"]).any():
        raise ValueError("Each run_num and seq_num pair must be unique.")

    frame = frame.sort_values(["run_num", "seq_num"], kind="stable").reset_index(drop=True)
    for name in NUMERIC:
        if name in ("previous_throughput_1", "previous_throughput_2"):
            continue
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
        # The source uses INT_MAX for unavailable signal measurements.
        if name != "Throughput":
            frame[name] = frame[name].mask(frame[name].abs() >= 1_000_000)
    if frame["Throughput"].isna().any():
        raise ValueError("Throughput contains missing or nonnumeric values.")

    by_run = frame.groupby("run_num", sort=False)
    for lag in (1, 2):
        frame[f"previous_throughput_{lag}"] = by_run["Throughput"].shift(lag)
    frame["next_throughput"] = by_run["Throughput"].shift(-1)
    frame["next_seq_num"] = by_run["seq_num"].shift(-1)
    # A gap means the next record is not the next measurement window.
    consecutive = frame["next_seq_num"].sub(frame["seq_num"]).eq(1)
    frame = frame.loc[
        consecutive & frame["previous_throughput_2"].notna()
    ].copy()
    if frame.empty:
        raise ValueError("No runs contain enough consecutive windows.")
    for name in CATEGORICAL:
        frame[name] = frame[name].astype("string").fillna("Unknown").astype(str)
    return frame


def split_runs(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    runs = sorted(frame["run_num"].unique().tolist())
    if len(runs) < 10:
        raise ValueError("At least 10 distinct runs are needed for the 70/15/15 split.")
    train_end = int(len(runs) * 0.70)
    valid_end = int(len(runs) * 0.85)
    ids = {
        "train": runs[:train_end],
        "validation": runs[train_end:valid_end],
        "test": runs[valid_end:],
    }
    assert all(ids.values()) and len(set().union(*map(set, ids.values()))) == len(runs)
    return {name: frame.loc[frame["run_num"].isin(group)].copy()
            for name, group in ids.items()}


def bands(values: pd.Series | np.ndarray, edges: tuple[float, float]) -> np.ndarray:
    # A value equal to an edge enters the higher band.
    return np.searchsorted(np.asarray(edges), np.asarray(values), side="right")


def evaluate(actual: np.ndarray, predicted: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=[0, 1, 2], zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(actual, predicted)), 4),
        "macro_f1": round(float(f1_score(actual, predicted, labels=[0, 1, 2],
                                         average="macro", zero_division=0)), 4),
        "per_class": {
            label: {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i, label in enumerate(LABELS)
        },
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=[0, 1, 2]
        ).tolist(),
    }


def build_model() -> Pipeline:
    preprocess = ColumnTransformer([
        ("numbers", SimpleImputer(strategy="median", add_indicator=True),
         list(NUMERIC)),
        ("categories", OneHotEncoder(handle_unknown="ignore"),
         list(CATEGORICAL)),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("forest", RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_leaf=5,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        )),
    ])


def run_results(frame: pd.DataFrame, actual: np.ndarray,
                predicted: np.ndarray, baseline: np.ndarray,
                probabilities: np.ndarray) -> list[dict]:
    records = frame[["run_num", "seq_num", "next_seq_num", "Throughput",
                     "next_throughput", "nrStatus", "nr_ssRsrp"]].copy()
    records["actual"] = actual
    records["predicted"] = predicted
    records["baseline"] = baseline
    records["confidence"] = probabilities.max(axis=1)
    output = []
    for run_id, group in records.groupby("run_num", sort=True):
        points = []
        for row in group.itertuples(index=False):
            points.append({
                "seq": int(row.seq_num),
                "next_seq": int(row.next_seq_num),
                "current_mbps": round(float(row.Throughput), 2),
                "next_mbps": round(float(row.next_throughput), 2),
                "actual": int(row.actual),
                "predicted": int(row.predicted),
                "baseline": int(row.baseline),
                "confidence": round(float(row.confidence), 3),
                "nr_status": str(row.nrStatus),
                "nr_rsrp": None if pd.isna(row.nr_ssRsrp)
                           else round(float(row.nr_ssRsrp), 1),
            })
        output.append({"id": int(run_id), "points": points})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path,
                        help="Path to Lumos5G-v1.0.csv")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "site" / "results.json",
                        help="JSON file used by the replay dashboard")
    args = parser.parse_args()

    frame = load_windows(args.data)
    partitions = split_runs(frame)
    train = partitions["train"]
    # Calculate band thresholds from training targets only.
    edges = tuple(float(v) for v in np.quantile(
        train["next_throughput"], [1 / 3, 2 / 3]
    ))
    if edges[0] >= edges[1]:
        raise ValueError("Training throughput does not support three distinct bands.")

    y_train = bands(train["next_throughput"], edges)
    if len(np.unique(y_train)) != 3:
        raise ValueError("The training split must contain all three bands.")
    model = build_model()
    model.fit(train[FEATURES], y_train)

    scores = {}
    test_details = None
    for name in ("validation", "test"):
        part = partitions[name]
        actual = bands(part["next_throughput"], edges)
        predicted = model.predict(part[FEATURES])
        baseline = bands(part["Throughput"], edges)
        report = {
            "rows": len(part),
            "runs": [int(x) for x in sorted(part["run_num"].unique())],
            "class_counts": {label: int(np.sum(actual == i))
                             for i, label in enumerate(LABELS)},
            "model": evaluate(actual, predicted),
            "baseline": evaluate(actual, baseline),
        }
        changing = baseline != actual
        report["transition_windows"] = int(changing.sum())
        report["transition_accuracy"] = (
            round(float(np.mean(predicted[changing] == actual[changing])), 4)
            if changing.any() else None
        )
        scores[name] = report
        if name == "test":
            test_details = run_results(
                part, actual, predicted, baseline,
                model.predict_proba(part[FEATURES])
            )

    output = {
        "meta": {
            "source": "Lumos5G-v1.0.csv from av-troshin77/5gthroughput",
            "source_url": "https://github.com/av-troshin77/5gthroughput",
            "data_type": "recorded replay",
            "labels": list(LABELS),
            "edges_mbps": [round(x, 3) for x in edges],
            "forecast": "throughput in the next recorded window of the same run",
            "train_rows": len(train),
            "train_runs": [int(x) for x in sorted(train["run_num"].unique())],
            "features": FEATURES,
        },
        "scores": scores,
        "runs": test_details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, separators=(",", ":"), allow_nan=False),
        encoding="utf-8"
    )
    print(f"Training labels (Mbps): Low < {edges[0]:.1f}, "
          f"Medium < {edges[1]:.1f}, High >= {edges[1]:.1f}")
    for name, report in scores.items():
        print(f"{name}: {report['rows']} windows across {len(report['runs'])} runs")
        print(f"  model macro-F1 {report['model']['macro_f1']:.4f}; "
              f"last-band baseline {report['baseline']['macro_f1']:.4f}")
        print(f"  per-class recall: "
              + ", ".join(f"{label} {report['model']['per_class'][label]['recall']:.3f}"
                          for label in LABELS))
    print(f"Replay data written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
