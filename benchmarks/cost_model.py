#!/usr/bin/env python3

"""Fit a linear cost model for tile selection and report prediction accuracy.

The model: cost(tile) = CTA_count * K_iters * (alpha * tile_m * tile_n * tile_k + beta)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from benchmarks.core import FULL_ARTIFACTS_DIR, PROJECT_ROOT


COST_MODEL_DIR = PROJECT_ROOT / "artifacts" / "cost_model"


def load_benchmark_data() -> list[dict]:
    """Load raw benchmark data from the full report artifacts."""
    raw_json = FULL_ARTIFACTS_DIR / "benchmark_raw.json"
    if not raw_json.exists():
        print(f"Missing {raw_json}. Run reports.full_report first.")
        return []
    return json.loads(raw_json.read_text(encoding="utf-8"))


def fit_cost_model(rows: list[dict]) -> dict:
    """Fit a linear cost model to cuTile benchmark data.

    Model: latency = CTA_count * K_iters * (alpha * tile_volume + beta)
    where tile_volume = tile_m * tile_n * tile_k
    """
    # Filter to cuTile rows with valid tile dimensions
    cutile_rows = [
        r for r in rows
        if r["backend"] == "cuTile"
        and r["tile_m"] is not None
        and r["latency_ms"] is not None
        and r["latency_ms"] > 0
    ]

    if len(cutile_rows) < 4:
        return {"error": "Not enough cuTile data points", "n_points": len(cutile_rows)}

    # Build feature matrix
    X = []
    y = []
    for r in cutile_rows:
        m, n, k = r["m"], r["n"], r["k"]
        tm, tn, tk = r["tile_m"], r["tile_n"], r["tile_k"]
        cta_count = (m // tm) * (n // tn)
        k_iters = k // tk
        tile_volume = tm * tn * tk
        # Features: [cta_count * k_iters * tile_volume, cta_count * k_iters]
        X.append([cta_count * k_iters * tile_volume, cta_count * k_iters])
        y.append(r["latency_ms"])

    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    # Least squares fit
    result = np.linalg.lstsq(X, y, rcond=None)
    coeffs = result[0]
    alpha, beta = float(coeffs[0]), float(coeffs[1])

    # Predictions
    y_pred = X @ coeffs
    residuals = y - y_pred
    r_squared = 1.0 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2)

    return {
        "alpha": alpha,
        "beta": beta,
        "r_squared": float(r_squared),
        "n_points": len(cutile_rows),
        "mean_abs_error_ms": float(np.mean(np.abs(residuals))),
    }


def evaluate_tile_predictions(rows: list[dict], model: dict) -> dict:
    """Evaluate whether the cost model predicts the correct best tile per dtype/size."""
    if "error" in model:
        return {"error": model["error"]}

    alpha, beta = model["alpha"], model["beta"]

    cutile_rows = [
        r for r in rows
        if r["backend"] == "cuTile"
        and r["tile_m"] is not None
        and r["latency_ms"] is not None
        and r["latency_ms"] > 0
    ]

    # Group by dtype and size
    groups: dict[tuple[str, int], list[dict]] = {}
    for r in cutile_rows:
        key = (r["dtype"], r["m"])
        groups.setdefault(key, []).append(r)

    correct = 0
    total = 0
    details = []

    for (dtype_name, size), group in sorted(groups.items()):
        # Actual best (lowest latency)
        actual_best = min(group, key=lambda r: r["latency_ms"])
        actual_tile = (actual_best["tile_m"], actual_best["tile_n"], actual_best["tile_k"])

        # Predicted best (lowest predicted cost)
        best_pred_cost = float("inf")
        pred_tile = None
        for r in group:
            m, n, k = r["m"], r["n"], r["k"]
            tm, tn, tk = r["tile_m"], r["tile_n"], r["tile_k"]
            cta_count = (m // tm) * (n // tn)
            k_iters = k // tk
            tile_volume = tm * tn * tk
            pred_cost = cta_count * k_iters * (alpha * tile_volume + beta)
            if pred_cost < best_pred_cost:
                best_pred_cost = pred_cost
                pred_tile = (tm, tn, tk)

        match = actual_tile == pred_tile
        if match:
            correct += 1
        total += 1
        details.append({
            "dtype": dtype_name,
            "size": size,
            "actual_best_tile": f"{actual_tile[0]}x{actual_tile[1]}x{actual_tile[2]}",
            "predicted_best_tile": f"{pred_tile[0]}x{pred_tile[1]}x{pred_tile[2]}" if pred_tile else "none",
            "correct": match,
        })

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "details": details,
    }


def main() -> None:
    COST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_benchmark_data()
    if not rows:
        return

    model = fit_cost_model(rows)
    predictions = evaluate_tile_predictions(rows, model)

    output = {
        "model": model,
        "predictions": predictions,
    }

    output_path = COST_MODEL_DIR / "cost_model_results.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    # Write CSV of prediction details
    if "details" in predictions:
        csv_path = COST_MODEL_DIR / "tile_predictions.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["dtype", "size", "actual_best_tile", "predicted_best_tile", "correct"])
            writer.writeheader()
            for d in predictions["details"]:
                writer.writerow(d)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
