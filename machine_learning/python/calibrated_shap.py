"""Permutation SHAP for the Platt-calibrated, 30-fit TabM ensembles."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import time

os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
import shap
import torch

ROOT = Path.cwd()
DATA = ROOT / "results"
OUT = DATA / "shap_calibrated"
NINE = ["age", "sex", "scr", "hba1c", "hemoglobin", "uric_acid", "albumin", "alp", "ada"]
CALIBRATION = {
    "TabM9": (-0.4957387365297589166, 1.00804860414728825),
    "TabM21": (-0.3641212405501325633, 1.04975618728113163),
}
ADAPTER = None


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def log(*values):
    print(time.strftime("%H:%M:%S"), *values, flush=True)


class CoalitionMasker:
    """Mask each clinical feature jointly across all imputed input vectors."""

    supports_delta_masking = False
    immutable_outputs = True

    def __init__(self, features, donors=8):
        self.shape = (donors, len(features))
        self.bits = 1 << np.arange(len(features), dtype=np.int64)

    def __call__(self, mask, row):
        code = int(np.sum(self.bits[np.asarray(mask, dtype=bool)]))
        return (np.column_stack((
            np.repeat(int(row[0]), self.shape[0]), np.arange(self.shape[0]),
            np.repeat(code, self.shape[0]),
        )),)

    def invariants(self, row):
        return np.zeros(self.shape, dtype=bool)


def keys(requests, width):
    a = np.asarray(requests, dtype=np.int64)
    return (a[:, 0] * 8 + a[:, 1]) * (1 << width) + a[:, 2]


def explain_requests(callback, indices, features, seed):
    explainer = shap.Explainer(
        callback, CoalitionMasker(features), algorithm="permutation", seed=seed,
        feature_names=features,
    )
    encoded = np.repeat(indices[:, None], len(features), axis=1)
    return explainer(encoded, max_evals=(2 * len(features) + 1) * 8,
                     batch_size=256, silent=True, error_bounds=True)


def masker_test():
    names = ["a", "b", "c"]
    background = np.array([[0., 0., 0.], [1., 1., 1.]] * 4)
    x = np.array([[.2, .4, .6], [.8, .1, .5]])
    beta = np.array([.1, .2, .3])

    def model(requests):
        q, b, code = np.asarray(requests, dtype=int).T
        selected = (code[:, None] & (1 << np.arange(3))) != 0
        return .1 + np.where(selected, x[q], background[b]) @ beta

    sv = explain_requests(model, np.arange(2), names, 123)
    assert np.max(np.abs(sv.values - (x - background.mean(0)) * beta)) < 1e-12


def load_ensemble(family, job, features, rows, donors):
    global ADAPTER
    if ADAPTER is None:
        spec = importlib.util.spec_from_file_location(
            "tabm_adapter", ROOT / "machine_learning/python/tabm_backend.py"
        )
        ADAPTER = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ADAPTER)
    adapter = ADAPTER
    categories = {k: v for k, v in job["categorical"].items() if k in features}
    if family == "TabM9":
        ref = pd.read_csv(DATA / "reduction/TabM_Path_k09_Evaluation.csv")
        reference = ref[[f"m{i}" for i in range(1, 31)]].to_numpy()[rows]
    else:
        reference = pd.read_csv(DATA / "models_TabM_Full.csv").to_numpy()[rows]
    fitted, display, maximum_error = [], [], 0.
    for mi in range(1, 31):
        prefix = (DATA / f"reduction/TabM_Path_k09_Final_m{mi:02d}"
                  if family == "TabM9" else DATA / f"models_TabM_m{mi:02d}")
        checkpoint = torch.load(str(prefix) + ".pt", map_location="cpu", weights_only=False)
        transformer = joblib.load(str(prefix) + ".preprocessing.joblib")
        if transformer.__class__.__name__ != "QuantileTransformer":
            raise ValueError("Coalition batching requires the fitted columnwise quantile transform.")
        network = adapter.make_model(checkpoint["parameters"], len(checkpoint["numeric"]), checkpoint["cardinalities"])
        network.load_state_dict(checkpoint["state"], strict=True)
        network.eval()
        train = pd.read_csv(job["inputs"][mi - 1]["train"])
        query = pd.read_csv(job["inputs"][mi - 1]["query"])
        a, b = query.iloc[rows][features], train.iloc[donors][features]
        numeric, cats = checkpoint["numeric"], checkpoint["categorical"]
        an = transformer.transform(a[numeric]).astype(np.float32)
        bn = transformer.transform(b[numeric]).astype(np.float32)
        mixed = a.iloc[[0]].copy()
        mixed.loc[:, numeric[::2]] = b.iloc[0][numeric[::2]].to_numpy()
        expected = an[[0]].copy()
        expected[:, ::2] = bn[0, ::2]
        assert np.allclose(transformer.transform(mixed[numeric]), expected, atol=1e-6)
        ac, bc = [], []
        for name in cats:
            mapping = {value: i for i, value in enumerate(categories[name])}
            ac.append(a[name].map(mapping).to_numpy(dtype=np.int64))
            bc.append(b[name].map(mapping).to_numpy(dtype=np.int64))
        state = dict(model=network,
                     x=torch.as_tensor(an, device="cuda"),
                     background=torch.as_tensor(bn, device="cuda"),
                     xc=torch.as_tensor(np.stack(ac, axis=1), device="cuda"),
                     bc=torch.as_tensor(np.stack(bc, axis=1), device="cuda"),
                     numeric_bits=torch.as_tensor([1 << features.index(f) for f in numeric], device="cuda"),
                     categorical_bits=torch.as_tensor([1 << features.index(f) for f in cats], device="cuda"))
        with torch.inference_mode():
            restored = []
            for start in range(0, len(rows), 512):
                logits = network(state["x"][start:start + 512], state["xc"][start:start + 512]).float()
                restored.append(logits.softmax(-1).mean(1)[:, 1].cpu().numpy())
        error = float(np.max(np.abs(np.concatenate(restored) - reference[:, mi - 1])))
        if error > 2e-5:
            raise ValueError(f"{family} fit {mi} prediction mismatch: {error}")
        maximum_error = max(maximum_error, error)
        fitted.append(state)
        display.append(a.to_numpy(dtype=float))
    return fitted, np.mean(display, axis=0), reference.mean(1), maximum_error


@torch.inference_mode()
def predict_coalitions(requests, ensemble, batch_size):
    probabilities = np.zeros(len(requests), dtype=float)
    for state in ensemble:
        for start in range(0, len(requests), batch_size):
            q, b, codes = torch.as_tensor(requests[start:start + batch_size], device="cuda", dtype=torch.long).T
            numeric_mask = (codes[:, None] & state["numeric_bits"]) != 0
            category_mask = (codes[:, None] & state["categorical_bits"]) != 0
            x = torch.where(numeric_mask, state["x"][q], state["background"][b])
            c = torch.where(category_mask, state["xc"][q], state["bc"][b])
            logits = state["model"](x, c).float()
            probabilities[start:start + len(q)] += logits.softmax(-1).mean(1)[:, 1].cpu().numpy() / len(ensemble)
    return probabilities


def run(family, limit=None, batch_size=1024):
    manifest = json.loads((DATA / "models_TAB_MANIFEST.json").read_text(encoding="utf-8"))
    plan = json.loads((DATA / "shap_full/SHAP_plan_v1.json").read_text(encoding="utf-8"))
    job = manifest["blocks"]["Full"]
    features = NINE if family == "TabM9" else job["features"]
    rows = np.arange(3038) if family == "TabM9" else np.asarray(plan["evaluation_rows_zero_based"])
    donors = np.asarray(plan["background_rows_zero_based"])
    if limit is not None:
        rows = rows[:limit]
    out = OUT / (family if limit is None else family + "_benchmark")
    out.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "RUN_STATUS.json", dict(stage="LOADING", family=family, n=len(rows), pid=os.getpid()))
    fitted, display, raw, parity = load_ensemble(family, job, features, rows, donors)
    intercept, slope = CALIBRATION[family]
    expected = expit(intercept + slope * logit(np.clip(raw, 1e-8, 1 - 1e-8)))
    completed = []
    started = time.monotonic()
    for block, start in enumerate(range(0, len(rows), 64)):
        target = out / f"block_{block:03d}.npz"
        indices = np.arange(start, min(start + 64, len(rows)))
        if target.exists():
            completed.append(target)
            continue
        requests = []

        def record(a):
            requests.append(np.asarray(a, dtype=np.int64))
            return np.zeros(len(a))

        explain_requests(record, indices, features, 123 + block)
        requests = np.unique(np.concatenate(requests), axis=0)
        raw_predictions = predict_coalitions(requests, fitted, batch_size)
        calibrated = expit(intercept + slope * logit(np.clip(raw_predictions, 1e-8, 1 - 1e-8)))
        request_keys = keys(requests, len(features))
        order = np.argsort(request_keys)
        sorted_keys, sorted_probabilities = request_keys[order], calibrated[order]

        def replay(a):
            requested_keys = keys(a, len(features))
            position = np.searchsorted(sorted_keys, requested_keys)
            if np.any(position >= len(sorted_keys)) or not np.array_equal(sorted_keys[position], requested_keys):
                raise ValueError("The SHAP replay requested an uncomputed coalition.")
            return sorted_probabilities[position]

        sv = explain_requests(replay, indices, features, 123 + block)
        residual = float(np.max(np.abs(sv.values.sum(1) + sv.base_values - expected[indices])))
        if residual > 2e-5:
            raise ValueError(f"Calibrated SHAP additivity mismatch: {residual}")
        temp = target.with_suffix(".tmp.npz")
        np.savez_compressed(temp, rows=rows[indices], phi=sv.values, baseline=sv.base_values,
                            prediction=expected[indices], values=display[indices],
                            sampling_variation=sv.error_std, features=np.asarray(features))
        os.replace(temp, target)
        completed.append(target)
        log(family, "BLOCK", block + 1, "RECORDS", start + len(indices), "OF", len(rows),
            "COALITIONS", len(requests), "SECONDS", round(time.monotonic() - started, 1), "RESIDUAL", residual)
        write_json(OUT / "RUN_STATUS.json", dict(stage="RUNNING", family=family, completed_records=start + len(indices), n=len(rows), pid=os.getpid()))
    arrays = [np.load(path, allow_pickle=False) for path in completed]
    phi = np.concatenate([z["phi"] for z in arrays])
    baseline = np.concatenate([z["baseline"] for z in arrays])
    assert phi.shape == (len(rows), len(features)) and np.isfinite(phi).all()
    frame = pd.DataFrame(phi, columns=features)
    frame.insert(0, "ExpectedValue", baseline)
    frame.insert(0, "Prediction", expected)
    frame.insert(0, "ValidationRow", rows + 1)
    frame.to_csv(out / "SHAP_values.csv", index=False)
    values = pd.DataFrame(display, columns=features)
    values.insert(0, "ValidationRow", rows + 1)
    values.to_csv(out / "display_values.csv", index=False)
    ranking = pd.DataFrame({"Variable": features, "MeanAbsSHAP": np.mean(np.abs(phi), axis=0)})
    ranking.sort_values("MeanAbsSHAP", ascending=False).to_csv(out / "importance.csv", index=False)
    metadata = dict(family=family, output_scale="Platt-calibrated probability", seed=123,
                    explanation_records=len(rows), imputations=30, members_per_fit=32,
                    background_records=8, permutation_pairs=8, features=features,
                    pooling="Mean member probabilities, mean 30 fits, then Platt calibration for every masked input",
                    masking="Common clinical-feature masks and training-donor identities across imputed inputs",
                    api="shap.Explainer(algorithm='permutation')", shap_version=importlib.metadata.version("shap"),
                    display_values="Across-imputation means for plotting only",
                    platt_intercept=intercept, platt_slope=slope,
                    maximum_frozen_prediction_error=parity,
                    maximum_additivity_error=float(np.max(np.abs(phi.sum(1) + baseline - expected))),
                    device=torch.cuda.get_device_name(0), precision="float32", seconds=time.monotonic() - started)
    write_json(out / "methods.json", metadata)

    if family == "TabM9":
        destination = DATA / "model_diagnostics"
        destination.mkdir(exist_ok=True)
        frame.rename(columns={"ValidationRow": "Row"}).to_csv(destination / "TabM9_SHAP_values.csv", index=False)
        values.rename(columns={"ValidationRow": "Row"}).to_csv(destination / "TabM9_SHAP_display_values.csv", index=False)
        ranking.sort_values("MeanAbsSHAP", ascending=False).to_csv(destination / "TabM9_SHAP_importance.csv", index=False)
        write_json(destination / "TabM9_SHAP_methods.json", metadata)
    else:
        destination = DATA / "shap_full"
        destination.mkdir(exist_ok=True)
        ranking.rename(columns={"Variable": "Feature"}).sort_values("MeanAbsSHAP", ascending=False).to_csv(
            destination / "TabM_OFFICIAL_v2_importance.csv", index=False)

    log(family, "COMPLETE", json.dumps(metadata))
    for a in arrays:
        a.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["TabM9", "TabM21", "both"], default="both")
    parser.add_argument("--benchmark", type=int)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    masker_test()
    try:
        for family in (["TabM9", "TabM21"] if args.family == "both" else [args.family]):
            run(family, args.benchmark, args.batch_size)
            torch.cuda.empty_cache()
        write_json(OUT / "RUN_STATUS.json", dict(stage="COMPLETE", pid=os.getpid(), benchmark=args.benchmark))
    except Exception as exc:
        write_json(OUT / "RUN_STATUS.json", dict(stage="FAILED", error=str(exc), pid=os.getpid()))
        raise
