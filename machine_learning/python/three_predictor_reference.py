"""Three-predictor TabM reference using the existing training schedule."""
import importlib.util
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

ROOT = Path.cwd()
OUT = ROOT / "results/clinical_reference_tabm3"
FEATURES = ["age", "sex", "scr"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("native_backend", ROOT / "machine_learning/python/feature_reduction.py")
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    assert backend.torch.cuda.is_available()
    job, manifest = backend.config()
    status = {"stage": "CV", "completed": 0, "total": 55, "features": FEATURES,
              "parameters": job["TabM_parameters"], "epochs": job["TabM_epochs"],
              "device": backend.torch.cuda.get_device_name(0), "seed": 123}
    backend.save_json(OUT / "RUN_STATUS.json", status)
    try:
        reference = pd.read_csv(backend.OUT / "Training_reference_v1.csv")
        oof = np.full((len(reference), 5), np.nan)
        for fold in range(1, 6):
            block = manifest["blocks"][f"Fold{fold}"]
            rows = np.asarray(block["query_rows"], dtype=int) - 1
            cats = {k: v for k, v in block["categorical"].items() if k in FEATURES}
            for m in range(1, 6):
                prefix = OUT / f"TabM3_F{fold}_m{m:02d}"
                p = fit(backend, job, block, cats, m, prefix, 123 + 100 * (m - 1) + fold)
                oof[rows, m - 1] = p
                status.update(completed=status["completed"] + 1, fold=fold, member=m)
                backend.save_json(OUT / "RUN_STATUS.json", status)
        assert np.isfinite(oof).all()
        pd.DataFrame(oof, columns=[f"m{i}" for i in range(1, 6)]).to_csv(OUT / "TabM3_OOF.csv", index=False)
        status["stage"] = "Final"
        block = manifest["blocks"]["Full"]
        cats = {k: v for k, v in block["categorical"].items() if k in FEATURES}
        final = {}
        for m in range(1, 31):
            prefix = OUT / f"TabM3_Final_m{m:02d}"
            final[f"m{m}"] = fit(backend, job, block, cats, m, prefix, 123 + m)
            pd.DataFrame(final).to_csv(OUT / "TabM3_Validation.csv", index=False)
            status.update(completed=status["completed"] + 1, member=m)
            backend.save_json(OUT / "RUN_STATUS.json", status)
        status["stage"] = "Predictions_complete"
        backend.save_json(OUT / "RUN_STATUS.json", status)
    except Exception:
        status.update(stage="Failed", error=traceback.format_exc())
        backend.save_json(OUT / "RUN_STATUS.json", status)
        raise


def fit(backend, job, block, cats, m, prefix, seed):
    path = prefix.with_suffix(".npz")
    if path.exists():
        saved = np.load(path, allow_pickle=False)
        assert saved["features"].tolist() == FEATURES
        return saved["prediction"]
    a, b, y = backend.frame_data(block, m, FEATURES)
    p, metadata = backend.tabm_adapter().fit_predict(a[FEATURES], b[FEATURES], y, None,
        FEATURES, cats, job["TabM_parameters"], seed, epochs=job["TabM_epochs"], save_prefix=prefix)
    assert len(p) == len(b) and np.isfinite(p).all()
    backend.save_npz(path, prediction=p, features=np.asarray(FEATURES), metadata=np.asarray(json.dumps(metadata)))
    print(time.strftime("%H:%M:%S"), prefix.name, "seconds", round(metadata["seconds"], 1), flush=True)
    return p


if __name__ == "__main__":
    main()
