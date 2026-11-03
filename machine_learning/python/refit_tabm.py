"""Refit the published full-input TabM specification."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

spec = importlib.util.spec_from_file_location("tabm_backend", "machine_learning/python/tabm_backend.py")
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)
manifest = json.loads(Path("results/models_TAB_MANIFEST.json").read_text())
published = json.loads(Path("docs/published_parameters.json").read_text())["TabM"]
parameters = published["parameters"]
full = manifest["blocks"]["Full"]
features, categories = full["features"], full["categorical"]
oof = np.full(6852, np.nan)
scores = []
for k in range(1, 6):
    block = manifest["blocks"][f"Fold{k}"]
    labels = pd.read_csv(block["query_labels"]).label.to_numpy(dtype=np.int64)
    predictions = []
    for m in range(5):
        prefix = Path(f"results/models_TabM_replay_F{k}_m{m+1:02d}")
        target = prefix.with_suffix(".npz")
        if target.exists():
            probability = np.load(target)["prediction"]
        else:
            a = pd.read_csv(block["inputs"][m]["train"])
            b = pd.read_csv(block["inputs"][m]["query"])
            probability, metadata = backend.fit_predict(a[features], b, a.label.to_numpy(dtype=np.int64),
                labels, features, categories, parameters, 123 + 100*m + k)
            np.savez_compressed(target, prediction=probability)
        predictions.append(probability)
    probability = np.mean(predictions, axis=0)
    oof[np.asarray(block["query_rows"])-1] = probability
    scores.append(float(roc_auc_score(labels, probability)))

result = dict(parameters=parameters, epochs=published["epochs"], features=features,
              cv_auc=float(np.mean(scores)), oof=oof.tolist(), trial_table=published["trial_table"])
Path("results/models_TabM_TUNING.json").write_text(json.dumps(result), encoding="utf-8")
query, train = {}, {}
for m in range(30):
    target = Path(f"results/models_TabM_replay_Final_m{m+1:02d}.npz")
    if target.exists():
        z = np.load(target)
        query[f"m{m+1}"], train[f"m{m+1}"] = z["query"], z["train"]
        continue
    a = pd.read_csv(full["inputs"][m]["train"])
    b = pd.read_csv(full["inputs"][m]["query"])
    joined = pd.concat([b, a[features]], ignore_index=True)
    probability, metadata = backend.fit_predict(a[features], joined, a.label.to_numpy(dtype=np.int64),
        None, features, categories, parameters, 124+m, epochs=published["epochs"],
        save_prefix=Path(f"results/models_TabM_m{m+1:02d}"))
    query[f"m{m+1}"], train[f"m{m+1}"] = probability[:len(b)], probability[len(b):]
    np.savez_compressed(target, query=query[f"m{m+1}"], train=train[f"m{m+1}"])
    pd.DataFrame(query).to_csv("results/models_TabM_Full.csv", index=False)
    pd.DataFrame(train).to_csv("results/models_TabM_Train.csv", index=False)
