"""Apply the frozen TabM ensemble to paired imputation-specific inputs."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "calculator"))
from predictor import Predictor


@torch.inference_mode()
def main():
    out = Path("results/supplementary_validation")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Predictor(ROOT / "calculator", device=device)
    data = pd.read_csv(out / "Inference_inputs.csv")
    records = data.loc[data.Imputation == 1, "Record"].tolist()
    if sorted(data.Imputation.unique()) != list(range(1, 31)):
        raise ValueError("Thirty completed input datasets are required.")
    probabilities = np.zeros((len(records), 30), dtype=np.float64)
    for m, (network, transformer, numeric) in enumerate(model.models, 1):
        x = data.loc[data.Imputation == m].reset_index(drop=True)
        if x.Record.tolist() != records:
            raise ValueError("Record order differs across imputations.")
        frame = model._validate_input(x[model.features])
        xn = transformer.transform(frame[numeric]).astype(np.float32)
        xc = frame[["sex"]].to_numpy(dtype=np.int64, copy=True)
        for start in range(0, len(frame), 512):
            batch = slice(start, start + 512)
            logits = network(torch.as_tensor(xn[batch], device=device),
                             torch.as_tensor(xc[batch], device=device)).float()
            probabilities[batch, m - 1] = logits.softmax(-1).mean(1)[:, 1].cpu().numpy()
    raw = probabilities.mean(axis=1)
    clipped = np.clip(raw, 1e-8, 1 - 1e-8)
    calibrated = expit(model.config["platt_intercept"] + model.config["platt_slope"] * np.log(clipped / (1 - clipped)))
    result = pd.DataFrame(probabilities, columns=[f"p_m{m:02d}" for m in range(1, 31)])
    result.insert(0, "Record", records)
    result["Raw_probability"] = raw
    result["Probability"] = calibrated
    result["Screen_positive"] = calibrated >= model.threshold
    result.to_csv(out / "Frozen_predictions.csv", index=False)


if __name__ == "__main__":
    main()
