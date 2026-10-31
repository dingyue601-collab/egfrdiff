"""Inference for the nine-variable TabM eGFR-discordance model."""
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
import torch

VENDOR = Path(__file__).resolve().parent / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
import tabm
import rtdl_num_embeddings


class Predictor:
    """Load the fitted ensemble. Class 1 denotes relative eGFRdiff < -30%."""

    def __init__(self, directory, device="cpu"):
        self.directory = Path(directory).resolve()
        self._verify_artifacts()
        self.config = json.loads(
            (self.directory / "model_config.json").read_text(encoding="utf-8")
        )
        self.features = self.config["features"]
        self.classes_ = np.asarray(self.config["classes"])
        self.threshold = float(self.config["threshold80"])
        self.model_version = self.config["version"]
        self.tabm_version = tabm.__version__
        self.device = torch.device(device)
        self.model_count = self.config["models"]
        if self.features != [
            "age", "sex", "scr", "hba1c", "hemoglobin", "uric_acid",
            "albumin", "alp", "ada"
        ] or self.config["classes"] != [0, 1]:
            raise ValueError("The model's feature schema or class encoding is incompatible.")
        if self.tabm_version != "0.0.3" or rtdl_num_embeddings.__version__ != "0.0.12":
            raise ValueError("Use the bundled TabM and numerical-embedding modules.")

        torch.manual_seed(123)
        np.random.seed(123)
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.models = []
        numeric = [name for name in self.features if name != "sex"]

        for m in range(1, self.model_count + 1):
            path = self.directory / "models" / f"TabM_Path_k09_Final_m{m:02d}"
            state = torch.load(
                path.with_suffix(".pt"), map_location="cpu", weights_only=True
            )
            if (state["numeric"] != numeric or state["categorical"] != ["sex"]
                    or state["cardinalities"] != [2]):
                raise ValueError(f"Incompatible feature order in checkpoint {m}.")
            parameters = state["parameters"]
            embeddings = rtdl_num_embeddings.PeriodicEmbeddings(
                len(numeric), d_embedding=16, n_frequencies=16,
                frequency_init_scale=.01, lite=False
            )
            model = tabm.TabM.make(
                n_num_features=len(numeric), cat_cardinalities=state["cardinalities"],
                d_out=2, num_embeddings=embeddings, arch_type="tabm",
                k=self.config["members"], d_block=parameters["width"],
                n_blocks=parameters["depth"], dropout=parameters["dropout"]
            )
            model.load_state_dict(state["state"], strict=True)
            model.to(device=self.device, dtype=torch.float32).eval()

            # joblib uses pickle; load only artifacts from a trusted release.
            transformer = joblib.load(path.with_suffix(".preprocessing.joblib"))
            if list(transformer.feature_names_in_) != numeric:
                raise ValueError(f"Incompatible preprocessing feature order in checkpoint {m}.")
            self.models.append((model, transformer, numeric))

    def _verify_artifacts(self):
        """Detect missing or changed model files before deserialization."""
        manifest_path = self.directory / "model_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("model_manifest.json is missing; use a complete release.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {"model_config.json", "vendor/tabm.py", "vendor/rtdl_num_embeddings.py"}
        expected.update(
            f"models/TabM_Path_k09_Final_m{m:02d}{suffix}"
            for m in range(1, 31)
            for suffix in (".pt", ".preprocessing.joblib")
        )
        if set(manifest) != expected:
            raise ValueError("The model manifest does not describe the expected 30-model release.")
        for name in sorted(expected):
            path = self.directory / name
            if not path.is_file():
                raise FileNotFoundError(f"Missing model artifact: {name}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != manifest[name]:
                raise ValueError(f"Model artifact checksum mismatch: {name}")

    def _validate_input(self, data):
        if isinstance(data, Mapping) and all(
                np.isscalar(value) or value is None for value in data.values()):
            data = [data]
        frame = pd.DataFrame(data)
        if frame.empty:
            raise ValueError("Provide at least one complete record.")
        if frame.columns.has_duplicates:
            raise ValueError("Input column names must be unique.")
        missing = [name for name in self.features if name not in frame.columns]
        if missing:
            raise ValueError("Missing predictors: " + ", ".join(missing))
        try:
            frame = frame.loc[:, self.features].astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("All nine predictors must be numeric; sex is 0 or 1.") from exc
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError("All nine predictors must be finite and nonmissing.")
        if not frame["sex"].isin([0, 1]).all():
            raise ValueError("Sex must be 0 (female) or 1 (male).")
        if not frame["age"].between(18, 120).all():
            raise ValueError("Age must be between 18 and 120 years.")
        positive = [name for name in self.features if name not in ("sex", "ada")]
        if (frame[positive] <= 0).any().any() or (frame["ada"] < 0).any():
            raise ValueError("Measurements must be positive; ADA may be zero.")
        if (frame["hba1c"] > 100).any():
            raise ValueError("HbA1c must not exceed 100 percent.")
        return frame

    @torch.inference_mode()
    def predict_proba(self, data, batch_size=512):
        """Return calibrated probabilities and a screening flag as a DataFrame.

        Input units follow model_config.json. No missing values are imputed.
        The clinical target is probability_negative_discordance, not p_negative.
        The p_positive/p_negative aliases retain the v6 class-1/class-0 meaning.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        frame = self._validate_input(data)
        raw = np.zeros(len(frame), dtype=np.float64)
        xc = frame[["sex"]].to_numpy(dtype=np.int64, copy=True)
        for model, transformer, numeric in self.models:
            xn = transformer.transform(frame[numeric]).astype(np.float32)
            if not np.isfinite(xn).all():
                raise ValueError("Preprocessing produced nonfinite values.")
            for start in range(0, len(frame), batch_size):
                stop = min(start + batch_size, len(frame))
                logits = model(
                    torch.as_tensor(xn[start:stop], device=self.device),
                    torch.as_tensor(xc[start:stop], device=self.device)
                ).float()
                # Average member probabilities, then average the fitted models.
                raw[start:stop] += (
                    logits.softmax(dim=-1).mean(dim=1)[:, 1].cpu().numpy()
                    / self.model_count
                )
        if not np.isfinite(raw).all():
            raise RuntimeError("The model produced nonfinite probabilities.")
        clipped = np.clip(raw, 1e-8, 1 - 1e-8)
        linear = (
            self.config["platt_intercept"]
            + self.config["platt_slope"] * np.log(clipped / (1 - clipped))
        )
        probability = expit(linear)
        selected = probability >= self.threshold
        return pd.DataFrame({
            "probability_negative_discordance": probability,
            "probability_no_negative_discordance": 1 - probability,
            "raw_probability": raw,
            "threshold": self.threshold,
            "screen_positive": selected,
            # Backward-compatible v6 names; positive means target class 1.
            "p_positive": probability,
            "p_negative": 1 - probability,
            "recommend_testing": selected,
        })
