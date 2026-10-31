"""Native TabM training, preprocessing and ensemble prediction."""
from __future__ import annotations
import os
import gc
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import time
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONIOENCODING='utf-8', DO_NOT_TRACK='1', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
import joblib
import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import tabm
from rtdl_num_embeddings import PeriodicEmbeddings
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
optuna.logging.set_verbosity(optuna.logging.WARNING)

def log(*args):
    print(time.strftime('%H:%M:%S'), *args, flush=True)

def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def release():
    gc.collect()
    torch.cuda.empty_cache()

def prepare(a, b, features, categories, pa, seed):
    cat = [c for c in features if c in categories]
    num = [c for c in features if c not in categories]
    scaler = StandardScaler() if pa['scaler'] == 'standard' else QuantileTransformer(n_quantiles=min(1000, len(a)), output_distribution='normal', random_state=seed)
    x = torch.as_tensor(scaler.fit_transform(a[num]).astype(np.float32), device='cuda')
    q = torch.as_tensor(scaler.transform(b[num]).astype(np.float32), device='cuda')
    ac, bc, cardinalities = (None, None, [])
    if cat:
        aa, bb = ([], [])
        for c in cat:
            levels = list(categories[c])
            mapping = {v: i for i, v in enumerate(levels)}
            aa.append(a[c].map(mapping).to_numpy(dtype=np.int64))
            bb.append(b[c].map(mapping).to_numpy(dtype=np.int64))
            cardinalities.append(len(levels))
        ac = torch.as_tensor(np.stack(aa, axis=1), device='cuda')
        bc = torch.as_tensor(np.stack(bb, axis=1), device='cuda')
    return (x, q, ac, bc, scaler, num, cat, cardinalities)

def make_model(pa, n_num, cardinalities):
    embedding = None
    if pa['embedding'] == 'periodic':
        embedding = PeriodicEmbeddings(n_num, d_embedding=16, n_frequencies=16, frequency_init_scale=0.01, lite=False)
    return tabm.TabM.make(n_num_features=n_num, cat_cardinalities=cardinalities or None, d_out=2, num_embeddings=embedding, arch_type='tabm', k=32, d_block=pa['width'], n_blocks=pa['depth'], dropout=pa['dropout']).to('cuda')

def predict(net, q, cat):
    net.eval()
    with torch.inference_mode():
        return np.concatenate([net(q[i:i + 256], cat[i:i + 256] if cat is not None else None).softmax(-1).mean(1)[:, 1].cpu().numpy() for i in range(0, len(q), 256)]).astype(float)

def fit_predict(a, b, y, labels, features, cats, pa, seed, epochs=None, save_prefix=None):
    assert torch.cuda.is_available() and 'label' not in b
    y = np.array(y, dtype=np.int64, copy=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    x, q, ac, bc, scaler, num, cat, card = prepare(a, b, features, cats, pa, seed)
    net = make_model(pa, len(num), card)
    opt = torch.optim.AdamW(net.parameters(), lr=pa['learning_rate'], weight_decay=pa['weight_decay'], foreach=False)
    yt = torch.as_tensor(y, dtype=torch.long, device='cuda')
    weights = torch.tensor([1.0, pa['positive_weight']], dtype=torch.float32, device='cuda')
    rng = np.random.default_rng(seed)
    best, bestloss, stale, chosen, state = (-np.inf, np.inf, 0, 0, None)
    start = time.monotonic()
    if epochs is None and labels is None:
        raise ValueError('Early stopping requires training-fold labels')
    for ep in range(1, (epochs if epochs is not None else 200) + 1):
        net.train()
        order = rng.permutation(len(y))
        for i in range(0, len(y), pa['batch_size']):
            ix = torch.as_tensor(order[i:i + pa['batch_size']], device='cuda')
            opt.zero_grad(set_to_none=True)
            logits = net(x[ix], ac[ix] if ac is not None else None)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), yt[ix].repeat_interleave(logits.shape[1]), weight=weights)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite TabM loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        if epochs is None:
            probability = predict(net, q, bc)
            score, ll = (roc_auc_score(labels, probability), log_loss(labels, probability, labels=[0, 1]))
            if score > best + 0.0002 or (abs(score - best) < 1e-12 and ll < bestloss):
                best, bestloss, stale, chosen = (score, ll, 0, ep)
                state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                stale += 1
            if stale >= 20:
                break
    if epochs is None:
        net.load_state_dict(state)
    else:
        chosen = epochs
    probability = predict(net, q, bc)
    md = dict(selected_epoch=int(chosen), evaluated_epochs=int(ep), seconds=time.monotonic() - start, members=32, precision='float32', loss='mean member-wise cross entropy')
    if save_prefix is not None:
        state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        record = dict(state=state, parameters=pa, numeric=num, categorical=cat, cardinalities=card)
        torch.save(record, str(save_prefix) + '.pt')
        joblib.dump(scaler, str(save_prefix) + '.preprocessing.joblib')
        restored = make_model(pa, len(num), card)
        loaded = torch.load(str(save_prefix) + '.pt', map_location='cpu', weights_only=False)
        restored.load_state_dict(loaded['state'])
        delta = float(np.max(np.abs(predict(restored, q[:32], bc[:32] if bc is not None else None) - probability[:32])))
        if delta > 1e-06:
            raise ValueError(f'TabM checkpoint parity {delta}')
        md['reload_max_abs_difference'] = delta
        del restored, loaded
    del net, opt, x, q, ac, bc, yt, weights, state
    release()
    return (probability, md)
