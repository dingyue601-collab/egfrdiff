"""Training-fold permutation ranking and sequential feature reduction."""
from __future__ import annotations
import os
import os
os.environ.update(HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')
import gc
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch
import joblib
ROOT = Path.cwd()
OUT = ROOT / 'results/reduction'
RSCRIPT = Path(os.environ.get('RSCRIPT', 'Rscript'))
REPORT = ROOT / 'machine_learning/R/06_feature_reduction.R'
CHECKPOINT = Path('models/pretrained/tabicl-classifier-v2-20260212.ckpt')
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
TABM = None

def log(*args):
    print(time.strftime('%H:%M:%S'), *args, flush=True)

def config():
    job = json.loads((OUT / 'Reduction_job_v1.json').read_text(encoding='utf-8'))
    manifest = json.loads(Path(job['source_manifest']).read_text(encoding='utf-8'))
    assert job['seed'] == 123 and len(job['features']) == 21
    return (job, manifest)

def save_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)

def save_npz(path, **values):
    temporary = Path(str(path) + '.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **values)
    os.replace(temporary, path)

def release():
    gc.collect()
    torch.cuda.empty_cache()

def tabm_adapter():
    global TABM
    if TABM is None:
        spec = importlib.util.spec_from_file_location('reduction_tabm_adapter', ROOT / 'machine_learning/python/tabm_backend.py')
        TABM = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(TABM)
    return TABM

def r_command(*args):
    result = subprocess.run([str(RSCRIPT), '--vanilla', str(REPORT), *args], capture_output=True, encoding='utf-8', errors='replace')
    if result.stdout:
        print(result.stdout, flush=True)
    if result.returncode:
        raise RuntimeError(f'R60 {args}: {result.stderr[-4000:]}')

def frame_data(block, m, features):
    item = block['inputs'][m - 1]
    a = pd.read_csv(item['train'])
    b = pd.read_csv(item['query'])
    assert 'label' in a and 'label' not in b
    assert np.isfinite(a[features].to_numpy()).all() and np.isfinite(b[features].to_numpy()).all()
    return (a, b, a.label.to_numpy(dtype=np.int64))

def subset_spec(family, tag, features):
    job, _ = config()
    assert features == [x for x in job['features'] if x in features]
    path = OUT / f'{family}_{tag}_spec.json'
    obj = dict(Model=family, Tag=tag, features=features, seed=123)
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8')) == obj
    else:
        save_json(path, obj)

def restore_cv_tabm(prefix, features, categories):
    module = tabm_adapter()
    state = torch.load(str(prefix) + '.pt', map_location='cpu', weights_only=False)
    scaler = joblib.load(str(prefix) + '.preprocessing.joblib')
    net = module.make_model(state['parameters'], len(state['numeric']), state['cardinalities'])
    net.load_state_dict(state['state'], strict=True)
    net.eval()

    def predict(frame):
        xn = torch.as_tensor(scaler.transform(frame[state['numeric']]).astype(np.float32), device='cuda')
        xc = None
        if state['categorical']:
            codes = []
            for name in state['categorical']:
                mapped = frame[name].map({v: i for i, v in enumerate(categories[name])})
                assert not mapped.isna().any()
                codes.append(mapped.to_numpy(dtype=np.int64))
            xc = torch.as_tensor(np.stack(codes, axis=1), device='cuda')
        return module.predict(net, xn, xc)
    return (predict, (net, scaler))

def run_cv(family, tag, features):
    job, manifest = config()
    subset_spec(family, tag, features)
    reference = pd.read_csv(OUT / 'Training_reference_v1.csv')
    oof = np.full((len(reference), 5), np.nan)
    for k in range(1, 6):
        block = manifest['blocks'][f'Fold{k}']
        cats = {v: levels for v, levels in block['categorical'].items() if v in features}
        rows = np.asarray(block['query_rows'], dtype=int) - 1
        for m in range(1, 6):
            prefix = OUT / f'{family}_{tag}_CV_F{k}_m{m:02d}'
            target = Path(str(prefix) + '.npz')
            if target.exists():
                saved = dict(np.load(target, allow_pickle=False))
                assert saved['features'].tolist() == features
                probability = saved['prediction']
            else:
                a, b, y = frame_data(block, m, features)
                start = time.monotonic()
                probability, meta = tabm_adapter().fit_predict(a[features], b[features], y, None, features, cats, job['TabM_parameters'], 123 + 100 * (m - 1) + k, epochs=job['TabM_epochs'], save_prefix=prefix if tag == 'Full21' else None)
                save_npz(target, prediction=probability, features=np.asarray(features), metadata=np.asarray(json.dumps(meta)))
                log('CV_COMPLETE', family, tag, 'FOLD', k, 'MI', m, 'SECONDS', round(time.monotonic() - start, 1))
            assert len(probability) == len(rows) and np.isfinite(probability).all()
            oof[rows, m - 1] = probability
    assert np.isfinite(oof).all()
    frame = pd.DataFrame(oof, columns=[f'm{i}' for i in range(1, 6)])
    frame.insert(0, 'Fold', reference.Fold)
    frame.insert(0, 'Row', reference.Row)
    frame.to_csv(OUT / f'{family}_{tag}_OOF.csv', index=False)
    r_command('score', family, tag)

def run_final(family, tag, features):
    job, manifest = config()
    subset_spec(family, tag, features)
    block = manifest['blocks']['Full']
    cats = {v: levels for v, levels in block['categorical'].items() if v in features}
    cols = {}
    for m in range(1, 31):
        prefix = OUT / f'{family}_{tag}_Final_m{m:02d}'
        target = Path(str(prefix) + '.npz')
        if target.exists():
            saved = dict(np.load(target, allow_pickle=False))
            assert saved['features'].tolist() == features
            probability = saved['prediction']
        else:
            a, b, y = frame_data(block, m, features)
            start = time.monotonic()
            probability, meta = tabm_adapter().fit_predict(a[features], b[features], y, None, features, cats, job['TabM_parameters'], 123 + m, epochs=job['TabM_epochs'], save_prefix=prefix)
            save_npz(target, prediction=probability, features=np.asarray(features), metadata=np.asarray(json.dumps(meta)))
            log('FINAL_COMPLETE', family, tag, 'MI', m, 'OF30', 'SECONDS', round(time.monotonic() - start, 1))
        assert len(probability) == 3038 and np.isfinite(probability).all()
        cols[f'm{m}'] = probability
    result = pd.DataFrame(cols)
    result.insert(0, 'Row', np.arange(1, 3039))
    result.to_csv(OUT / f'{family}_{tag}_Evaluation.csv', index=False)

def training_pfi(family):
    job, manifest = config()
    features = job['features']
    reference = pd.read_csv(OUT / 'Training_reference_v1.csv')
    baseline = np.full(len(reference), np.nan)
    perturbed = np.full((len(reference), len(features), 3), np.nan)
    for k in range(1, 6):
        block = manifest['blocks'][f'Fold{k}']
        cats = block['categorical']
        ix = np.asarray(block['query_rows'], dtype=int) - 1
        permutations = pd.read_csv(OUT / f'Training_PFI_Fold{k}_permutations.csv').to_numpy(dtype=int)
        arrays = []
        bases = []
        for m in range(1, 6):
            target = OUT / f'{family}_Training_PFI_F{k}_m{m:02d}.npz'
            if target.exists():
                saved = dict(np.load(target, allow_pickle=False))
            else:
                start = time.monotonic()
                a, b, y = frame_data(block, m, features)
                basepath = OUT / f'{family}_Full21_CV_F{k}_m{m:02d}'
                predictor, keep = restore_cv_tabm(basepath, features, cats)
                pp = predictor(b)
                original = np.load(str(basepath) + '.npz', allow_pickle=False)['prediction']
                error = float(np.max(np.abs(pp - original)))
                assert error < 2e-05
                values = b[features].to_numpy()
                permuted = np.empty((len(features), 3, len(b)))
                for j, v in enumerate(features):
                    for r in range(3):
                        x = values.copy()
                        x[:, j] = values[permutations[:, r], j]
                        permuted[j, r] = predictor(pd.DataFrame(x, columns=features))
                saved = dict(pfi=permuted, baseline=pp, features=np.asarray(features), parity=np.asarray(error))
                save_npz(target, **saved)
                del predictor, keep
                release()
                log('TRAINING_PFI_COMPLETE', family, 'FOLD', k, 'MI', m, 'SECONDS', round(time.monotonic() - start, 1))
            assert saved['features'].tolist() == features
            arrays.append(saved['pfi'])
            bases.append(saved['baseline'])
        baseline[ix] = np.mean(bases, axis=0)
        perturbed[ix] = np.mean(arrays, axis=0).transpose(2, 0, 1)
    assert np.isfinite(baseline).all() and np.isfinite(perturbed).all()
    frame = pd.DataFrame({f + '__r' + str(r + 1): perturbed[:, j, r] for j, f in enumerate(features) for r in range(3)})
    frame.insert(0, 'Baseline', baseline)
    frame.insert(0, 'Fold', reference.Fold)
    frame.insert(0, 'Row', reference.Row)
    frame.to_csv(OUT / f'{family}_Training_PFI_predictions_v1.csv', index=False)
    r_command('rank', family)

def recursive_path(family):
    job, _ = config()
    ranking = pd.read_csv(OUT / f'{family}_Training_PFI_rank_v1.csv')
    assert len(ranking) == 21 and set(ranking.Feature) == set(job['features'])
    for k in range(20, 0, -1):
        retained = set(ranking.sort_values('Rank').Feature.iloc[:k])
        features = [v for v in job['features'] if v in retained]
        run_cv(family, f'Path_k{k:02d}', features)
        log('RECURSIVE_SET_COMPLETE', family, 'FEATURES', k)
    r_command('select', family)

if __name__ == "__main__":
    mode = sys.argv[1]
    job, _ = config()
    if mode == "baseline":
        run_cv("TabM", "Full21", job["features"])
    elif mode == "rank":
        training_pfi("TabM")
    elif mode == "path":
        recursive_path("TabM")
    elif mode == "final":
        selected = json.loads((OUT / "TabM_Selection_v1.json").read_text())["Selected"]
        if isinstance(selected, dict):
            selected = [selected]
        seen = set()
        for specification in selected:
            tag = specification["Tag"]
            if tag in seen or tag == "Full21":
                continue
            seen.add(tag)
            run_final("TabM", tag, specification["Features"])
    else:
        raise ValueError(mode)
