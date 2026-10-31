"""Tabular-model tuning and patient-fold evaluation."""
from __future__ import annotations
import os
import gc
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import warnings

os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TABPFN_DISABLE_TELEMETRY="1",
    DO_NOT_TRACK="1", TABPFN_NO_BROWSER="1", TABPFN_MODEL_CACHE_SIZE="1", OMP_NUM_THREADS="2",
    MKL_NUM_THREADS="2", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

ASSETS = Path("models/pretrained")
CHECKPOINTS = {"TabPFN": ASSETS / "tabpfn-v3.5-20260909.safetensors",
    "TabICL": ASSETS / "tabicl-classifier-v2-20260212.ckpt"}
ROOT = Path.cwd()
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
optuna.logging.set_verbosity(optuna.logging.WARNING)

def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)

def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

def release():
    gc.collect()
    torch.cuda.empty_cache()

def load_block(job, m):
    z = job["inputs"][m]
    a, b = pd.read_csv(z["train"]), pd.read_csv(z["query"])
    assert "label" in a and "label" not in b
    features = job["features"]
    assert np.isfinite(a[features].to_numpy()).all() and np.isfinite(b[features].to_numpy()).all()
    return a, b, a.label.to_numpy(dtype=np.int64)

def foundation_predict(family, a, b, features, cats, pa, seed, train_prediction=False):
    """Complete permitted training context, fixed float32; only query batching changes memory."""
    assert CHECKPOINTS[family].is_file() and torch.cuda.is_available()
    torch.manual_seed(seed); np.random.seed(seed)
    x, q = a[features].copy(), b[features].copy()
    y = a.label.to_numpy(dtype=np.int64)
    start = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if family == "TabPFN":
            from tabpfn import TabPFNClassifier
            model = TabPFNClassifier(model_path=str(CHECKPOINTS[family]), device="cuda",
                n_estimators=pa["members"], auto_scale_n_estimators=False,
                categorical_features_indices=[i for i,c in enumerate(features) if c in cats],
                random_state=seed, n_preprocessing_jobs=1, show_progress_bar=False,
                inference_precision=torch.float32, fit_mode="fit_with_cache", memory_saving_mode=True,
                keep_cache_on_device=False, kv_cache_precision=None,
                softmax_temperature=pa["temperature"], average_before_softmax=pa["average_logits"],
                balance_probabilities=pa["balance_probabilities"], tuning_config=None)
        else:
            from tabicl import TabICLClassifier
            for c, levels in cats.items():
                x[c] = pd.Categorical(x[c], categories=levels)
                q[c] = pd.Categorical(q[c], categories=levels)
            model = TabICLClassifier(model_path=str(CHECKPOINTS[family]), allow_auto_download=False,
                device="cuda", n_estimators=pa["members"], norm_methods=pa["norm_methods"],
                outlier_threshold=pa["outlier_threshold"], softmax_temperature=pa["temperature"],
                average_logits=pa["average_logits"], feat_shuffle_method="latin", class_shuffle_method="shift",
                batch_size=1, kv_cache=True, use_amp=False, n_jobs=1,
                offload_mode="auto", random_state=seed, verbose=False)
        model.fit(x, y)
        positive = np.flatnonzero(np.asarray(model.classes_) == 1)
        assert len(positive) == 1
        col = int(positive[0])
        def predict(frame):
            batch = 64
            while True:
                try:
                    values = [model.predict_proba(frame.iloc[i:i+batch])[:,col]
                              for i in range(0,len(frame),batch)]
                    return np.concatenate(values)
                except torch.cuda.OutOfMemoryError:
                    if batch <= 8:
                        raise
                    batch //= 2
                    torch.cuda.empty_cache()
                    log(family,"QUERY_BATCH_REDUCED",batch,"CONTEXT_MEMBERS_PRECISION_UNCHANGED")
        prob = predict(q)
        tp = predict(x) if train_prediction else None
        parity = float(np.max(np.abs(model.predict_proba(q.iloc[:16])[:,col]-prob[:16])))
        assert parity < 1e-5 and len(prob) == len(q) and np.isfinite(prob).all()
        assert ((prob >= 0) & (prob <= 1)).all()
        meta = dict(seconds=time.monotonic()-start, train_n=len(x), query_n=len(q), precision="float32",
            members=pa["members"], checkpoint=str(CHECKPOINTS[family]), query_batch_parity=parity,
            fit_kind="Local pretrained in-context inference tuning; NO weight fine-tuning",
            warnings=sorted({str(w.message) for w in caught}))
    del model, x, q, y
    release()
    return prob, tp, meta

def load_tabm():
    path = ROOT / "machine_learning/python/tabm_backend.py"
    spec = importlib.util.spec_from_file_location("established_tabm_backend",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def parameters(family, trial):
    if family == "TabM":
        return dict(width=trial.suggest_categorical("width",[128,256,512]),
            depth=trial.suggest_int("depth",2,4),dropout=trial.suggest_float("dropout",0,.35),
            learning_rate=trial.suggest_float("learning_rate",.0002,.003,log=True),
            weight_decay=trial.suggest_float("weight_decay",.000001,.01,log=True),
            batch_size=trial.suggest_categorical("batch_size",[128,256,512]),
            positive_weight=trial.suggest_float("positive_weight",1,3),
            scaler=trial.suggest_categorical("scaler",["standard","quantile"]),
            embedding=trial.suggest_categorical("embedding",["none","periodic"]))
    pa = dict(members=trial.suggest_categorical("members",[16,32]),
        temperature=trial.suggest_float("temperature",.5,1.4),
        average_logits=trial.suggest_categorical("average_logits",[False,True]))
    if family == "TabPFN":
        pa["balance_probabilities"] = trial.suggest_categorical("balance_probabilities",[False,True])
    else:
        pa["norm_methods"] = {"mixed":["none","power"],"none":["none"],"power":["power"]}[
            trial.suggest_categorical("normalization",["mixed","none","power"])]
        pa["outlier_threshold"] = trial.suggest_float("outlier_threshold",3,6)
    return pa

def run(manifest, family):
    assert family in ("TabPFN","TabICL","TabM") and torch.cuda.is_available()
    prefix = Path(manifest["prefix"])
    seed = int(manifest["seed"])
    assert seed == 123, "This version is reserved for the master-seed-123 run"
    blocks = manifest["blocks"]
    full = blocks["Full"]
    features, cats = full["features"], full["categorical"]
    if isinstance(cats,list):
        cats = {}
    module = load_tabm() if family == "TabM" else None
    if family != "TabM":
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
    max_trials, min_trials, block_size = (32,16,8) if family == "TabM" else (24,12,6)
    tuning_path = Path(str(prefix)+f"_{family}_TUNING.json")
    def perform(a,b,y,job,pa,s,labels=None,epochs=None,final_prefix=None):
        if family == "TabM":
            return module.fit_predict(a[features],b,y,labels,features,cats,pa,s,epochs=epochs,save_prefix=final_prefix)
        prob,_,meta = foundation_predict(family,a,b,features,cats,pa,s)
        return prob,meta
    def score(pa, ms, trial=None, final_epochs=None):
        scores, briers, epochs, metrics = [],[],[],[]
        oof = np.full(sum(len(pd.read_csv(blocks[f"Fold{k}"]["query_labels"])) for k in range(1,6)),np.nan)
        for k in range(1,6):
            job = blocks[f"Fold{k}"]
            labels = pd.read_csv(job["query_labels"]).label.to_numpy(dtype=np.int64)
            preds = []
            for m in ms:
                a,b,y = load_block(job,m)
                prob,meta = perform(a,b,y,job,pa,seed+100*m+k,labels=labels,epochs=final_epochs)
                preds.append(prob);metrics.append(meta)
                if "selected_epoch" in meta:
                    epochs.append(meta["selected_epoch"])
                log(family,"DEVELOPMENT",k,"MI",m+1,"SECONDS",round(meta["seconds"]))
            probability = np.mean(preds,axis=0)
            scores.append(float(roc_auc_score(labels,probability)))
            briers.append(float(brier_score_loss(labels,probability)))
            ix = np.asarray(job["query_rows"],dtype=int)-1
            oof[ix] = probability
            if trial is not None:
                trial.report(float(np.mean(scores)),step=k)
                if k >= 2 and trial.should_prune():
                    raise optuna.TrialPruned()
        assert np.isfinite(oof).all()
        return dict(score=float(np.mean(scores)),brier=float(np.mean(briers)),fold_scores=scores,
            oof=oof.tolist(),epochs=epochs,execution=metrics)
    study = optuna.create_study(study_name=family,direction="maximize",
        storage="sqlite:///"+str(prefix)+f"_OPTUNA_{family}.sqlite3",load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed,n_startup_trials=8),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8,n_warmup_steps=2))
    if tuning_path.exists():
        selected = json.loads(tuning_path.read_text(encoding="utf-8"))
    else:
        if len(study.trials) == 0:
            if family == "TabM":
                study.enqueue_trial(dict(width=128,depth=2,dropout=.15,learning_rate=.001,weight_decay=.001,
                    batch_size=256,positive_weight=1.,scaler="quantile",embedding="none"))
            elif family == "TabPFN":
                study.enqueue_trial(dict(members=16,temperature=.9,average_logits=False,balance_probabilities=False))
            else:
                study.enqueue_trial(dict(members=16,temperature=.9,average_logits=False,normalization="mixed",outlier_threshold=4.))
        best, blockbest, stalled = -np.inf,-np.inf,0
        while len(study.trials) < max_trials:
            trial = study.ask();pa = parameters(family,trial);start=time.monotonic()
            record = dict(trial=trial.number,parameters=pa,state="Complete")
            try:
                z = score(pa,[0],trial)
                study.tell(trial,z["score"]);record["result"]=z;best=max(best,z["score"])
            except optuna.TrialPruned:
                study.tell(trial,state=optuna.trial.TrialState.PRUNED);record["state"]="Pruned"
            except Exception as exc:
                study.tell(trial,state=optuna.trial.TrialState.FAIL)
                record.update(state="Failed",error=repr(exc));release()
            record["seconds"]=time.monotonic()-start
            save(str(prefix)+f"_{family}_TRIAL{trial.number:03d}.json",record)
            log(family,"TRIAL_DONE",trial.number,record["state"],"BEST",round(best,4))
            n=len(study.trials)
            if n>=min_trials and n%block_size==0:
                stalled=stalled+1 if best-blockbest<.002 else 0
                blockbest=best
                if stalled>=2:
                    break
        records=[json.loads(p.read_text(encoding="utf-8")) for p in prefix.parent.glob(prefix.name+f"_{family}_TRIAL*.json")]
        complete=sorted([r for r in records if r["state"]=="Complete"],key=lambda r:r["result"]["score"],reverse=True)
        if not complete:
            raise RuntimeError(f"{family}: no complete configuration; failures are not evidence of poor model performance")
        confirmations=[]
        for rec in complete[:3]:
            path=Path(str(prefix)+f"_{family}_CONFIRM{rec['trial']:03d}.json")
            if path.exists():
                conf=json.loads(path.read_text(encoding="utf-8"))
            else:
                conf=dict(trial=rec["trial"],parameters=rec["parameters"],result=score(rec["parameters"],range(5)))
                save(path,conf)
            confirmations.append(conf)
        chosen=sorted(confirmations,key=lambda r:(-r["result"]["score"],r["result"]["brier"]))[0]
        epochs=chosen["result"]["epochs"]
        selected=dict(parameters=chosen["parameters"],cv_auc=chosen["result"]["score"],
            epochs=int(round(float(np.median(epochs)))) if epochs else None,
            oof=chosen["result"]["oof"],features=features,max_trials=max_trials,
            selection="Training-side grouped CV, confirmed across five imputations; evaluation labels unavailable",
            trial_table=[dict(Trial=r["trial"],Status=r["state"],Score=r.get("result",{}).get("score"),
                Parameters=json.dumps(r["parameters"])) for r in records])
        save(tuning_path,selected)
        log(family,"PARAMETERS_FROZEN","CV_AUC",round(selected["cv_auc"],4))
    qpath,tpath=Path(str(prefix)+f"_{family}_Full.csv"),Path(str(prefix)+f"_{family}_Train.csv")
    meta_path=Path(str(prefix)+f"_{family}_FINAL.json")
    qp,tp,execution={},{},[]
    if qpath.exists():
        qp={k:v.to_numpy() for k,v in pd.read_csv(qpath).items()}
        tp={k:v.to_numpy() for k,v in pd.read_csv(tpath).items()}
        execution=json.loads(meta_path.read_text(encoding="utf-8"))["execution"]
    for m in range(30):
        if f"m{m+1}" in qp:
            continue
        a,b,y=load_block(full,m)
        if family=="TabM":
            q=pd.concat([b,a[features]],ignore_index=True)
            pred,meta=perform(a,q,y,full,selected["parameters"],seed+m+1,
                epochs=selected["epochs"],final_prefix=Path(str(prefix)+f"_TabM_m{m+1:02d}"))
            prob,train=pred[:len(b)],pred[len(b):]
        else:
            prob,train,meta=foundation_predict(family,a,b,features,cats,selected["parameters"],seed+m+1,train_prediction=True)
        qp[f"m{m+1}"],tp[f"m{m+1}"]=prob,train
        execution.append(dict(imputation=m+1,**meta))
        pd.DataFrame(qp).to_csv(qpath,index=False,encoding="utf-8")
        pd.DataFrame(tp).to_csv(tpath,index=False,encoding="utf-8")
        save(meta_path,dict(family=family,features=features,members=32 if family=="TabM" else selected["parameters"]["members"],
            selected=selected,execution=execution,precision="float32",final_imputations=30,
            training_prediction="Apparent in-context self-prediction for TabPFN/TabICL; development OOF also retained",
            versions={n:importlib.metadata.version(n) for n in ("torch","tabpfn","tabicl","optuna","scikit-learn")}))
        log(family,"FINAL_MI_COMPLETE",m+1,"OF 30","SECONDS",round(meta["seconds"]))

if __name__=="__main__":
    if sys.argv[1]=="smoke":
        family=sys.argv[2];rng=np.random.default_rng(123)
        features=[f"x{k}" for k in range(8)]
        a=pd.DataFrame(rng.normal(size=(128,8)),columns=features)
        b=pd.DataFrame(rng.normal(size=(24,8)),columns=features)
        y=(a.x0+.4*a.x1+rng.normal(size=128)>.2).to_numpy(dtype=np.int64)
        if family=="TabM":
            module=load_tabm()
            pa=dict(width=128,depth=2,dropout=.15,learning_rate=.001,weight_decay=.001,
                batch_size=128,positive_weight=1.,scaler="quantile",embedding="none")
            pred,meta=module.fit_predict(a,b,y,None,features,{},pa,123,epochs=2)
        else:
            a["label"]=y
            pa=dict(members=16,temperature=.9,average_logits=False,
                balance_probabilities=False,norm_methods=["none","power"],outlier_threshold=4.)
            pred,_,meta=foundation_predict(family,a,b,features,{},pa,123)
        assert len(pred)==24 and np.isfinite(pred).all()
        log("SEED123_API_GPU_SMOKE_PASS",family,meta)
    else:
        manifest=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        run(manifest,sys.argv[2])
