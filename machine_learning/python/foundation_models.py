"""Fixed-configuration TabPFN and TabICL prediction."""
from __future__ import annotations
import os
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT=Path.cwd()
spec=importlib.util.spec_from_file_location("existing_tab_backend",ROOT/"machine_learning/python/tabular_training.py")
backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend)
log,save,load_block,release=backend.log,backend.save,backend.load_block,backend.release

def predict_fixed(family,a,b,features,cats,seed=123):
    assert torch.cuda.is_available() and "label" not in b
    torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    x,q=a[features].copy(),b[features].copy();y=a.label.to_numpy(dtype=np.int64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0=time.monotonic()
        if family=="TabPFN":
            from tabpfn import TabPFNClassifier
            model=TabPFNClassifier(model_path=str(backend.CHECKPOINTS[family]),device="cuda",
                categorical_features_indices=[i for i,v in enumerate(features) if v in cats],
                random_state=seed,inference_precision=torch.float32,
                fit_mode="fit_with_cache",keep_cache_on_device=False,
                memory_saving_mode=True,show_progress_bar=False,tuning_config=None)
        else:
            from tabicl import TabICLClassifier
            for v,levels in cats.items():
                x[v]=pd.Categorical(x[v],categories=levels);q[v]=pd.Categorical(q[v],categories=levels)
            model=TabICLClassifier(model_path=str(backend.CHECKPOINTS[family]),allow_auto_download=False,
                device="cuda",random_state=seed,use_amp=False,batch_size=1,kv_cache=True,
                n_jobs=1,offload_mode="auto",verbose=False)
        constructor=time.monotonic()-t0;t0=time.monotonic()
        model.fit(x,y);torch.cuda.synchronize();fit_time=time.monotonic()-t0
        positive=int(np.flatnonzero(np.asarray(model.classes_)==1)[0])
        batch=256;t0=time.monotonic();calls=0
        while True:
            try:
                parts=[]
                for j in range(0,len(q),batch):
                    calls+=1;parts.append(model.predict_proba(q.iloc[j:j+batch])[:,positive])
                prob=np.concatenate(parts);break
            except torch.cuda.OutOfMemoryError:
                if batch<=16:raise
                batch//=2;torch.cuda.empty_cache()
                log(family,"QUERY_BATCH",batch,"FULL_CONTEXT_AND_PRECISION_UNCHANGED")
        torch.cuda.synchronize();prediction=time.monotonic()-t0
        assert len(prob)==len(q) and np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
        member_count=getattr(model,"n_estimators_",getattr(model,"n_estimators","auto"))
        if member_count=="auto":
            configs=getattr(model,"ensemble_configs_",None)
            if configs is not None:member_count=len(configs)
        meta=dict(family=family,class_name=type(model).__name__,scientific_configuration="Installed default; no Optuna or fine-tuning",
            members=member_count,constructor_seconds=constructor,fit_seconds=fit_time,predict_seconds=prediction,
            seconds=time.monotonic()-start,fit_calls=1,predict_proba_calls=calls,query_batch=batch,
            train_n=len(x),query_n=len(q),device=torch.cuda.get_device_name(0),precision="float32",
            peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
            internal_fit_components="Checkpoint loading and preprocessing are included in fit, not separately instrumented",
            checkpoint=str(backend.CHECKPOINTS[family]),warnings=sorted({str(w.message) for w in caught}))
    del model,x,q,y;release();return prob,meta

def run(manifest,family,benchmark=False):
    assert family in ("TabPFN","TabICL") and int(manifest["seed"])==123
    prefix=manifest["prefix"];blocks=manifest["blocks"];full=blocks["Full"]
    features,cats=full["features"],full["categorical"]
    if isinstance(cats,list):cats={}
    records=[];fold_results=[];oof=np.full(sum(len(pd.read_csv(blocks[f"Fold{k}"]["query_labels"])) for k in range(1,6)),np.nan)
    for k in range(1,6):
        job=blocks[f"Fold{k}"];preds=[]
        for m in range(5):
            ck=Path(prefix+f"_{family}_FIXED_F{k}_m{m+1:02d}.json")
            if ck.exists():rec=json.loads(ck.read_text(encoding="utf-8"))
            else:
                a,b,y=load_block(job,m);prob,meta=predict_fixed(family,a,b,features,cats,123)
                rec=dict(probability=prob.tolist(),fold=k,imputation=m+1,**meta);save(ck,rec)
            records.append({v:z for v,z in rec.items() if v!="probability"});preds.append(rec["probability"])
            log(family,"FIXED_FOLD",k,"MI",m+1,"SECONDS",round(rec["seconds"],1))
            if benchmark:
                log(family,"FIXED_REAL_DATA_BENCHMARK",records[-1]);return
        probability=np.mean(preds,axis=0);labels=pd.read_csv(job["query_labels"]).label.to_numpy(dtype=np.int64)
        ix=np.asarray(job["query_rows"],dtype=int)-1;oof[ix]=probability
        fold_results.append(dict(AUC=float(roc_auc_score(labels,probability)),Brier=float(brier_score_loss(labels,probability))))
    assert np.isfinite(oof).all()
    auc=float(np.mean([z["AUC"] for z in fold_results]))
    record=dict(parameters=dict(configuration="Fixed installed defaults",seed=123),cv_auc=auc,oof=oof.tolist(),
        features=features,max_trials=0,epochs=None,fold_results=fold_results,execution=records,
        selection="No classifier configuration search; common training-side patient five-fold comparison over five MI",
        trial_table=[dict(Trial=0,Status="Fixed default; no search",Score=auc,Parameters="Installed default scientific configuration")])
    save(prefix+f"_{family}_TUNING.json",record)
    qp={};qp_path=Path(prefix+f"_{family}_Full.csv")
    for m in range(30):
        ck=Path(prefix+f"_{family}_FIXED_FULL_m{m+1:02d}.json")
        if ck.exists():rec=json.loads(ck.read_text(encoding="utf-8"))
        else:
            a,b,y=load_block(full,m);prob,meta=predict_fixed(family,a,b,features,cats,123)
            rec=dict(probability=prob.tolist(),imputation=m+1,**meta);save(ck,rec)
        qp[f"m{m+1}"]=rec["probability"];pd.DataFrame(qp).to_csv(qp_path,index=False,encoding="utf-8")
        log(family,"FIXED_FINAL_MI",m+1,"OF30","SECONDS",round(rec["seconds"],1))
    save(prefix+f"_{family}_FINAL.json",dict(family=family,final_imputations=30,seed=123,features=features,
        scientific_configuration="Fixed defaults",no_training_self_predictions=True,training_performance="Use conditional OOF, not in-context self-predictions",
        versions={p:importlib.metadata.version(p) for p in ("tabpfn","tabicl","torch","scikit-learn")}))

if __name__=="__main__":
    manifest=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    run(manifest,sys.argv[2],len(sys.argv)>3 and sys.argv[3]=="benchmark")
