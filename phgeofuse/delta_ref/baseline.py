"""Refit every component of the frozen complete baseline on allowed labels.

The deployed v3 homology residual scale is zero and every non-residual weight
is identical to its tuned-v1 initialization. That equality is checked before
omitting the inactive residual-gate training. The historical neural epoch is
frozen before this experiment; no excluded label selects a new epoch or scale.
"""
from __future__ import annotations

import copy
import gc
import json
import os
from dataclasses import replace
from pathlib import Path
import time

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from phgeofuse.cache import atomic_json, atomic_torch_save, sha256_file
from phgeofuse.config import load_config, path, save_resolved
from phgeofuse.io import write_manifest
from phgeofuse.retrieval import RetrievalStore, record_key, _build_retrieval_rows
from phgeofuse.robust_train import frequency_weights
from phgeofuse.dual_fusion import retrieval_sequence_anchor, DualFusion
from .data import freeze_json, atomic_npz, validate_certificate, assert_disjoint, read_predictions, stable_hash


class FullBaseline:
    def __init__(self, data, output, seed=42, device="cuda"):
        self.data, self.output, self.seed, self.device = data, Path(output), seed, device
        self.output.mkdir(parents=True, exist_ok=True)
        root = Path(data.config["_root"])
        os.environ["PATH"] = str(Path(os.sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
        self.original = path(data.config, "paths.baseline_experiment")
        self.research = path(data.config, "paths.source_experiment")
        self.base_config = load_config(self.original / "seed42/baseline.yaml")
        self.retrieval_config = load_config(self.original / "seed42/v3.yaml")
        self.retrieval_config["retrieval"].update(require_mmseqs=True, require_foldseek=True, search_threads=8)
        self.original_store = RetrievalStore.load(root / "artifacts/phgeofuse/retrieval.pt")
        trainkeys = data.keys[data.train].tolist()
        if self.original_store.payload["training_keys"] != trainkeys:
            raise ValueError("original retrieval reference keys differ from PHOPT training")
        if not np.allclose(self.original_store.payload["training_labels"].numpy(), data.labels[data.train], atol=1e-6, rtol=0):
            raise ValueError("original reference labels differ")
        self.vector_index = {k:i for i,k in enumerate(trainkeys)}
        self.fixed_epoch = self._verify_inactive_gate()
        self.protocol = {"seed": seed, "neural_fixed_epochs": self.fixed_epoch, "scheduler_horizon": 40,
                         "historical_hyperparameters_frozen_before_experiment": True,
                         "homology_gate_scale": 0., "refit_all_supervised_weights": True,
                         "dual_expert_meta_training": "within-subset grouped cross-fitting",
                         "original_selection_split": "original PHOPT validation (historical)",
                         "base_config_sha256": sha256_file(self.original / "seed42/baseline.yaml"),
                         "retrieval_config_sha256": sha256_file(self.original / "seed42/v3.yaml")}
        freeze_json(self.output / "protocol.json", self.protocol)

    def _verify_inactive_gate(self):
        runs = self.original / "seed42/runs"
        a = torch.load(runs / "phgeofuse_phopt_tuned_mse_v1_frozen_seed42/best.pt", map_location="cpu")
        b = torch.load(runs / "phgeofuse_phopt_homology_gate_v3_shrinkage_frozen_seed42/best_calibrated.pt", map_location="cpu")
        if b["homology_residual_scale"] != 0:
            raise ValueError("this baseline refitter requires the verified zero residual-scale recipe")
        for key, value in a["model_state_dict"].items():
            if key not in b["model_state_dict"] or not torch.equal(value, b["model_state_dict"][key]):
                raise ValueError(f"baseline shared weights changed: {key}")
        return int(a["epoch"]) + 1

    def status(self, event, **details):
        row = {"event": event, "pid": os.getpid(), "updated": time.time(), **details}
        atomic_json(self.output / "status.json", row)
        print(json.dumps(row), flush=True)

    def _retrieval_rows(self, fit, query, destination):
        data = self.data
        expected = {"fit_keys": data.keys[fit].tolist(), "query_keys": data.keys[query].tolist(),
                    "fit_labels": data.labels[fit].tolist(), "retrieval_config": self.retrieval_config["retrieval"]}
        certificate = Path(destination).with_suffix(".json")
        freeze_json(certificate, expected)
        if Path(destination).exists():
            payload = torch.load(destination, map_location="cpu")
            if payload["metadata"] != expected:
                raise ValueError("retrieval cache provenance differs")
            return RetrievalStore(payload)
        self.status("build_subset_retrieval", fit=len(fit), query=len(query), output=str(destination))
        training = [data.records[i] for i in fit]
        queries = [replace(data.records[i], ph_opt=float("nan"), ec="", organism="", sample_weight=1.) for i in query]
        vectors = self.original_store.payload["training_vectors"][[self.vector_index[k] for k in data.keys[fit]]].float()
        rows = _build_retrieval_rows(queries, training, vectors, torch.tensor(data.labels[fit],dtype=torch.float32), self.retrieval_config)
        if set(rows) != set(data.keys[query]):
            raise ValueError("retrieval query coverage differs")
        payload = {"rows": rows, "metadata": expected}
        atomic_torch_save(destination, payload)
        return RetrievalStore(payload)

    def base_features(self, excluded):
        data = self.data
        fit, query = data.partition(excluded)
        tag = "_".join(map(str, sorted(excluded)))
        destination = self.output / "base_features" / f"excluded_{tag}.npz"
        metadata = data.certificate(fit,query,excluded)
        freeze_json(destination.with_suffix(".json"),metadata)
        if destination.exists():
            with np.load(destination,allow_pickle=False) as z:
                if not np.array_equal(z["keys"],data.keys[query]):
                    raise ValueError("base feature keys mismatch")
                return query,{k:z[k] for k in z.files if k!='keys'}
        store = self._retrieval_rows(fit,query,destination.with_suffix('.retrieval.pt'))
        ridge=Ridge(alpha=.2,solver='cholesky').fit(data.embeddings[fit],data.labels[fit])
        payload={'retrieval':np.array([store.features(k).numpy() for k in data.keys[query]],dtype=float),
                 'sequence':ridge.predict(data.embeddings[query])}
        atomic_npz(destination,keys=data.keys[query],**payload)
        return query,payload

    def _full_training_store(self, fit, query, destination):
        data=self.data
        # Build with original training keys, so self-hits remain excluded.
        all_rows=np.r_[fit,query]
        store=self._retrieval_rows(fit,all_rows,destination)
        return store

    def _graph(self, fit, query, store, directory):
        from phgeofuse.engine import train_model, load_checkpoint, _amp_dtype
        from phgeofuse.dataset import ProteinGraphDataset, collate_graphs, move_batch
        from phgeofuse.model import PHGeoFuse
        from utils.distributed import DistributedContext
        from torch.utils.data import DataLoader
        data=self.data
        directory.mkdir(parents=True,exist_ok=True)
        cfg=copy.deepcopy(self.base_config)
        cfg['_root']=data.config['_root']
        cfg['paths'].update(manifest=str(directory/'manifest.csv'),retrieval=str(directory/'retrieval.pt'),runs=str(directory/'runs'))
        cfg['data']={'dataset_fingerprint':stable_hash(data.keys[fit].tolist()),'validation_role':'training_probe_ignored_in_fixed_epoch_refit'}
        cfg['training'].update(seed=self.seed,epochs=40,stop_after_epochs=self.fixed_epoch,
                              early_stopping_patience=41,num_workers=2,
                              run_name='delta_baseline_subset',diagnostics={'evaluate_train':False})
        ready=[data.records[i] for i in fit]
        # The engine requires a nonempty validation loader. A single training
        # probe meets that contract; its scores select NOTHING. We always use
        # last.pt after the predeclared epoch, never best.pt from this probe.
        probe=replace(ready[0],split='validation')
        records=[*ready,probe]
        payload={'schema_version':'2','dataset_fingerprint':cfg['data']['dataset_fingerprint'],
                 'rows':{k:store.rows[k] for k in data.keys[fit]},'training_keys':data.keys[fit].tolist()}
        payload['rows'][record_key(probe)]=store.rows[record_key(ready[0])]
        atomic_torch_save(directory/'retrieval.pt',payload)
        write_manifest(directory/'manifest.csv',records)
        save_resolved(cfg,directory/'config.yaml')
        context=DistributedContext(False,0,0,1,torch.device(self.device))
        # Run directory naming is the engine's public naming convention.
        run_dir=directory/'runs'/f'delta_baseline_subset_frozen_seed{self.seed}'
        checkpoint=run_dir/'last.pt'
        complete=directory/'training.complete.json'
        if not complete.exists():
            # The legacy engine does not restore the DataLoader/random state on
            # resume. Restart an interrupted fit from its seed in a fresh run
            # directory, preserving the partial attempt for provenance.
            if checkpoint.exists():
                attempt=1
                while (directory/'restart_attempts'/str(attempt)).exists():attempt+=1
                cfg['paths']['runs']=str(directory/'restart_attempts'/str(attempt))
                run_dir=Path(cfg['paths']['runs'])/f'delta_baseline_subset_frozen_seed{self.seed}'
                checkpoint=run_dir/'last.pt'
                save_resolved(cfg,directory/'config.yaml')
            self.status('train_graph_baseline',fit=len(fit),query=len(query),output=str(directory))
            train_model(records,cfg,context)
            trained=torch.load(checkpoint,map_location='cpu')
            if int(trained['epoch'])+1 != self.fixed_epoch:
                raise ValueError('baseline stopped before the fixed refit epoch')
            atomic_json(complete,{'checkpoint_sha256':sha256_file(checkpoint),'epochs':self.fixed_epoch,
                                  'checkpoint_path':str(checkpoint),
                                  'probe_metrics_are_not_validation':True,'fit_keys':data.keys[fit].tolist()})
        cert=json.loads(complete.read_text())
        checkpoint=Path(cert.get('checkpoint_path',checkpoint))
        if cert['fit_keys']!=data.keys[fit].tolist() or cert['checkpoint_sha256']!=sha256_file(checkpoint):
            raise ValueError('graph checkpoint provenance mismatch')
        graph=PHGeoFuse(cfg,context.device).to(context.device)
        load_checkpoint(checkpoint,graph);graph.eval()
        queries=[replace(data.records[i],ph_opt=float('nan'),ec='',sample_weight=1.) for i in query]
        querysplit=queries[0].split
        if len(set(r.split for r in queries))!=1:
            raise ValueError('graph query prediction expects one source split')
        dataset=ProteinGraphDataset(queries,querysplit,store,'frozen')
        loader=DataLoader(dataset,batch_size=2,num_workers=2,collate_fn=collate_graphs,pin_memory=True)
        predictions={}
        with torch.inference_mode():
            for batch in loader:
                batch=move_batch(batch,context.device)
                with torch.autocast(context.device.type,dtype=_amp_dtype(cfg,context.device),enabled=context.device.type=='cuda'):
                    result=graph(batch)['mean'].float().cpu().numpy()
                predictions.update(zip(batch['keys'],result.tolist()))
        if set(predictions)!=set(data.keys[query]):
            raise ValueError('graph prediction coverage differs')
        out=np.array([predictions[k] for k in data.keys[query]])
        del graph
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        return out

    def predict_excluded(self, excluded):
        data=self.data
        fit,query=data.partition(excluded)
        tag='_'.join(map(str,sorted(excluded)))
        directory=self.output/f'excluded_{tag}'
        directory.mkdir(parents=True,exist_ok=True)
        certificate={**data.certificate(fit,query,excluded),'baseline_protocol':self.protocol}
        freeze_json(directory/'fit.json',certificate)
        prediction_file=directory/'predictions.npz'
        if prediction_file.exists():
            with np.load(prediction_file,allow_pickle=False) as z:
                if not np.array_equal(z['keys'],data.keys[query]):raise ValueError('baseline query keys differ')
                return query,{k:z[k] for k in z.files if k!='keys'}
        started=time.monotonic()
        store=self._full_training_store(fit,query,directory/'all.retrieval.pt')
        raw=self._graph(fit,query,store,directory/'graph')
        y=data.labels[fit];chem=data.x[:,-25:]
        rtrain=np.array([store.features(k).numpy() for k in data.keys[fit]],dtype=float)
        rquery=np.array([store.features(k).numpy() for k in data.keys[query]],dtype=float)
        lowtrain=np.array([store.features(k,view='low_homology').numpy() for k in data.keys[fit]],dtype=float)
        ridge=Ridge(alpha=.1,solver='cholesky').fit(data.embeddings[fit,2560:],y,sample_weight=frequency_weights(y,.25))
        seq=ridge.predict(data.embeddings[query,2560:])
        def anchor(r):
            a=r[:,7:9]
            return np.where(a.sum(1)>0,(r[:,:2]*a).sum(1)/np.maximum(a.sum(1),1),y.mean())
        meta=np.concatenate([np.column_stack([rtrain,chem[fit]]),np.column_stack([lowtrain,chem[fit]])])
        w=frequency_weights(y,.5)
        residual=HistGradientBoostingRegressor(max_leaf_nodes=7,max_iter=150,learning_rate=.05,
            min_samples_leaf=60,l2_regularization=20,early_stopping=False,random_state=self.seed)
        residual.fit(meta,np.tile(y,2)-np.r_[anchor(rtrain),anchor(lowtrain)],sample_weight=np.r_[w,w*.25])
        robust=.5*raw+.25*seq+.25*(anchor(rquery)+residual.predict(np.column_stack([rquery,chem[query]])))
        inner_r=np.full((len(data.keys),15),np.nan);inner_s=np.full(len(data.keys),np.nan)
        inner_audit=[]
        for inner in sorted(set(data.folds[fit])):
            excluded_inner=sorted(set(excluded)|{int(inner)})
            q,features=self.base_features(excluded_inner)
            keep=data.folds[q]==inner
            inner_r[q[keep]],inner_s[q[keep]]=features['retrieval'][keep],features['sequence'][keep]
            inner_audit.append({'excluded_folds':excluded_inner,'used_query_keys':data.keys[q[keep]].tolist()})
        if not np.isfinite(inner_r[fit]).all() or not np.isnan(inner_s[query]).all():
            raise ValueError('full-baseline inner cross-fit isolation failed')
        dual_sequence=Ridge(alpha=.2,solver='cholesky').fit(data.embeddings[fit],y)
        ds=dual_sequence.predict(data.embeddings[query])
        ta=retrieval_sequence_anchor(inner_r[fit],inner_s[fit])
        qa=retrieval_sequence_anchor(rquery,ds)
        dual_residual=HistGradientBoostingRegressor(max_leaf_nodes=7,max_iter=50,learning_rate=.05,
            min_samples_leaf=80,l2_regularization=30,early_stopping=False,random_state=self.seed)
        dual_residual.fit(np.column_stack([inner_r[fit],inner_s[fit],chem[fit]]),y-ta,
                          sample_weight=frequency_weights(y,.25))
        dual=qa+dual_residual.predict(np.column_stack([rquery,ds,chem[query]]))
        prediction=.5*robust+.5*dual
        for name,model in [('robust_sequence',ridge),('robust_residual',residual),('dual_sequence',dual_sequence),('dual_residual',dual_residual)]:
            joblib.dump(model,directory/f'{name}.joblib')
        atomic_json(directory/'inner_audit.json',inner_audit)
        result={'prediction':prediction,'graph':raw,'robust':robust,'dual':dual,'ridge':ds,
                'retrieval':rquery,'low_homology':~((rquery[:,4]>=.2)&(rquery[:,9]>=.8)&(rquery[:,10]>=.8))}
        if not all(np.isfinite(v).all() for v in result.values()):raise ValueError('nonfinite baseline result')
        atomic_npz(prediction_file,keys=data.keys[query],**result)
        self.status('complete_baseline_subset',excluded_folds=list(excluded),seconds=time.monotonic()-started)
        return query,result

    def frozen_validation(self, seed):
        data=self.data;query=data.validation
        # The historical CSV is the v3 PHGeoFuse component, not the complete
        # dual blend. Recompute the latter from its checksummed fitted bundle.
        raw=read_predictions(self.research/f'baseline_seed{seed}_validation.csv',data.keys[query]) if seed!=42 else read_predictions(self.research/'baseline_validation.csv',data.keys[query])
        bundle=DualFusion(self.research/'dual_candidate_float64')
        parts=[]
        for name in ('esm1v','esm2'):
            with np.load(path(data.config,'paths.'+name),allow_pickle=False) as z:
                order={str(k):i for i,k in enumerate(z['keys'])};idx=[order[k] for k in data.keys[query]]
                parts.extend([z['mean'][idx],z['std'][idx]])
        r=np.array([self.original_store.features(k).numpy() for k in data.keys[query]],dtype=float)
        result=bundle.predict(raw,*parts,r,[data.records[i].sequence for i in query])
        return result['prediction']
