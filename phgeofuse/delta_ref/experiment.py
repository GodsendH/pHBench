"""Resumable five-outer/four-inner evaluation of the complete prediction path."""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
from pathlib import Path
import time
import numpy as np

from phgeofuse.cache import atomic_json, sha256_file
from phgeofuse.config import path
from .baseline import FullBaseline
from .data import DevelopmentData, atomic_npz, freeze_json, stable_hash, write_predictions
from .training import fit_subset, recipes, save_bundle, load_bundle
from .metrics import metrics, select_strength, selection_score, acceptance, guardrails, seed_summary, paired_family_bootstrap
from .model import ReferencePredictor, blend_predictions, select_panel


@contextlib.contextmanager
def experiment_lock(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with (output/'runner.lock').open('a') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('another runner already owns this experiment') from exc
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)


def status(output,event,**details):
    row={'event':event,'pid':os.getpid(),'updated':time.time(),**details}
    atomic_json(Path(output)/'status.json',row)
    print(json.dumps(row),flush=True)


def audit(data,output):
    code=Path(data.config['_root'])
    sources=sorted((code/'phgeofuse').rglob('*.py'))+sorted((code/'utils').glob('*.py'))
    from .inference import baseline_dependencies
    assets=sorted({Path(filename) for record in data.records for filename in (record.graph_path,record.embedding_path)})
    asset_hashes={str(p):sha256_file(p) for p in assets}
    freeze_json(Path(output)/'feature_assets.json',asset_hashes)
    research=path(data.config,'paths.source_experiment')
    protocol={'inputs':data.provenance,'configuration':{k:v for k,v in data.config.items() if not k.startswith('_')},
        'source_root':str(code),'source_hashes':{str(p.relative_to(code)):sha256_file(p) for p in sources},
        'baseline_dependencies':{str(seed):baseline_dependencies(data.config,seed) for seed in data.config['protocol']['seeds']},
        'feature_assets_sha256':sha256_file(Path(output)/'feature_assets.json'),
        'baseline_test_files':{str(research/f'dual_test/seed{s}.csv'):sha256_file(research/f'dual_test/seed{s}.csv') for s in data.config['protocol']['seeds']},
        'development_only':True,'test_used_for_selection':False,
        'tail_definition':{'acid':'y <= 4','alkaline':'y >= 10','core':'4 < y < 10'},
        'seeds':data.config['protocol']['seeds'],'outer_folds':5,'inner_folds':4,
        'baseline_role':'complete dual blend; every supervised component refit within exclusions',
        'prior_hyperparameters':'historical baseline recipe held fixed; new recipe and strength selected inside each outer fold',
        'counts':{split:{'total':int((data.splits==split).sum()),
            'acid':int(((data.splits==split)&(data.labels<=4)).sum()),
            'alkaline':int(((data.splits==split)&(data.labels>=10)).sum())} for split in ('train','validation')}}
    freeze_json(Path(output)/'protocol.json',protocol)
    return protocol


def _aligned(query,values,wanted):
    mapping={int(k):i for i,k in enumerate(query)}
    if not set(map(int,wanted))<=set(mapping):raise ValueError('prediction subset missing rows')
    return values[[mapping[int(k)] for k in wanted]]


def nested(data,output,candidates=None,seed=42,fit_function=fit_subset,baseline=None,outer_recipes=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    candidates=candidates or recipes(data.config)
    provider=baseline or FullBaseline(data,output.parent/'baseline'/f'seed{seed}',seed,data.config['training']['device'])
    n=len(data.keys);base=np.full(n,np.nan);selected_prediction=np.full(n,np.nan)
    retrieval_low=np.zeros(n,bool)
    all_predictions={r['name']:{'transfer':np.full(n,np.nan),'dispersion':np.full(n,np.nan),'valid':np.zeros(n,bool),
                              'inner_selected':np.full(n,np.nan)} for r in candidates}
    outer_rows=[]
    for outer in range(5):
        status(output,'outer_start',outer=outer)
        train,heldout=data.partition([outer])
        q,b=provider.predict_excluded([outer])
        base[heldout]=_aligned(q,b['prediction'],heldout)
        retrieval_low[heldout]=_aligned(q,b['low_homology'],heldout)
        inner_base=np.full(n,np.nan)
        for inner in sorted(set(data.folds[train])):
            ii=train[data.folds[train]==inner]
            q,b=provider.predict_excluded([outer,int(inner)])
            inner_base[ii]=_aligned(q,b['prediction'],ii)
        if not np.isfinite(inner_base[train]).all() or np.isfinite(inner_base[heldout]).any():
            raise ValueError('inner baseline exclusion failure')
        choices=[]
        for recipe in (outer_recipes[outer] if outer_recipes is not None else candidates):
            inner_transfer=np.full(n,np.nan);inner_disp=np.full(n,np.nan);inner_valid=np.zeros(n,bool)
            epoch_choices=[]
            for inner in sorted(set(data.folds[train])):
                fit=train[data.folds[train]!=inner];validation=train[data.folds[train]==inner]
                directory=output/f'outer{outer}'/recipe['name']/f'inner{inner}'
                predictor,report=fit_function(data,fit,validation,inner_base[validation],recipe,directory,
                    seed=seed,excluded=[outer,int(inner)])
                t,d,v=predictor.transfer(data.x[validation],data.keys[validation])
                inner_transfer[validation],inner_disp[validation],inner_valid[validation]=t,d,v
                epoch_choices.append(report['best_epoch'])
                del predictor
            if not np.isfinite(inner_transfer[train]).all():raise ValueError('incomplete inner predictions')
            choice,grid=select_strength(data.labels[train],inner_base[train],inner_transfer[train],inner_disp[train],inner_valid[train],data.groups[train])
            refit_epochs=max(1,int(np.median(epoch_choices)))
            atomic_npz(output/f'outer{outer}'/recipe['name']/'inner_predictions.npz',
                       keys=data.keys[train],baseline=inner_base[train],transfer=inner_transfer[train],
                       dispersion=inner_disp[train],valid=inner_valid[train])
            atomic_json(output/f'outer{outer}'/recipe['name']/'selection.json',{'choice':choice,'grid':grid,'epochs':refit_epochs})
            refit=output/f'outer{outer}'/recipe['name']/'refit'
            predictor,report=fit_function(data,train,heldout,None,recipe,refit,seed=seed,fixed_epochs=refit_epochs,excluded=[outer])
            t,d,v=predictor.transfer(data.x[heldout],data.keys[heldout])
            for key,value in zip(('transfer','dispersion','valid'),(t,d,v)):
                all_predictions[recipe['name']][key][heldout]=value
            all_predictions[recipe['name']]['inner_selected'][heldout]=blend_predictions(
                base[heldout],t,d,choice['strength'],v)['prediction']
            choice={**choice,'recipe':recipe,'epochs':refit_epochs,'refit':str(refit)}
            choices.append(choice)
            del predictor
        winner=min(choices,key=lambda c:(tuple(c['rank']),c['recipe']['name']))
        r=all_predictions[winner['recipe']['name']]
        pred=blend_predictions(base[heldout],r['transfer'][heldout],r['dispersion'][heldout],winner['strength'],r['valid'][heldout])['prediction']
        selected_prediction[heldout]=pred
        outer_rows.append({'outer':outer,'winner':winner,'choices':choices,
                          'metrics':metrics(data.labels[heldout],pred,data.groups[heldout]),
                          'baseline':metrics(data.labels[heldout],base[heldout],data.groups[heldout])})
        atomic_json(output/'outer_results.json',outer_rows)
        status(output,'outer_complete',outer=outer,winner=winner['recipe']['name'],strength=winner['strength'])
    train=data.train
    final_choices=[]
    saved={'keys':data.keys[train],'y':data.labels[train],'groups':data.groups[train],'fold':data.folds[train],
           'baseline':base[train],'nested_selected':selected_prediction[train],'low_homology':retrieval_low[train]}
    for recipe in candidates:
        r=all_predictions[recipe['name']]
        choice,grid=select_strength(data.labels[train],base[train],r['transfer'][train],r['dispersion'][train],r['valid'][train],data.groups[train])
        epochs=[next(c['epochs'] for c in row['choices'] if c['recipe']['name']==recipe['name']) for row in outer_rows]
        final_choices.append({**choice,'recipe':recipe,'epochs':max(1,int(np.median(epochs))),
                              'nested_metrics':metrics(data.labels[train],r['inner_selected'][train],data.groups[train])})
        for key,value in r.items():saved[recipe['name']+'__'+key]=value[train]
    winner=min(final_choices,key=lambda c:(tuple(c['rank']),c['recipe']['name']))
    result={'baseline':metrics(data.labels[train],base[train],data.groups[train],retrieval_low[train]),
            'nested_selected':metrics(data.labels[train],selected_prediction[train],data.groups[train],retrieval_low[train]),
            'final_selection':winner,'final_choices':final_choices,
            'selection_warning':'final_selection metrics are development scores; nested_selected evaluates inner selection',
            'test_access':False}
    result['acceptance']=acceptance(result['nested_selected'],result['baseline'])
    result['frozen_direction_pass']=not guardrails(result['nested_selected'],result['baseline']) and all(
        result['nested_selected'][g]['rmse']<result['baseline'][g]['rmse'] for g in ('acid','alkaline'))
    result['recipe_direction_pass']={c['recipe']['name']:not guardrails(c['nested_metrics'],result['baseline']) and all(
        c['nested_metrics'][g]['rmse']<result['baseline'][g]['rmse'] for g in ('acid','alkaline')) for c in final_choices}
    result['lora_triggered_by_protocol']=not any(result['recipe_direction_pass'].values()) and not result['frozen_direction_pass']
    result['family_bootstrap']=paired_family_bootstrap(data.labels[train],selected_prediction[train][None,:],
        {'baseline':base[train][None,:]},data.groups[train],data.config['protocol'].get('bootstrap_draws',10000))
    write_predictions(output/'predictions.csv',data.keys[train],{'label':data.labels[train],
        'group':data.groups[train],'outer_fold':data.folds[train],'low_homology':retrieval_low[train],
        'baseline_prediction':base[train],'prediction':selected_prediction[train],
        'error':selected_prediction[train]-data.labels[train]})
    atomic_npz(output/'predictions.npz',**saved)
    atomic_json(output/'results.json',result)
    status(output,'nested_complete',acceptance=result['acceptance'],direction_pass=result['frozen_direction_pass'])
    return result


def evaluate_ablations(data,output,main,seed=42):
    """Train direct/additive controls with the same nested selection machinery.

    Panel-distribution, agreement, anchor-removal and output-expansion controls
    use only inner-held-out labels to choose their scalar hyperparameters.
    """
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    shared_baseline=FullBaseline(data,output.parent/'baseline'/f'seed{seed}',seed,data.config['training']['device'])
    fitted={}
    for kind in ('absolute','additive'):
        controls=[{**recipe,'name':kind+'_'+recipe['name'],'kind':kind} for recipe in recipes(data.config)]
        fitted[kind]=nested(data,output/kind,controls,seed,baseline=shared_baseline)
    summaries={'learned_controls':fitted,'fixed_controls':{}}
    source=output.parent/'frozen'
    allp={name:np.full(len(data.keys),np.nan) for name in ('natural_panel','no_agreement','no_baseline','affine_expansion')}
    base=np.full(len(data.keys),np.nan)
    for outer in range(5):
        train,query=data.partition([outer])
        selected=json.loads((source/'outer_results.json').read_text())[outer]['winner']
        rp=selected['recipe'];name=rp['name']
        with np.load(source/f'outer{outer}'/name/'inner_predictions.npz') as z:
            if not np.array_equal(z['keys'],data.keys[train]):raise ValueError('ablation inner keys differ')
            inner={k:z[k] for k in z.files}
        with np.load(source/'predictions.npz') as z:
            mapping={str(k):i for i,k in enumerate(z['keys'])};idx=[mapping[k] for k in data.keys[query]]
            b=z['baseline'][idx];base[query]=b
        predictor,_=load_bundle(source/f'outer{outer}'/name/'refit',data.config['training']['device'])
        t,d,v=predictor.transfer(data.x[query],data.keys[query])
        chosen,_=select_strength(data.labels[train],inner['baseline'],inner['transfer'],inner['dispersion'],inner['valid'],data.groups[train],consistency=False)
        allp['no_agreement'][query]=blend_predictions(b,t,d,chosen['strength'],v,consistency=False)['prediction']
        allp['no_baseline'][query]=np.where(v,np.clip(t,0,14),b)
        # Fixed affine-expansion grid about pH 7. It is evaluated as a control,
        # never silently used as the proposed model's correction.
        base_metrics=metrics(data.labels[train],inner['baseline'],data.groups[train])
        scales=[1.,1.05,1.1,1.2,1.3,1.5]
        scale=min(scales,key=lambda s:selection_score(metrics(data.labels[train],np.clip(7+s*(inner['baseline']-7),0,14),data.groups[train]),base_metrics,s-1))
        allp['affine_expansion'][query]=np.clip(7+scale*(b-7),0,14)
        natural_t=np.full(len(train),np.nan);natural_d=np.full(len(train),np.nan);natural_v=np.zeros(len(train),bool)
        for j in sorted(set(data.folds[train])):
            fit=train[data.folds[train]!=j];val=train[data.folds[train]==j]
            p,_=load_bundle(source/f'outer{outer}'/name/f'inner{j}',data.config['training']['device'])
            refs,w=select_panel(data.embeddings[fit],data.labels[fit],data.groups[fit],data.keys[fit],data.config['model']['references_per_bin'],balanced=False)
            refs=fit[refs]
            natural=ReferencePredictor(p.network,p.scaler,data.x[refs],data.labels[refs],data.groups[refs],data.keys[refs],w,p.device,balanced=False)
            index=np.flatnonzero(data.folds[train]==j)
            natural_t[index],natural_d[index],natural_v[index]=natural.transfer(data.x[val],data.keys[val])
        chosen,_=select_strength(data.labels[train],inner['baseline'],natural_t,natural_d,natural_v,data.groups[train])
        refs,w=select_panel(data.embeddings[train],data.labels[train],data.groups[train],data.keys[train],data.config['model']['references_per_bin'],balanced=False)
        refs=train[refs]
        natural=ReferencePredictor(predictor.network,predictor.scaler,data.x[refs],data.labels[refs],data.groups[refs],data.keys[refs],w,predictor.device,balanced=False)
        allp['natural_panel'][query]=natural.predict(data.x[query],b,chosen['strength'],data.keys[query])['prediction']
    for name,p in allp.items():summaries['fixed_controls'][name]=metrics(data.labels[data.train],p[data.train],data.groups[data.train])
    atomic_npz(output/'fixed_controls.npz',keys=data.keys[data.train],y=data.labels[data.train],groups=data.groups[data.train],
               **{k:v[data.train] for k,v in allp.items()})
    atomic_json(output/'results.json',summaries)
    return summaries


def run(config_path,output_override=None,baseline_only=False):
    data=DevelopmentData.load(config_path)
    output=Path(output_override) if output_override else path(data.config,'paths.output')
    with experiment_lock(output):
        status(output,'protocol_audit_started')
        audit(data,output)
        try:
            seed=data.config['protocol']['development_seed']
            if baseline_only:
                provider=FullBaseline(data,output/'baseline'/f'seed{seed}',seed,data.config['training']['device'])
                import itertools
                for size in (2,1):
                    for excluded in itertools.combinations(range(5),size):provider.predict_excluded(excluded)
                status(output,'baseline_cache_complete')
                return
            status(output,'frozen_nested_started')
            main=nested(data,output/'frozen',seed=seed)
            from .comparisons import nested_classical
            status(output,'classical_comparisons_started')
            nested_classical(data,output/'comparisons',seed)
            status(output,'ablations_started')
            evaluate_ablations(data,output/'ablations',main,seed)
            if main['lora_triggered_by_protocol']:
                from .lora import nested_lora
                status(output,'lora_triggered',reason='no frozen recipe simultaneously improved both tails within guardrails')
                main=nested_lora(data,output/'lora',main,seed)
            if not main['acceptance']['passed']:
                status(output,'complete_not_accepted',acceptance=main['acceptance'],default_model_replaced=False)
                write_report(output)
                return
            status(output,'five_seed_refit_started')
            finalize_development(data,output,main)
            release=json.loads((output/'frozen_release.json').read_text())
            if release['ready_for_followup_test']:
                from .evaluation import followup_test
                followup_test(data.config['_config_path'],output)
            write_report(output)
        except BaseException as exc:
            status(output,'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                   exception=type(exc).__name__,reason=str(exc),default_model_replaced=False)
            raise


def finalize_development(data,output,main):
    from .inference import attach_baseline
    from .comparisons import finalize_controls,frozen_field_manifests
    finalize_controls(data,output)
    selected=main['final_selection'];recipe=selected['recipe'];strength=selected['strength']
    predictor_rows=[];baseline_rows=[];paths=[]
    for seed in data.config['protocol']['seeds']:
        directory=Path(output)/'final'/f'seed{seed}'
        provider=FullBaseline(data,Path(output)/'baseline'/f'seed{seed}',seed,data.config['training']['device'])
        b=provider.frozen_validation(seed)
        fit_function=fit_subset
        if recipe.get('representation')=='lora':
            from .lora import fit_lora_subset
            fit_function=fit_lora_subset
        predictor,report=fit_function(data,data.train,data.validation,None,recipe,directory,
                                     seed=seed,fixed_epochs=selected['epochs'],excluded=[])
        pred=predictor.predict(data.x[data.validation],b,strength,data.keys[data.validation])
        predictor_rows.append(pred['prediction']);baseline_rows.append(b)
        write_predictions(directory/'validation_predictions.csv',data.keys[data.validation],{'label':data.labels[data.validation],**pred})
        cfg=json.loads((directory/'model.json').read_text());cfg['strength']=strength
        atomic_json(directory/'model.json',cfg)
        attach_baseline(directory,data.config,seed)
        paths.append({'seed':seed,'path':str(directory),'model_sha256':sha256_file(directory/'model.json')})
    y=data.labels[data.validation]
    candidate=seed_summary(y,np.asarray(predictor_rows));baseline=seed_summary(y,np.asarray(baseline_rows))
    guard=guardrails(candidate['mean'],baseline['mean'])
    tail_direction=all(candidate['mean'][g]['rmse']<baseline['mean'][g]['rmse'] for g in ('acid','alkaline'))
    research=path(data.config,'paths.source_experiment')
    baseline_test_files=json.loads((Path(output)/'protocol.json').read_text())['baseline_test_files']
    if any(sha256_file(filename)!=digest for filename,digest in baseline_test_files.items()):
        raise ValueError('original baseline predictions changed since protocol freeze')
    frozen={'selected':selected,'models':paths,'validation':candidate,'baseline':baseline,
            'internal_acceptance':main['acceptance'],'protocol_sha256':sha256_file(Path(output)/'protocol.json'),
            'controls_release_sha256':sha256_file(Path(output)/'comparisons/frozen_release.json'),
            'field_manifests':frozen_field_manifests(output),
            'baseline_test_files':baseline_test_files,
            'validation_guard_failures':guard,'validation_tail_direction_pass':tail_direction,
            'test_access':False,'ready_for_followup_test':not guard and tail_direction,
            'field_leadership_established':False,'default_model_replaced':False}
    freeze_json(Path(output)/'frozen_release.json',frozen)
    status(output,'candidate_frozen' if frozen['ready_for_followup_test'] else 'validation_veto',
           validation_guard_failures=guard,default_model_replaced=False)


def write_report(output):
    output=Path(output)
    state=json.loads((output/'status.json').read_text())
    lines=['# DeltaRef-pH experiment status','',f"Status: `{state['event']}`",'',
           'The default predictor has not been replaced. No result in this report establishes independent external generalization.','']
    for stage in ('frozen','lora'):
        source=output/stage/'results.json'
        if not source.exists():continue
        result=json.loads(source.read_text())
        lines.extend([f'## {stage}', '', '| Model | All RMSE | Acid RMSE | Alkaline RMSE | All MAE |', '|---|---:|---:|---:|---:|'])
        for name in ('baseline','nested_selected'):
            m=result[name]
            lines.append(f"| {name} | {m['all']['rmse']:.6f} | {m['acid']['rmse']:.6f} | {m['alkaline']['rmse']:.6f} | {m['all']['mae']:.6f} |")
        lines.extend(['', 'Acceptance: '+json.dumps(result['acceptance']), ''])
    followup=output/'followup_test/results.json'
    if followup.exists():
        result=json.loads(followup.read_text())
        lines.extend(['## Follow-up test (previously inspected)', '',
            'Acceptance: '+json.dumps(result['acceptance']),
            'Missing controlled field comparisons: '+', '.join(result.get('missing_field_comparisons',[])),
            'Field leadership established: '+str(result['field_leadership_established']),''])
    costs=[]
    for p in output.rglob('complete.json'):
        if 'smoke' in p.parts or 'baseline' in p.parts:continue
        row=json.loads(p.read_text())
        if 'wall_seconds' in row:costs.append({'path':str(p.parent),**row})
    atomic_json(output/'run_costs.json',costs)
    lines.extend(['Completed delta/control fits: '+str(len(costs)),
                  'Summed fit-loop seconds (excludes baseline preparation): '+str(sum(r['wall_seconds'] for r in costs)),
                  '', 'External field reproductions are recorded separately; paper-reported scores are never inserted into the ranking.'])
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
