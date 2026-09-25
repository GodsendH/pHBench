"""python -m phgeofuse.delta_ref --help"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description='DeltaRef-pH: experimental reference-difference learning')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('audit','run','baseline','train','evaluate','status'):
        p=sub.add_parser(name)
        p.add_argument('--config',default='configs/delta_ref_phopt.yaml')
        p.add_argument('--output')
        if name=='train':
            p.add_argument('--fit-folds',type=int,nargs='+',required=True)
            p.add_argument('--validation-fold',type=int,required=True)
            p.add_argument('--width',type=int,choices=[64,128],default=64)
            p.add_argument('--power',type=float,choices=[0.,.5],default=0.)
            p.add_argument('--seed',type=int,default=42)
            p.add_argument('--fixed-epochs',type=int)
    p=sub.add_parser('predict')
    p.add_argument('--bundle',required=True)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--fasta')
    source.add_argument('--features',help='Label-free NPZ containing keys and 5145-column features')
    p.add_argument('--baseline-predictions',help='Complete baseline predictions, keyed CSV; required with --features')
    p.add_argument('--config',default='configs/delta_ref_phopt.yaml')
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--online',action='store_true',help='Allow the existing structure preparation workflow to download structures')
    p=sub.add_parser('compare')
    p.add_argument('--candidate',nargs=5,required=True,help='Five keyed CSVs in seed order 0,1,2,3,42')
    p.add_argument('--baseline',nargs=5,required=True)
    p.add_argument('--families',required=True,help='CSV containing key and group; sequence-only clustering')
    p.add_argument('--output',required=True)
    p.add_argument('--draws',type=int,default=10000)
    p=sub.add_parser('register-comparison')
    p.add_argument('--manifest',required=True)
    p.add_argument('--config',default='configs/delta_ref_phopt.yaml')
    p.add_argument('--output')
    args=parser.parse_args()
    if args.command=='predict':
        from .inference import predict_fasta,predict_cached
        if args.fasta:
            predict_fasta(args.fasta,args.bundle,args.config,args.output,args.device,args.online)
        else:
            if not args.baseline_predictions:parser.error('--features requires --baseline-predictions')
            predict_cached(args.bundle,args.features,args.baseline_predictions,args.output,args.device)
        return
    if args.command=='compare':
        from .evaluation import compare_csvs
        result=compare_csvs(args.candidate,{'baseline':args.baseline},args.families,args.draws)
        from phgeofuse.cache import atomic_json
        atomic_json(args.output,result);print(json.dumps(result['acceptance'],indent=2));return
    from phgeofuse.config import load_config,path
    config=load_config(args.config)
    output=Path(args.output) if args.output else path(config,'paths.output')
    if args.command=='register-comparison':
        from .comparisons import import_field_manifest
        if (output/'frozen_release.json').exists():parser.error('register comparators before the candidate/test release is frozen')
        print(import_field_manifest(args.manifest,output,args.config));return
    if args.command=='status':
        for p in (output/'status.json',output/'baseline/seed42/status.json'):
            if p.exists():print(p.read_text())
        return
    if args.command in ('run','baseline'):
        from .experiment import run
        run(args.config,args.output,baseline_only=args.command=='baseline');return
    if args.command=='evaluate':
        from .evaluation import followup_test
        followup_test(args.config,output);return
    from .data import DevelopmentData
    data=DevelopmentData.load(args.config)
    if args.command=='audit':
        from .experiment import audit
        print(json.dumps(audit(data,output),indent=2));return
    if args.command=='train':
        if not args.output:parser.error('train requires a separate --output directory')
        from .baseline import FullBaseline
        from .training import fit_subset
        if args.validation_fold in args.fit_folds or not set(args.fit_folds)<set(range(5)):
            parser.error('training folds must be a proper subset excluding the validation fold')
        excluded=sorted(set(range(5))-set(args.fit_folds))
        fit=np.flatnonzero(np.isin(data.folds,args.fit_folds))
        validation=np.flatnonzero(data.folds==args.validation_fold)
        provider=FullBaseline(data,output.parent/'baseline'/f'seed{args.seed}',args.seed,config['training']['device'])
        baseline=None
        if args.fixed_epochs is None:
            q,b=provider.predict_excluded(excluded)
            order={int(k):i for i,k in enumerate(q)}
            baseline=b['prediction'][[order[int(k)] for k in validation]]
        recipe={'name':f'pair_w{args.width}_p{args.power:g}','kind':'pair','width':args.width,'power':args.power}
        _,report=fit_subset(data,fit,validation,baseline,recipe,output,args.seed,args.fixed_epochs,excluded)
        print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
