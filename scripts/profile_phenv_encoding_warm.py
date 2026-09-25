"""Reprofile frozen ESM after explicit warmup; never cache the full dataset.

Use the same 36 length-stratified sequences as the first resource probe.
Report startup separately so a cold first CUDA call is not multiplied by
every sequence in a length stratum. No model is fitted or PHOPT label read.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from localph.phenv_data import TRANSLATE,sha_file,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('audit','plan','profile','pilot-data','pilot-cache','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=Path.home()/'.cache/torch/hub/checkpoints/esm1v_t33_650M_UR90S_1.pt')
    a=p.parse_args()
    original=json.loads((a.profile/'result.json').read_text())
    ac=json.loads((a.audit/'report.json').read_text())
    plan=json.loads((a.plan/'plan.json').read_text())
    if (sha_file(a.audit/'manifest.sqlite')!=ac['artifact_sha256']['manifest.sqlite']
        or plan['audit_report_sha256']!=sha_file(a.audit/'report.json')
        or plan['selection_sha256']!=sha_file(a.plan/'selection.sqlite')
        or sha_file(a.checkpoint)!=original['encoder_checkpoint_sha256']):
        raise ValueError('resource probe sources differ')
    a.output.mkdir(parents=True,exist_ok=False)
    db=sqlite3.connect(f'file:{(a.audit/"manifest.sqlite").resolve()}?mode=ro',uri=True)
    db.execute('ATTACH DATABASE ? AS plan',(f'file:{(a.plan/"selection.sqlite").resolve()}?mode=ro',))
    edges=[r['length_upper_inclusive'] for r in original['length_strata']]
    populations={}
    for name,join in [('all_eligible_candidates','search_candidates'),('expanded_before_homology','plan.selected')]:
        bins=Counter()
        residues=0
        for length,count in db.execute(f'SELECT length,COUNT(*) FROM records JOIN {join} USING(row_id) GROUP BY length'):
            bins[int(np.searchsorted(edges,length,side='left'))]+=count
            residues+=length*count
        populations[name]={'length_bin_counts':[bins[i] for i in range(len(edges))],'residues':residues}
    with (a.pilot_data/'records.csv').open(newline='') as f:
        lengths=[len(r['sequence']) for r in csv.DictReader(f)]
    bins=np.bincount(np.searchsorted(edges,lengths,side='left'),minlength=len(edges))
    populations['completed_pilot']={'length_bin_counts':bins.tolist(),'residues':sum(lengths)}
    cc=json.loads((a.pilot_cache/'complete.json').read_text())
    if cc['rows']!=len(lengths) or cc['residues']!=sum(lengths):
        raise ValueError('pilot cache coverage differs')
    import torch
    import esm
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    start=time.monotonic()
    model,alphabet=esm.pretrained.load_model_and_alphabet_local(str(a.checkpoint))
    model=model.float().cuda().eval().requires_grad_(False)
    torch.cuda.synchronize()
    load_seconds=time.monotonic()-start
    converter=alphabet.get_batch_converter()
    strata=[]
    for row in original['length_strata']:
        samples=[]
        for measurement in row['measurements']:
            rid=measurement['row_id']
            seq,length=db.execute('SELECT sequence,length FROM records WHERE row_id=?',(rid,)).fetchone()
            if length!=measurement['length']:
                raise ValueError('probe sequence length changed')
            samples.append((rid,seq.translate(TRANSLATE),length))
        def timed(sample):
            rid,seq,length=sample
            _,_,tokens=converter([(str(rid),seq)])
            tokens=tokens.cuda()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begin=time.monotonic()
            with torch.inference_mode():
                h=model(tokens,repr_layers=[33],return_contacts=False)['representations'][33][0,1:-1]
                packed=h.half().cpu().numpy()
            torch.cuda.synchronize()
            elapsed=time.monotonic()-begin
            if packed.shape!=(length,1280) or not np.isfinite(packed).all():
                raise ValueError('invalid full-residue coverage')
            return {'seconds':elapsed,'peak_cuda_bytes':torch.cuda.max_memory_allocated()}
        warmup=[timed(samples[0]) for _ in range(2)]
        measurements=[]
        for sample in samples:
            repeats=[timed(sample) for _ in range(3)]
            measurements.append({'row_id':sample[0],'length':sample[2],'repeats':repeats,
                'mean_seconds':float(np.mean([v['seconds'] for v in repeats]))})
        entry={'lower_exclusive':row['length_lower_exclusive'],'upper_inclusive':row['length_upper_inclusive'],
            'warmup':warmup,'measurements':measurements,
            'mean_seconds':float(np.mean([v['mean_seconds'] for v in measurements]))}
        strata.append(entry)
        print(json.dumps({'stratum_upper':entry['upper_inclusive'],'warm_mean_seconds':entry['mean_seconds']}),flush=True)
    db.close()
    for pop in populations.values():
        pop['rows']=sum(pop['length_bin_counts'])
        pop['encoding_seconds']=load_seconds+sum(n*s['mean_seconds'] for n,s in zip(pop['length_bin_counts'],strata))
        pop['encoding_gpu_hours']=pop['encoding_seconds']/3600
        pop['fp16_token_bytes_excluding_headers']=pop['residues']*1280*2
    result={'state':'complete','scope':'Resource estimate, not a completed full encoding or training run',
        'encoder':original['encoder'],'checkpoint_sha256':original['encoder_checkpoint_sha256'],
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'precision':'float32','batch_size':1,
        'load_seconds':load_seconds,'strata':strata,'populations':populations,
        'pilot_actual_encoding_session_seconds':cc['session_seconds'],
        'original_estimate_gpu_hours':original['full_encoding_estimated_gpu_hours'],
        'why_reprofile':'Original short-length mean included the first cold CUDA forward call, which should not be multiplied by every short protein.',
        'limitations':['Small length-stratified probe; not an exact runtime forecast.',
            'Excludes full cache writes, filesystem checks, head training and scheduler queuing.',
            'Expanded candidate counts precede homology purging; re-estimate final retained data.',
            'Repeated same-shape forwards may be faster than a changing-length production stream.'],
        'source_sha256':sha_file(Path(__file__)),'original_profile_sha256':sha_file(a.profile/'result.json'),
        'plan_sha256':sha_file(a.plan/'plan.json'),'phopt_labels_consumed':False,'full_encoding_started':False}
    write_json(a.output/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='strata'}),flush=True)


if __name__=='__main__':
    main()
