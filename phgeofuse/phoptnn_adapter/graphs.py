"""Portable atom graphs from fixed AMBER PQR; no task labels in this cache."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from .constants import ALL_ATOM_LABELS, DICT_AA_RESIDUE_EDGES

GRAPH_VERSION = 'phoptnn_amber_heavy_v1'
ALIASES = {'HID':'HIS', 'HIE':'HIS', 'HIP':'HIS', 'ASH':'ASP', 'GLH':'GLU',
           'CYX':'CYS', 'CYM':'CYS', 'LYN':'LYS'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True)+'\n')
    os.replace(tmp, path)


def parse_pqr(path):
    atoms = []
    for line in Path(path).read_text().splitlines():
        if not line.startswith('ATOM'):
            continue
        # PDB2PQR's fixed columns can contain A1000 or adjacent negative xyz.
        # Whitespace tokenization alone loses the chain and coordinate boundaries.
        try:
            atom, residue = line[12:16].strip(), line[17:20].strip()
            chain, resid = line[21:22].strip() or 'A', int(line[22:26])
            xyz = tuple(float(line[a:b]) for a,b in [(30,38),(38,46),(46,54)])
            charge,radius=map(float,line[54:].split())
        except (ValueError,IndexError):
            p = line.split()
            if len(p) == 10:
                p.insert(4, 'A')
            if len(p) != 11:
                raise ValueError(f'unsupported PQR tokens: {line}')
            atom, residue, chain, resid = p[2:6]
            resid=int(resid);xyz=tuple(map(float,p[6:9]));charge,radius=map(float,p[9:11])
        if atom.startswith('H'):
            continue
        if atom not in ALL_ATOM_LABELS:
            raise ValueError(f'unsupported heavy atom {atom}')
        residue = ALIASES.get(residue, residue)
        if residue not in DICT_AA_RESIDUE_EDGES:
            raise ValueError(f'unsupported residue {residue}')
        atoms.append(dict(name=atom, residue=residue, chain=chain, resid=int(resid),
                          xyz=xyz, charge=charge, radius=radius))
    keys = [(a['chain'],a['resid'],a['name']) for a in atoms]
    if not atoms or len(keys) != len(set(keys)):
        raise ValueError('empty or duplicate PQR atom identities')
    return atoms


def atom_edges(atoms):
    residues = {}
    for i, a in enumerate(atoms):
        residues.setdefault((a['chain'],a['resid']), []).append((i,a))
    edges = set(); previous = None
    def add(i,j):
        edges.add((i,j)); edges.add((j,i))
    for key, group in residues.items():
        names = {a['name']:i for i,a in group}
        residue = group[0][1]['residue']
        for a,b in [('N','CA'),('CA','C'),('C','O'),('C','OXT'),('CA','CB'),*DICT_AA_RESIDUE_EDGES[residue]]:
            if a in names and b in names:
                add(names[a],names[b])
        if previous is not None:
            prevkey, prevnames = previous
            if key[0] == prevkey[0] and key[1] == prevkey[1]+1 and 'C' in prevnames and 'N' in names:
                i,j = prevnames['C'],names['N']
                if np.linalg.norm(np.subtract(atoms[i]['xyz'],atoms[j]['xyz'])) < 2.2:
                    add(i,j)
        previous = (key,names)
    result = np.asarray(sorted(edges),dtype=np.int64).T
    if result.size == 0:
        raise ValueError('graph has no edges')
    return result


def edge_attributes(atoms, edges):
    # Construct RDKit's heavy-atom molecule in exactly the PQR node order.
    # Verify serials after parsing; CIF array indices are never mixed with PQR.
    features = {}; rdkit_status = 'ok'
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog('rdApp.*')
        lines = []
        for i,a in enumerate(atoms):
            x,y,z = a['xyz']
            lines.append(f"ATOM  {i+1:5d} {a['name']:>4s} {a['residue']:3s} {a['chain'][:1]}{a['resid']:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{1:6.2f}{0:6.2f}          {a['name'][0]:>2s}  ")
        mol = Chem.MolFromPDBBlock('\n'.join(lines)+'\nEND\n', removeHs=False, sanitize=False)
        if mol is None or mol.GetNumAtoms()!=len(atoms):
            raise ValueError('RDKit atom count changed')
        serials = [a.GetPDBResidueInfo().GetSerialNumber() for a in mol.GetAtoms()]
        if serials != list(range(1,len(atoms)+1)):
            raise ValueError('RDKit atom order changed')
        Chem.SanitizeMol(mol)
        types = [Chem.BondType.SINGLE,Chem.BondType.DOUBLE,Chem.BondType.TRIPLE,Chem.BondType.AROMATIC]
        for b in mol.GetBonds():
            features[tuple(sorted((b.GetBeginAtomIdx(),b.GetEndAtomIdx())))] = [float(b.GetBondType()==t) for t in types]+[float(b.IsInRing())]
    except Exception as exc:
        rdkit_status = f'distance_fallback: {type(exc).__name__}: {exc}'
    pos = np.asarray([a['xyz'] for a in atoms],dtype=np.float32)
    attr=[]; fallback=0
    for i,j in edges.T:
        feature=features.get(tuple(sorted((int(i),int(j)))))
        if feature is None:
            d=float(np.linalg.norm(pos[i]-pos[j])); idx=0 if d<1.6 else 1 if d<3 else 2 if d<5 else 3
            feature=[float(k==idx) for k in range(4)]+[0.]; fallback+=1
        attr.append(feature)
    return np.asarray(attr,dtype=np.float32),rdkit_status,fallback


def graph_arrays(atoms):
    pos = np.array([a['xyz'] for a in atoms],dtype=np.float32)
    charge = np.array([a['charge'] for a in atoms],dtype=np.float32)
    atom_type = np.array([ALL_ATOM_LABELS.index(a['name']) for a in atoms],dtype=np.int64)
    edges = atom_edges(atoms)
    attr,rdkit_status,fallback = edge_attributes(atoms,edges)
    if not all(np.isfinite(a).all() for a in [pos,charge,attr]):
        raise ValueError('nonfinite graph')
    n=len(atoms)
    if edges.min()<0 or edges.max()>=n:
        raise ValueError('edge out of bounds')
    return dict(positions=pos, charges=charge, atom_type=atom_type, edge_index=edges, edge_attr=attr,
                residue_number=np.array([a['resid'] for a in atoms],dtype=np.int32),
                atom_keys=np.array([f"{a['chain']}:{a['resid']}:{a['name']}" for a in atoms])), {
                    'nodes':n,'edges':edges.shape[1], 'isolated_nodes':n-len(np.unique(edges)),
                    'rdkit_status':rdkit_status,'distance_fallback_edges':fallback}


def build_one(job):
    row, root, converter, timeout = job
    root=Path(root); key=row['structure_sha256']
    folder=root/'atom_graphs'/key[:2];folder.mkdir(parents=True,exist_ok=True)
    graph=folder/(key+'.npz');meta=folder/(key+'.json');pqr=folder/(key+'.pqr')
    started=time.time()
    result=dict(structure_sha256=key, graph_path=str(graph), status='failed')
    try:
        if graph.exists() and meta.exists():
            info=json.loads(meta.read_text())
            if info['graph_version']==GRAPH_VERSION and info['structure_sha256']==key and info['graph_sha256']==digest(graph):
                return {**info,'cached':True}
        if digest(row['structure_path'])!=key:
            raise ValueError('input PDB hash mismatch')
        command=[converter,'--ff=AMBER','--keep-chain',row['structure_path'],str(pqr)]
        run=subprocess.run(command,capture_output=True,text=True,timeout=timeout)
        (folder/(key+'.log')).write_text(run.stdout+run.stderr)
        if run.returncode:
            raise RuntimeError(f'PDB2PQR exit {run.returncode}: {run.stderr[-1200:]}')
        atoms=parse_pqr(pqr)
        arrays,info=graph_arrays(atoms)
        # Every target residue must retain its CA; avoid silently training truncated structures.
        ca_count=sum(a['name']=='CA' for a in atoms)
        if ca_count!=len(row['sequence']):
            raise ValueError(f'PQR coverage {ca_count}/{len(row["sequence"])}')
        tmp=graph.with_suffix('.tmp.npz');np.savez(tmp,**arrays);os.replace(tmp,graph)
        result.update(info,status='ready', graph_version=GRAPH_VERSION, pqr_sha256=digest(pqr),
                      graph_sha256=digest(graph),command=command, seconds=time.time()-started,
                      sequence_sha256=row['sequence_sha256'],coverage=ca_count/len(row['sequence']),
                      charge_policy='PDB2PQR AMBER default states; no target-dependent pH')
        atomic_json(meta,result)
        (folder/(key+'.failure.json')).unlink(missing_ok=True)
    except Exception as exc:
        result.update(error=f'{type(exc).__name__}: {exc}',seconds=time.time()-started)
        atomic_json(folder/(key+'.failure.json'),result)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--workers',type=int,default=12)
    ap.add_argument('--timeout',type=int,default=600)
    ap.add_argument('--limit',type=int)
    ap.add_argument('--converter',default='/home/hetianci/envs/miniforge3/envs/pHoptNN/bin/pdb2pqr30')
    args=ap.parse_args();args.output=args.output.resolve();args.manifest=args.manifest.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader(args.manifest.open()))
    unique={r['structure_sha256']:r for r in rows}
    jobs=[(r,str(args.output),args.converter,args.timeout) for r in unique.values()]
    if args.limit:jobs=jobs[:args.limit]
    started=time.time();results={}
    def status(state):
        atomic_json(args.output/'atom_graph_status.json',dict(status=state,pid=os.getpid(),updated=time.time(),
                    completed=len(results),total=len(jobs),ready=sum(r['status']=='ready' for r in results.values()),
                    failures=sum(r['status']!='ready' for r in results.values()),seconds=time.time()-started))
    status('running')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(build_one,j) for j in jobs]):
            result=future.result();results[result['structure_sha256']]=result
            if len(results)%25==0 or result['status']!='ready':
                status('running');print('ATOMS',len(results),len(jobs),'failures',sum(r['status']!='ready' for r in results.values()),flush=True)
    output=[]
    for r in rows:
        if r['structure_sha256'] not in results:continue
        info=results[r['structure_sha256']]
        output.append(dict(key=f"{r['split']}::{r['protein_id']}", protein_id=r['protein_id'], split=r['split'],
                           sequence_sha256=r['sequence_sha256'],structure_sha256=r['structure_sha256'],
                           label=float(r['ph_opt']), graph_path=str(Path(info['graph_path']).resolve()) if info.get('graph_path') else '',status=info['status'],
                           error=info.get('error','')))
    with (args.output/'atom_manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    atomic_json(args.output/'atom_graph_audit.json',dict(graph_version=GRAPH_VERSION,manifest_sha256=digest(args.manifest),
                unique_results=list(results.values()),sample_count=len(output)))
    status('complete' if all(r['status']=='ready' for r in results.values()) else 'complete_with_failures')


if __name__=='__main__':main()
