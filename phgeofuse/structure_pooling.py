"""Fixed label-free regional pooling of cached SaProt residue embeddings."""
import numpy as np


def regional_pool(sequence, embedding, rsa, plddt, projection):
    h=np.asarray(embedding,dtype=np.float64)
    rsa=np.asarray(rsa,dtype=float);plddt=np.asarray(plddt,dtype=float)
    n=len(sequence)
    if h.ndim!=2 or h.shape[0]!=n or rsa.shape!=(n,) or plddt.shape!=(n,):
        raise ValueError('residue/structure length mismatch')
    if not all(np.isfinite(x).all() for x in (h,rsa,plddt,projection)):
        raise ValueError('nonfinite residue features')
    mean=h.mean(0);std=h.std(0)
    norm=max(float(np.linalg.norm(mean)),1e-12)
    aa=np.asarray(list(sequence))
    masks=[rsa>=.25,rsa<.25,np.isin(aa,list('DE')),
           np.isin(aa,list('HKR')),aa=='H',aa=='C']
    features=[]
    for mask in masks:
        missing=not mask.any()
        delta=np.zeros(h.shape[1]) if missing else (h[mask].mean(0)-mean)/norm
        features.extend([delta@projection,np.array([mask.mean(),float(missing)])])
    features.append(np.array([np.clip(plddt.mean()/100.,0.,1.),
        np.mean(plddt<70),np.mean(plddt<50),np.mean(rsa),np.std(rsa)]))
    return mean.astype(np.float32),std.astype(np.float32),np.concatenate(features).astype(np.float32)
