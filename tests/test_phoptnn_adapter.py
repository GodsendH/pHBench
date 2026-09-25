import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch

from phgeofuse.phoptnn_adapter.train import collate_graphs, create_model, predict_batch
from phgeofuse.phoptnn_adapter.graphs import atom_edges,parse_pqr


def sample(n,key):
    rng=np.random.default_rng(n)
    return dict(key=key,label=7.,positions=rng.normal(size=(n,3)).astype('float32'),
                charges=rng.normal(size=n).astype('float32'),atom_type=np.arange(n)%37,
                edge_index=np.array([(i,j) for i in range(n) for j in range(n) if abs(i-j)==1]).T,
                edge_attr=np.ones((2*(n-1),5),dtype='float32'))


class AdapterTests(unittest.TestCase):
    def test_fixed_column_pqr_handles_four_digit_residue_and_negative_coordinates(self):
        text='ATOM  15165  N   GLN A1000       6.207 -12.788 -31.866 -0.4157 1.8240\n'
        text+='ATOM      8  CE  MET A   1     -12.009-101.190 -62.445 -0.0341 1.9080\n'
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'sample.pqr';path.write_text(text)
            atoms=parse_pqr(path)
        self.assertEqual(atoms[0]['resid'],1000)
        self.assertEqual(atoms[1]['xyz'],(-12.009,-101.190,-62.445))

    def test_padding_edges_last_batch_and_rigid_invariance(self):
        torch.manual_seed(1);torch.set_num_threads(1)
        model=create_model(dict(hidden_dim=16,layers=2,attention=True),'cpu').eval()
        a,b=sample(3,'a'),sample(7,'b')
        batch=collate_graphs([a,b]);edges=batch['edges']
        self.assertTrue(torch.all(edges[:,a['edge_index'].shape[1]:]>=7))
        with torch.no_grad():
            together=predict_batch(model,batch,'cpu')
            single=torch.cat([predict_batch(model,collate_graphs([s]),'cpu') for s in [a,b]])
            torch.testing.assert_close(together,single,atol=1e-6,rtol=1e-6)
            rot=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],dtype='float32')
            moved=[{**s,'positions':s['positions']@rot+4} for s in [a,b]]
            torch.testing.assert_close(together,predict_batch(model,collate_graphs(moved),'cpu'),atol=1e-5,rtol=1e-5)
        model.train();predict_batch(model,batch,'cpu').square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_no_fabricated_peptide_across_chain_or_gap(self):
        atoms=[dict(name=n,residue='GLY',chain=ch,resid=res,xyz=(x,0,0))
               for ch,res,offset in [('A',1,0),('B',2,4),('B',4,8)]
               for n,x in [('N',offset),('CA',offset+1),('C',offset+2),('O',offset+3)]]
        edges=atom_edges(atoms)
        self.assertTrue(np.all(edges[0]//4==edges[1]//4))


if __name__=='__main__':unittest.main()
