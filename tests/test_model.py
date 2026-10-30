"""Input-contract, covariance, decomposition and checkpoint integrity tests."""
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from phytopert import PhytoPert, summarize_counts, build_episode

ROOT = Path(__file__).resolve().parents[1]

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model = PhytoPert.from_pretrained(ROOT / "checkpoints/arabidopsis")
        rng = np.random.default_rng(42)
        genes = cls.model.genes
        cls.summaries = [summarize_counts(rng.poisson(1, (8, len(genes))), genes, genes, str(i)) for i in range(5)]
        q,c,t1,t2,t3 = cls.summaries
        cls.episode = build_episode(q, [c,c,c], [t1,t2,t3])

    def test_shared_control_covariance(self):
        np.testing.assert_array_equal(self.episode['support_cov'][0,0,1], self.summaries[1].variance_of_mean)
        np.testing.assert_allclose(self.episode['support_cov'][0,0,0], self.summaries[1].variance_of_mean + self.summaries[2].variance_of_mean)

    def test_zero_response_and_bounds(self):
        zero = dict(self.episode, support_effect=np.zeros_like(self.episode['support_effect']))
        np.testing.assert_allclose(self.model.predict(**zero).numpy(), 0, atol=1e-7)
        pred = self.model.predict(**self.episode).numpy()
        treated = pred + self.episode['query_control']
        self.assertTrue((treated >= -1e-6).all())
        self.assertTrue((treated[...,1] <= 1+1e-6).all())

    def test_additive_explanation(self):
        prediction, c = self.model.predict(**self.episode, return_components=True)
        actual = sum(c[k] for k in ('base','gene_correction','program_correction','context_correction','projection_adjustment'))
        torch.testing.assert_close(prediction, actual, atol=1e-6, rtol=1e-5)

    def test_support_order_invariance(self):
        order=[2,0,1];e=self.episode
        changed=dict(e,support_control=e['support_control'][:,order],support_effect=e['support_effect'][:,order],available=e['available'][:,order],support_cov=e['support_cov'][:,order][:,:,order])
        torch.testing.assert_close(self.model.predict(**e),self.model.predict(**changed),atol=2e-6,rtol=1e-5)

    def test_invalid_covariance_rejected(self):
        cov=self.episode['support_cov'].copy();cov[0,0,1,0,0]=99
        with self.assertRaises(ValueError):self.model.predict(**dict(self.episode,support_cov=cov))

    def test_unavailable_genes(self):
        mask=self.episode['available'].copy();mask[:,:,0]=False
        prediction=self.model.predict(**dict(self.episode,available=mask))
        torch.testing.assert_close(prediction[:,0],torch.zeros_like(prediction[:,0]))

    def test_checksum_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            dest=Path(directory)/'checkpoint';shutil.copytree(ROOT/'checkpoints/arabidopsis',dest)
            with (dest/'config.json').open('a') as f:f.write(' ')
            with self.assertRaisesRegex(ValueError,'Checksum'):PhytoPert.from_pretrained(dest)

    def test_normalization_before_gene_selection(self):
        x=np.array([[1,3],[2,2]])
        summary=summarize_counts(x,['a','b'],['a','missing'],'sample')
        np.testing.assert_allclose(summary.mean[0,0],np.log1p([2500,5000]).mean(),rtol=1e-6)
        np.testing.assert_array_equal(summary.measured,[True,False])
        expected=np.var(np.log1p([2500,5000]),ddof=1)/2
        self.assertAlmostEqual(float(summary.variance_of_mean[0,0]),expected,places=6)

    def test_mismatched_endpoint_identity_rejected(self):
        from dataclasses import replace
        c=self.summaries[1]
        changed=replace(c,n_cells=c.n_cells+1)
        with self.assertRaisesRegex(ValueError,'different sampled cells'):
            build_episode(self.summaries[0],[c,changed],self.summaries[2:4])

if __name__ == '__main__':
    unittest.main()
