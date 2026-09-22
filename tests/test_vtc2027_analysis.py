import json
import tempfile
import unittest
from pathlib import Path
from scripts.analyze_vtc2027 import read_run,contrast,fit_policy


class StudyAnalysisTests(unittest.TestCase):
    def test_balanced_count_does_not_hide_wrong_condition(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'metadata.json').write_text(json.dumps({'status':'complete','expected_records':1,
                'images':[{'index':5}],'manifest':{'modes':['A4'],'seeds':[10],'conditions':[{}]}}))
            (p/'records.jsonl').write_text(json.dumps({'index':5,'seed':10,'condition':1,'mode':'A4'})+'\n')
            with self.assertRaisesRegex(ValueError,'Unexpected or missing'):
                read_run(p)

    def test_cluster_unit_and_known_paired_effect(self):
        groups=[]
        for i in range(8):
            for condition in range(10):
                a=dict(index=i,psnr=20,mse=.01,lpips_vgg=.3,ms_ssim=.8)
                b=dict(a,psnr=20.2)
                groups.append({'A4':a,'SELECT':b})
        result=contrast(groups,'SELECT','A4')
        self.assertEqual(result['image_clusters'],8)
        self.assertAlmostEqual(result['mean_delta_psnr'],.2)
        self.assertAlmostEqual(result['ci95_image_cluster'][0],.2)

    def test_confirmation_cannot_fit_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'restricted'):
                fit_policy([],Path(tmp),{'manifest':{'stage':'confirmation'},'images':[{}]*64})

    def test_constant_receiver_prevents_spurious_selector_claim(self):
        groups=[]
        for i in range(64):
            a={'index':i,'features':{'power':i/8,'heterogeneity':i/64},'psnr':22,'receiver_seconds':1}
            b=dict(a,psnr=20,receiver_seconds=.5)
            groups.append({'A4':a,'CDDM_BLIND':b})
        with tempfile.TemporaryDirectory() as tmp:
            report,_=fit_policy(groups,Path(tmp),{'manifest':{'stage':'development'},'images':[{}]*64,'signature':'fixture'})
            self.assertEqual(report['selected']['scientific_status'],'no_cv_gain_over_fixed')
            self.assertEqual(report['selected']['family'],'fixed_A4')

    def test_high_power_gaussian_region_is_learnable(self):
        groups=[]
        for i in range(64):
            high=i>=32
            a={'index':i,'features':{'power':2 if high else .2,'heterogeneity':.1},
               'psnr':18 if high else 22,'receiver_seconds':1}
            b=dict(a,psnr=24 if high else 20,receiver_seconds=.5)
            groups.append({'A4':a,'CDDM_BLIND':b})
        with tempfile.TemporaryDirectory() as tmp:
            report,_=fit_policy(groups,Path(tmp),{'manifest':{'stage':'development'},'images':[{}]*64,'signature':'fixture'})
            self.assertEqual(report['selected']['gaussian_side'],'above')
            self.assertAlmostEqual(report['energy']['cv_psnr'],23)


if __name__=='__main__': unittest.main()
