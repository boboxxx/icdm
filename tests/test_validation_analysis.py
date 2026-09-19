import json
from pathlib import Path
import tempfile
import unittest
from scripts.analyze_moment_validation import analyze


class ValidationAnalysisTests(unittest.TestCase):
    def fixture(self, root, duplicate=False, gain=.2):
        modes = ['NO_ICDM','A3','A4','A4_R','A5_DIAG','A5_FULL']
        manifest = dict(max_images=8,seeds=[700,701],snr_sinr_pairs=[[20,-7],[20,-4]],
                        profiles=['stationary','alternating','burst'],modes=modes)
        (root/'metadata.json').write_text(json.dumps(dict(status='complete',manifest=manifest,input_signature='fixture')))
        rows = []
        for i in range(8):
            for seed in manifest['seeds']:
                for snr,sinr in manifest['snr_sinr_pairs']:
                    for profile in manifest['profiles']:
                        for mode in modes:
                            rows.append(dict(index=i,seed=seed,snr=snr,true_global_sinr=sinr,profile=profile,
                                mode=mode,image=str(i),interference_image=str(i),mse=.01,
                                psnr=20+(gain if mode=='A5_FULL' else 0),receiver_seconds=.2))
        if duplicate: rows.append(rows[0])
        (root/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def test_image_cluster_gate_positive_and_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for gain in [.2,-.2]:
                self.fixture(root,gain=gain)
                report = analyze(root)
                self.assertAlmostEqual(report['primary']['mean_db'],gain)
                self.assertEqual(report['primary']['image_clusters'],8)
                self.assertEqual(report['validation_gate_passed'],gain>0)

    def test_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.fixture(root,duplicate=True)
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                analyze(root)


if __name__ == '__main__':
    unittest.main()
