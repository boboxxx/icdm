"""Check locked baseline evidence, receipts, pairing and optional server files."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path.cwd()))
from scripts.analyze_backbone_baselines import load_backbones
from scripts.analyze_public_pretrained_baselines import load_public
from scripts.analyze_vtc2027 import read_run


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-server-files', action='store_true')
    args = parser.parse_args()
    bm, br = load_backbones(Path('research/backbone_baselines/confirmation'))
    pm, pr = load_public(Path('research/pretrained_baselines/confirmation'))
    cm, cr, _ = read_run(args.reference)
    checks = {}
    for name, meta, rows, fields in [
        ('matched', bm, br, ['manifest','protocol_sha256','evaluator_sha256','trainer_adapter_sha256','external_commits','weights_sha256','images']),
        ('public', pm, pr, ['manifest','smoke','protocol_sha256','evaluator_sha256','weights_sha256','deep_historical_source_sha256','cuda_extensions_sha256','images'])
    ]:
        actual = hashlib.sha256(json.dumps({k: meta[k] for k in fields}, sort_keys=True).encode()).hexdigest()
        assert actual == meta['signature'], name + ' signature'
        assert meta['images'] == cm['images'], name + ' input hashes'
        for key in ['seeds','conditions','block_size','offset','max_images']:
            assert meta['manifest'][key] == cm['manifest'][key], (name,key)
        for row in rows:
            condition = meta['manifest']['conditions'][row['condition']]
            assert all(row[k] == v for k,v in condition.items()), 'condition labels'
        report = json.loads(Path('research', 'backbone_baselines' if name=='matched' else 'pretrained_baselines','confirmation/analysis.json').read_text())
        assert report['reference_signature'] == cm['signature']
        assert report['signature'] == meta['signature']
        checks[name] = {'signature': actual, 'records':len(rows),'image_pairs':len(meta['images']),
                        'input_hashes_seeds_conditions_match_frozen_confirmation':True}
    assert digest('scripts/train_backbone_baseline.py') == bm['trainer_adapter_sha256']
    assert digest('scripts/evaluate_backbone_baselines.py') == bm['evaluator_sha256']
    assert digest('scripts/evaluate_public_pretrained_baselines.py') == pm['evaluator_sha256']
    assert digest('research/vtc2027_sheng/BACKBONE_BASELINE_PROTOCOL.md') == bm['protocol_sha256']
    assert digest('research/vtc2027_sheng/PRETRAINED_BASELINE_PROTOCOL.md') == pm['protocol_sha256']
    receipts = {}
    for architecture in ['deepjscc','mambajscc']:
        for role in ['source','interference']:
            name = architecture+'_'+role
            folder = Path('research/backbone_baselines/formal') / name
            receipt = json.loads((folder/'receipt.json').read_text())
            logs = [json.loads(s) for s in (folder/'training.jsonl').read_text().splitlines()]
            assert receipt['status']=='complete' and receipt['epochs']==40 and receipt['smoke_batches']==0
            assert [r['epoch'] for r in logs] == list(range(1,41))
            assert all(math.isfinite(r['train_mse']) and math.isfinite(r['val_mse']) for r in logs)
            assert receipt['adapter_sha256'] == bm['trainer_adapter_sha256']
            assert receipt['external_commit'] == bm['external_commits'][architecture]
            assert receipt['best_sha256'] == bm['weights_sha256'][name]
            assert receipt['best_val_mse'] == min(r['val_mse'] for r in logs)
            assert receipt['completed_at'] < bm['started_at']
            assert receipt['complex_channel_uses']==1024 and receipt['seed']==20260922
            receipts[name] = {'epochs':40,'train_images':receipt['train_images'],
                'batches_per_epoch':receipt['train_images']//receipt['batch_size'],
                'drop_last':True, 'validation_images':receipt['validation_images'],
                'best_sha256':receipt['best_sha256'],'latest_sha256':receipt['latest_sha256'],
                'data_inventories':receipt['data_inventories']}
            if args.verify_server_files:
                for kind in ['best','latest']:
                    assert digest(folder/(kind+'.pt')) == receipt[kind+'_sha256']
                commit = subprocess.check_output(['git','-C',receipt['external_repo'],'rev-parse','HEAD'],text=True).strip()
                assert commit == receipt['external_commit']
    for role in ['source','interference']:
        assert receipts['deepjscc_'+role]['data_inventories'] == receipts['mambajscc_'+role]['data_inventories']
    if args.verify_server_files:
        for image in cm['images']:
            assert digest(image['image']) == image['image_sha256']
            assert digest(image['interference']) == image['interference_sha256']
        for role in ['source','interference']:
            effective = bm['effective_models']['mambajscc'][role]
            assert digest(effective['cuda_extension']) == effective['cuda_extension_sha256']
        for path, expected in pm['cuda_extensions_sha256'].items():
            assert digest(path) == expected
        # Recompute training and full validation inventories from actual file content.
        for role, dataset in [('source','celeba'),('interference','cifar10')]:
            receipt = json.loads(Path(f'research/backbone_baselines/formal/deepjscc_{role}/receipt.json').read_text())
            for split, label in [('train','train'),('val','validation_full')]:
                root = Path(receipt['data_root'])/dataset/split
                files = sorted(p for p in root.rglob('*') if p.suffix.lower() in ['.jpg','.jpeg','.png','.bmp','.ppm','.pgm','.tif','.tiff','.webp'])
                h=hashlib.sha256(); total=0
                for path in files:
                    h.update(str(path.relative_to(root)).encode()); h.update(b'\0')
                    h.update(bytes.fromhex(digest(path))); total+=path.stat().st_size
                assert {'sha256':h.hexdigest(),'files':len(files),'bytes':total} == receipt['data_inventories'][label]
    report={'status':'passed','checked_at':time.time(),'reference_signature':cm['signature'],
            'checks':checks,'training':receipts,'source_protocol_hashes_verified':True,
            'server_weight_extension_input_training_file_hashes_verified':args.verify_server_files,
            'scope':'Cross-backbone inputs match; encoded received tensors differ. Forty scheduled epochs drop final incomplete training batches.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'server_files':args.verify_server_files,'records':[len(br),len(pr)]}))


if __name__ == '__main__': main()
