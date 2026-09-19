"""Wait for authorized transfers, then verify every expected byte by SHA256."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=7000)
    a = p.parse_args()
    inventory = json.loads(a.inventory.read_text())
    root = a.root.resolve()
    for name in inventory:
        if root not in (root/name).resolve().parents:
            raise ValueError('Inventory path escapes asset directory')
    deadline = time.monotonic() + a.timeout
    while True:
        pending = [name for name, info in inventory.items()
                   if not (root/name).is_file() or (root/name).stat().st_size != info['size']]
        if not pending:
            break
        print(json.dumps({'waiting_for': pending[:5], 'pending': len(pending)}), flush=True)
        if time.monotonic() >= deadline:
            raise TimeoutError('Asset transfer has not completed; no GPU experiment may start')
        time.sleep(30)
    for name, info in inventory.items():
        if digest(root/name) != info['sha256']:
            raise ValueError('Asset checksum mismatch: '+name)
    overlap = {}
    for dataset in ['celeba', 'cifar10']:
        val = {v['sha256'] for k,v in inventory.items() if k.startswith('data/'+dataset+'/val/')}
        test = {v['sha256'] for k,v in inventory.items() if k.startswith('data/'+dataset+'/test/')}
        overlap[dataset] = len(val & test)
        if overlap[dataset]:
            raise ValueError('Selected val/test image overlap: '+dataset)
    report = dict(status='verified', files=len(inventory),
                  bytes=sum(v['size'] for v in inventory.values()),
                  inventory_sha256=digest(a.inventory), selected_val_test_hash_overlap=overlap,
                  caveat='Training-set overlap has not been audited; no independent test claim.')
    temporary = root/'verified.tmp'
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(root/'verified.json')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
