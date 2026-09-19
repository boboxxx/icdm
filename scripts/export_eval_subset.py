"""Stream only authorized validation/test subsets as a tar archive to stdout."""
import argparse
from pathlib import Path
import sys
import tarfile

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
a=p.parse_args()
paths=[]
for dataset in ['celeba','cifar10']:
    for split,count in [('val',128),('test',700)]:
        folder=a.root/dataset/split/'class0'
        images=sorted(x for x in folder.iterdir() if x.suffix.lower() in {'.png','.jpg','.jpeg'})
        if len(images)<count: raise ValueError('Insufficient images: '+str(folder))
        paths.extend(images[:count])
paths.append(a.root/'dataset_metadata.json')
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
    for path in paths: archive.add(path,arcname=str(path.relative_to(a.root)),recursive=False)
