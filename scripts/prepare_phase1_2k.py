"""Prepare deterministic 2,000/200 CelebA and CIFAR-10 subsets."""
import argparse
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


def write_image(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="PNG")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--celeba-parquet", type=Path, required=True)
    parser.add_argument("--cifar-train-parquet", type=Path, required=True)
    parser.add_argument("--cifar-val-parquet", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=200)
    args = parser.parse_args()
    if args.train_size < 1 or args.val_size < 1:
        raise ValueError("Subset sizes must be positive")

    metadata_path = args.root / "subset_metadata.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        expected = {"train_size": args.train_size, "val_size": args.val_size}
        if all(existing.get(key) == value for key, value in expected.items()):
            print("Prepared subsets already exist", flush=True)
            return
        raise RuntimeError("Existing subset metadata uses different sizes")

    total = args.train_size + args.val_size
    table = pq.read_table(args.celeba_parquet, columns=["image"], use_threads=True)
    if table.num_rows < total:
        raise RuntimeError("CelebA shard is too small")
    image_column = table["image"]
    for index in range(total):
        item = image_column[index].as_py()
        image = Image.open(io.BytesIO(item["bytes"]))
        split = "train" if index < args.train_size else "val"
        split_index = index if split == "train" else index - args.train_size
        write_image(image, args.root / "celeba" / split / "class0" / f"{split_index:05d}.png")

    for parquet, split, count in [
        (args.cifar_train_parquet, "train", args.train_size),
        (args.cifar_val_parquet, "val", args.val_size),
    ]:
        column = pq.read_table(parquet, columns=["img"])["img"]
        if len(column) < count:
            raise RuntimeError(f"CIFAR-10 {split} shard is too small")
        for index in range(count):
            item = column[index].as_py()
            image = Image.open(io.BytesIO(item["bytes"]))
            write_image(image, args.root / "cifar10" / split / "class0" / f"{index:05d}.png")

    metadata = {
        "train_size": args.train_size,
        "val_size": args.val_size,
        "celeba_source": "flwrlabs/celeba:img_align+identity+attr/train-00000-of-00019.parquet",
        "celeba_train_rows": [0, args.train_size - 1],
        "celeba_val_rows": [args.train_size, total - 1],
        "cifar10_train_rows": [0, args.train_size - 1],
        "cifar10_val_source": "official test split",
        "cifar10_val_rows": [0, args.val_size - 1],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
