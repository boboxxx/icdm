"""Download and prepare the full CelebA/CIFAR-10 phase-one datasets.

The operation is resumable at both the Hugging Face download layer and the
per-image extraction layer. Images are copied from parquet without recompressing
them, which preserves the source pixels and avoids unnecessary CPU work.
"""
import argparse
import json
from pathlib import Path
import time

from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq


CELEBA_REPO = "flwrlabs/celeba"
CIFAR_REPO = "uoft-cs/cifar10"
EXPECTED = {
    "celeba_train": 162770,
    "celeba_val": 19867,
    "celeba_test": 19962,
    "cifar10_train": 50000,
    "cifar10_official_test": 10000,
}


def log(**record):
    record["time"] = time.time()
    print(json.dumps(record, allow_nan=False), flush=True)


def source_files(repo, prefix):
    files = HfApi().list_repo_files(repo, repo_type="dataset")
    matches = sorted(path for path in files if path.startswith(prefix) and path.endswith(".parquet"))
    if not matches:
        raise RuntimeError(f"No parquet shards found for {repo}:{prefix}")
    return matches


def download(repo, filenames, cache):
    paths = []
    for filename in filenames:
        log(event="download_start", repo=repo, file=filename)
        path = hf_hub_download(
            repo_id=repo,
            filename=filename,
            repo_type="dataset",
            local_dir=cache / repo.replace("/", "--"),
        )
        paths.append(Path(path))
        log(event="download_complete", repo=repo, file=filename, bytes=Path(path).stat().st_size)
    return paths


def write_bytes(path, value):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(value)
    temporary.replace(path)


def extract(paths, column, destination, expected, offset=0, limit=None):
    destination.mkdir(parents=True, exist_ok=True)
    seen = 0
    written = 0
    stop = expected if limit is None else offset + limit
    for shard_index, path in enumerate(paths):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=512, columns=[column], use_threads=True):
            for item in batch.column(0).to_pylist():
                if offset <= seen < stop:
                    value = item["bytes"]
                    if value is None:
                        raise RuntimeError(f"Missing embedded image bytes in {path}")
                    write_bytes(destination / f"{written:06d}.png", value)
                    written += 1
                seen += 1
        log(event="extract_shard", destination=str(destination), shard=shard_index + 1, shards=len(paths), source_rows=seen, output_rows=written)
    expected_output = expected if limit is None else limit
    if written != expected_output:
        raise RuntimeError(f"Expected {expected_output} rows for {destination}, got {written}")
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    metadata_path = args.root / "dataset_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("status") == "complete" and metadata.get("counts") == EXPECTED:
            log(event="already_complete", root=str(args.root))
            return

    celeba = {
        split: source_files(CELEBA_REPO, f"img_align+identity+attr/{split}-")
        for split in ("train", "valid", "test")
    }
    cifar = {
        split: [f"plain_text/{split}-00000-of-00001.parquet"]
        for split in ("train", "test")
    }
    downloaded_celeba = {split: download(CELEBA_REPO, files, args.cache) for split, files in celeba.items()}
    downloaded_cifar = {split: download(CIFAR_REPO, files, args.cache) for split, files in cifar.items()}

    counts = {}
    counts["celeba_train"] = extract(downloaded_celeba["train"], "image", args.root / "celeba/train/class0", EXPECTED["celeba_train"])
    counts["celeba_val"] = extract(downloaded_celeba["valid"], "image", args.root / "celeba/val/class0", EXPECTED["celeba_val"])
    counts["celeba_test"] = extract(downloaded_celeba["test"], "image", args.root / "celeba/test/class0", EXPECTED["celeba_test"])
    counts["cifar10_train"] = extract(downloaded_cifar["train"], "img", args.root / "cifar10/train/class0", EXPECTED["cifar10_train"])
    counts["cifar10_official_test"] = EXPECTED["cifar10_official_test"]
    extract(downloaded_cifar["test"], "img", args.root / "cifar10/val/class0", EXPECTED["cifar10_official_test"], offset=0, limit=5000)
    extract(downloaded_cifar["test"], "img", args.root / "cifar10/test/class0", EXPECTED["cifar10_official_test"], offset=5000, limit=5000)

    metadata = {
        "status": "complete",
        "counts": counts,
        "cifar10_validation_rows": [0, 4999],
        "cifar10_test_rows": [5000, 9999],
        "sources": {
            "celeba": {"repo": CELEBA_REPO, "files": celeba},
            "cifar10": {"repo": CIFAR_REPO, "files": cifar},
        },
        "completed_at": time.time(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    log(event="complete", root=str(args.root), counts=counts)


if __name__ == "__main__":
    main()
