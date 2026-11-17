#!/usr/bin/env python3
"""
Package the pre-trained DreaMS model weights into a Kaggle Dataset and
(optionally) upload it via the official ``kaggle`` CLI.

This makes the weights available inside Kaggle notebooks as a Dataset, so a
notebook can fetch them quickly instead of downloading from Hugging Face on
the first run.

Model files packaged
--------------------
- ``embedding_model.ckpt`` (~1.2 GB) - embedding model used by the README
  example (``dreams_embeddings``).
- ``ssl_model.ckpt`` (~1.4 GB)      - the pre-trained SSL backbone used for
  fine-tuning heads.

Both are downloaded from the Hugging Face repo ``roman-bushuiev/DreaMS`` (to
``dreams/models/pretrained/``) if not already present locally.

Usage
-----
1. Install the Kaggle CLI and set up credentials (one time)::

      pip install kaggle
      # put your Kaggle username and API key in ~/.kaggle/kaggle.json

2. Package the weights (no upload)::

      python scripts/export_kaggle_dataset.py

3. Create the dataset or push a new version::

      python scripts/export_kaggle_dataset.py --upload
      python scripts/export_kaggle_dataset.py --upload --message "Updated weights"

The dataset slug is ``{username}/dreams-pretrained`` (override with
``--dataset-id``). ``--upload`` creates the dataset the first time and
uploads a new version on later runs.

Example: loading the dataset in a Kaggle notebook
-------------------------------------------------
.. code-block:: python

    import pathlib
    DST = pathlib.Path("/kaggle/working/DreaMS/dreams/models/pretrained")
    DST.mkdir(parents=True, exist_ok=True)
    for ckpt in ("embedding_model.ckpt", "ssl_model.ckpt"):
        src = pathlib.Path(f"/kaggle/input/dreams-pretrained/{ckpt}")
        dst = DST / ckpt
        if not dst.exists():
            dst.write_bytes(src.read_bytes())
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

#: Hugging Face repo hosting the published DreaMS weights.
HF_REPO = "roman-bushuiev/DreaMS"

#: Model files to package (relative to the local pretrained dir).
MODEL_FILES = ("embedding_model.ckpt", "ssl_model.ckpt")


def repo_pretrained_dir() -> Path:
    """Location of the local pretrained checkpoints (dreams/models/pretrained)."""
    return Path(__file__).resolve().parent.parent / "dreams" / "models" / "pretrained"


def ensure_weights(local_dir: Path) -> None:
    """Download any missing checkpoint into ``local_dir`` from the HF repo."""
    from huggingface_hub import hf_hub_download

    local_dir.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        target = local_dir / name
        if target.exists() and target.stat().st_size > 0:
            print(f"[*] found {name} ({target.stat().st_size / 1e9:.2f} GB)")
            continue
        print(f"[*] downloading {name} from {HF_REPO} ...")
        hf_hub_download(
            repo_id=HF_REPO,
            filename=name,
            local_dir=local_dir,
            repo_type="model",
            force_download=True,
        )


def write_metadata(staging: Path, dataset_id: str, title: str, subtitle: str) -> None:
    metadata = {
        "id": dataset_id,
        "title": title,
        "subtitle": subtitle,
        "licenses": [{"name": "MIT"}],
    }
    with (staging / "kaggle-metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[*] wrote {staging / 'kaggle-metadata.json'}")


def package(local_dir: Path, staging: Path, dataset_id: str, title: str, subtitle: str) -> None:
    """Copy the checkpoints and write Kaggle metadata into ``staging``."""
    ensure_weights(local_dir)

    staging.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        src = local_dir / name
        if not src.exists():
            raise FileNotFoundError(f"missing weight file: {src}")
        dst = staging / name
        # Hardlink when possible (same filesystem) to avoid doubling ~2.6 GB.
        if not dst.exists():
            try:
                dst.hardlink_to(src)
            except OSError:
                shutil.copy2(src, dst)
        print(f"[*] packaged {name} -> {dst} ({dst.stat().st_size / 1e9:.2f} GB)")

    write_metadata(staging, dataset_id, title, subtitle)
    print(f"\nDataset staging directory ready: {staging}")
    print("Files: " + ", ".join(sorted(p.name for p in staging.iterdir())))


def run_kaggle(staging: Path, update: bool, message: str) -> None:
    """Create a new dataset or upload a new version with the ``kaggle`` CLI."""
    if not shutil.which("kaggle"):
        print(
            "\n[!] 'kaggle' CLI not found. Install it with `pip install kaggle` and\n"
            "    add your credentials to ~/.kaggle/kaggle.json, then re-run with --upload.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    if update:
        cmd = ["kaggle", "datasets", "version", "-p", str(staging), "-m", message]
    else:
        cmd = ["kaggle", "datasets", "create", "-p", str(staging)]

    print(f"[*] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=repo_pretrained_dir(),
        help=f"local dir with checkpoints (default: {repo_pretrained_dir()})",
    )
    parser.add_argument("--staging-dir", type=Path, default=Path("build/dreams_kaggle_dataset"))
    parser.add_argument(
        "--dataset-id",
        default="amansinclair/dreams-pretrained",
        help="{username}/slug of the Kaggle dataset (default: amansinclair/dreams-pretrained)",
    )
    parser.add_argument("--title", default="DreaMS Pre-trained Model Weights")
    parser.add_argument("--subtitle", default="Pre-trained DreaMS SSL and embedding model checkpoints (Python 3.13 compatible)")
    parser.add_argument("--upload", action="store_true", help="create/upload the dataset with the kaggle CLI")
    parser.add_argument("--update", action="store_true", help="upload a new version instead of creating")
    parser.add_argument("--message", default="Updated DreaMS pretrained weights")
    args = parser.parse_args()

    package(
        local_dir=args.model_dir,
        staging=args.staging_dir,
        dataset_id=args.dataset_id,
        title=args.title,
        subtitle=args.subtitle,
    )

    if args.upload or args.update:
        run_kaggle(args.staging_dir, update=args.update, message=args.message)
        print(f"\n[+] Dataset '{args.dataset_id}' is available.")


if __name__ == "__main__":
    main()