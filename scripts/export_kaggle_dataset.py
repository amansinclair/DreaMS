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

By default the dataset also contains a clean copy of the DreaMS repo
(git archive of HEAD, ~70 MB) as the ``DreaMS/`` folder, so a Kaggle
notebook needs only this one dataset to get both the code and the weights.

Example: setting up a Kaggle notebook
-------------------------------------
Add ``smeeee/dreams-pretrained`` as an input to the notebook, then run the
ready-made setup script from the dataset::

    %run -i /kaggle/input/dreams-pretrained/DreaMS/scripts/kaggle_setup.py

or use the small helper below::

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


def package_repo(staging: Path, repo_root: Path, dest_dir: str = "DreaMS") -> None:
    """Extract a clean copy of the tracked repo into ``staging/dest_dir``.

    Uses ``git archive HEAD`` so only tracked files are included (no
    checkpoints, venvs, staging dirs, etc.).
    """
    if not (repo_root / ".git").exists():
        raise FileNotFoundError(f"{repo_root} is not a git repository; cannot package the repo")

    dest = staging / dest_dir
    dest.mkdir(parents=True, exist_ok=True)

    archive = subprocess.run(
        ["git", "-C", str(repo_root), "archive", "--format=tar", "HEAD"],
        capture_output=True,
    )
    if archive.returncode != 0:
        raise RuntimeError(f"git archive failed: {archive.stderr.decode().strip()}")

    extract = subprocess.run(["tar", "-x", "-C", str(dest)], input=archive.stdout)
    if extract.returncode != 0:
        raise RuntimeError(f"tar extraction failed: {extract.stderr.decode().strip()}")

    n_files = sum(1 for _ in dest.rglob("*"))
    print(f"[*] packaged repo -> {dest} ({n_files} files)")


def write_metadata(staging: Path, dataset_id: str, title: str, subtitle: str) -> None:
    metadata = {
        "id": dataset_id,
        "title": title,
        "subtitle": subtitle,
        "licenses": [{"name": "MIT"}],
    }
    with (staging / "dataset-metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[*] wrote {staging / 'dataset-metadata.json'}")


def package(
    local_dir: Path,
    staging: Path,
    dataset_id: str,
    title: str,
    subtitle: str,
    include_repo: bool = True,
    repo_root: Path | None = None,
) -> None:
    """Copy the checkpoints, (optionally) the repo, and write metadata."""
    ensure_weights(local_dir)

    if include_repo:
        repo_root = repo_root or Path(__file__).resolve().parent.parent
        package_repo(staging, repo_root)


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


def find_kaggle_cli() -> str | None:
    """Locate the ``kaggle`` CLI: on PATH, or in the current venv's bin dir."""
    on_path = shutil.which("kaggle")
    if on_path:
        return on_path
    venv_bin = Path(sys.executable).parent / "kaggle"
    return str(venv_bin) if venv_bin.exists() else None


def run_kaggle(staging: Path, update: bool, message: str) -> None:
    """Create a new dataset or upload a new version with the ``kaggle`` CLI."""
    kaggle_exe = find_kaggle_cli()
    if kaggle_exe is None:
        print(
            "\n[!] 'kaggle' CLI not found. Install it with `pip install kaggle` and\n"
            "    add your credentials to ~/.kaggle/kaggle.json, then re-run with --upload.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    if update:
        cmd = [kaggle_exe, "datasets", "version", "-p", str(staging), "-m", message]
    else:
        cmd = [kaggle_exe, "datasets", "create", "-p", str(staging)]

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
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="local git checkout of DreaMS to bundle (default: the repo containing this script)",
    )
    parser.add_argument(
        "--no-include-repo",
        action="store_true",
        help="only package the checkpoints, not the repo source",
    )
    parser.add_argument(
        "--dataset-id",
        default=None,
        help="{username}/slug of the Kaggle dataset (default: auto-detected from kaggle credentials)",
    )
    parser.add_argument("--title", default="DreaMS Pre-trained Model Weights")
    parser.add_argument("--subtitle", default="Pre-trained DreaMS SSL and embedding model checkpoints (Python 3.13 compatible)")
    parser.add_argument("--upload", action="store_true", help="create/upload the dataset with the kaggle CLI")
    parser.add_argument("--update", action="store_true", help="upload a new version instead of creating")
    parser.add_argument("--message", default="Updated DreaMS pretrained weights")
    args = parser.parse_args()

    if args.dataset_id is None:
        # Auto-detect the Kaggle username from the authenticated kaggle CLI.
        try:
            from kaggle import KaggleApi

            api = KaggleApi()
            api.authenticate()
            args.dataset_id = f"{api.config_values[api.CONFIG_NAME_USER]}/dreams-pretrained"
            print(f"[*] detected dataset id: {args.dataset_id}")
        except Exception as exc:  # pragma: no cover - only when kaggle CLI is missing
            print(f"[!] could not auto-detect Kaggle username ({exc}); pass --dataset-id explicitly.", file=sys.stderr)
            sys.exit(1)

    package(
        local_dir=args.model_dir,
        staging=args.staging_dir,
        dataset_id=args.dataset_id,
        title=args.title,
        subtitle=args.subtitle,
        include_repo=not args.no_include_repo,
        repo_root=args.repo_root,
    )

    if args.upload or args.update:
        run_kaggle(args.staging_dir, update=args.update, message=args.message)
        print(f"\n[+] Dataset '{args.dataset_id}' is available.")


if __name__ == "__main__":
    main()