#!/usr/bin/env python3
"""One-cell setup for DreaMS in a Kaggle notebook using the bundled dataset.

The ``smeeee/dreams-pretrained`` dataset contains BOTH the DreaMS repo (as
``DreaMS/``) and the pre-trained checkpoints (``embedding_model.ckpt``,
``ssl_model.ckpt``).  This script turns that single input into a working
installation inside the notebook:

    1. copy the repo from the (read-only) dataset to /kaggle/working
    2. ``pip install -e`` it (Kaggle already has torch; the rest is small)
    3. place the checkpoints into dreams/models/pretrained/
    4. smoke-test by computing embeddings for the README example

Usage (after adding ``smeeee/dreams-pretrained`` as a notebook input)::

    %run -i /kaggle/input/dreams-pretrained/DreaMS/scripts/kaggle_setup.py

or just paste the contents of this file into a notebook cell and run it.

For local testing (not on Kaggle)::

    python scripts/kaggle_setup.py \
        --input-dir /path/to/dataset_staging \
        --work-dir /tmp/dreams_work \
        --python /path/to/venv/bin/python
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

INPUT_DEFAULT = Path("/kaggle/input/dreams-pretrained")
WORK_DEFAULT = Path("/kaggle/working/DreaMS")

#: Checkpoint files stored at the top level of the dataset.
CKPTS = ("embedding_model.ckpt", "ssl_model.ckpt")


def _fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"\n[!] {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DEFAULT,
                        help="dataset staging/input dir (default: %(default)s)")
    parser.add_argument("--work-dir", type=Path, default=WORK_DEFAULT,
                        help="writable target for the repo copy (default: %(default)s)")
    parser.add_argument("--python", dest="python_bin", type=Path, default=Path(sys.executable),
                        help="python interpreter to install/import with (default: %(default)s)")
    parser.add_argument("--skip-install", action="store_true",
                        help="skip pip install (e.g. deps already installed)")
    args = parser.parse_args()

    inp = args.input_dir
    repo_src = inp / "DreaMS"
    repo_tar = inp / "DreaMS.tar"
    if repo_src.exists():
        pass
    elif repo_tar.exists():
        repo_src = None  # handled below
    else:
        _fail(
            f"repo not found at {inp / 'DreaMS'} (or {inp / 'DreaMS.tar'}). "
            "Add the 'smeeee/dreams-pretrained' dataset (or a newer version of it) "
            "as an input to this notebook."
        )

    # 1. Get the repo out of the read-only dataset dir.
    print(f"[1/4] preparing repo -> {args.work_dir}")
    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    args.work_dir.parent.mkdir(parents=True, exist_ok=True)
    if repo_src is not None:
        shutil.copytree(repo_src, args.work_dir)
    else:
        print(f"      extracting {repo_tar}")
        with tarfile.open(repo_tar, "r") as tar:
            tar.extractall(args.work_dir)
        # The tar stores the repo contents at its root (or nested under
        # DreaMS/), so normalize to the dir that actually contains setup.py.
        if (args.work_dir / "DreaMS").exists() and not (args.work_dir / "setup.py").exists():
            args.work_dir = args.work_dir / "DreaMS"
    if not (args.work_dir / "setup.py").exists():
        _fail(f"copied repo looks incomplete (no setup.py in {args.work_dir})")

    # 2. Install the package (editable) with its dependencies.
    if args.skip_install:
        print("[2/4] skipping pip install (--skip-install)")
    else:
        print(f"[2/4] pip install -e {args.work_dir}  (this can take a few minutes)")
        subprocess.run(
            [str(args.python_bin), "-m", "pip", "install", "-e", str(args.work_dir)],
            check=True,
        )

    # 3. Put the checkpoints where the package expects them.
    pretrained = args.work_dir / "dreams" / "models" / "pretrained"
    pretrained.mkdir(parents=True, exist_ok=True)
    print("[3/4] placing checkpoints:")
    for ckpt in CKPTS:
        src = inp / ckpt
        dst = pretrained / ckpt
        if not src.exists():
            _fail(f"checkpoint {ckpt} missing from dataset {inp}")
        if not dst.exists():
            shutil.copy2(src, dst)
        print(f"      {ckpt} ({dst.stat().st_size / 1e9:.2f} GB)")

    # 4. Smoke-test: run the README example.
    print("[4/4] smoke test - DreaMS embeddings for the README example:")
    example = args.work_dir / "data" / "examples" / "example_5_spectra.mgf"
    code = (
        "from dreams.api import dreams_embeddings\n"
        f"embs = dreams_embeddings({str(example)!r})\n"
        "print('SMOKE TEST OK, embeddings shape:', embs.shape, embs.dtype)\n"
    )
    proc = subprocess.run([str(args.python_bin), "-c", code])
    if proc.returncode != 0:
        print("\n[!] Smoke test failed (see traceback above); the setup may still be usable.", file=sys.stderr)
    else:
        print(f"\n[+] Setup complete. Repo: {args.work_dir}")
        print("    Weights: " + ", ".join(str(pretrained / c) for c in CKPTS))
        print('    Usage:   from dreams.api import dreams_embeddings')


if __name__ == "__main__":
    main()