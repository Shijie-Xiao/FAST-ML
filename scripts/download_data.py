#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fetch the large input files that are distributed outside git.

The repository carries the model weights, the drag field, the normalisation
statistics, the ensemble ODE output and the Google FNV3 reference -- everything
needed to redraw a figure. Two categories are too large for git and are hosted
separately:

``--single-track``
    The 2024 North Atlantic ERA5 storm archive (about 2.1 GB). Needed only to
    run the CNN; the published NetCDF output in ``results/`` can be replotted
    without it.

``--ensemble``
    The per-member ``chi``/``S`` files for the two case studies (about 53 GB
    combined is *not* required -- these are 18 MB and 35 MB). Needed only for
    the ventilation panel of the ensemble figures.

Locations are read from ``data/manifest.json``. Each entry records an expected
size and SHA-256 so a truncated or corrupted download is caught rather than
silently producing wrong figures.

Usage::

    python scripts/download_data.py --all
    python scripts/download_data.py --ensemble
    python scripts/download_data.py --verify        # check what is present
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastml.config import DATA_DIR, REPO_ROOT  # noqa: E402

MANIFEST_PATH = DATA_DIR / "manifest.json"
_CHUNK = 1 << 20


def _human(n):
    """Format a byte count."""
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def check_entry(entry, verbose=True):
    """Report whether a manifest entry is present and intact.

    Returns one of ``'ok'``, ``'missing'``, ``'wrong_size'`` or ``'wrong_hash'``.
    """
    target = REPO_ROOT / entry["target"]
    marker = REPO_ROOT / entry["marker"] if entry.get("marker") else target

    if not marker.exists():
        if verbose:
            print(f"  missing    {entry['name']}  ({_human(entry.get('size', 0))})")
        return "missing"

    if entry.get("archive"):
        # Archives are validated by their extracted marker only; the tar itself
        # is deleted after a successful extraction.
        if verbose:
            print(f"  ok         {entry['name']}  (extracted)")
        return "ok"

    actual_size = target.stat().st_size
    if entry.get("size") and actual_size != entry["size"]:
        if verbose:
            print(f"  wrong size {entry['name']}  "
                  f"({_human(actual_size)} vs {_human(entry['size'])})")
        return "wrong_size"

    if entry.get("sha256"):
        if _sha256(target) != entry["sha256"]:
            if verbose:
                print(f"  CORRUPT    {entry['name']}  (SHA-256 mismatch)")
            return "wrong_hash"

    if verbose:
        print(f"  ok         {entry['name']}  ({_human(actual_size)})")
    return "ok"


def _download(entry, target):
    """Download one entry, preferring gdown for Google Drive URLs."""
    url = entry.get("url")
    drive_id = entry.get("drive_id")
    if not url and not drive_id:
        raise RuntimeError(
            f"{entry['name']}: no 'url' or 'drive_id' in the manifest.\n"
            f"See the download link in README.md and fetch it manually to {target}."
        )

    target.parent.mkdir(parents=True, exist_ok=True)

    if drive_id:
        try:
            import gdown
        except ImportError:
            raise RuntimeError(
                f"{entry['name']} is hosted on Google Drive; install gdown "
                f"(`pip install gdown`) or download it manually to {target}."
            )
        gdown.download(id=drive_id, output=str(target), quiet=False)
        return

    import urllib.request
    print(f"  fetching {url}")
    with urllib.request.urlopen(url) as response, open(target, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            block = response.read(_CHUNK)
            if not block:
                break
            out.write(block)
            done += len(block)
            if total:
                print(f"\r    {_human(done)} / {_human(total)}"
                      f"  ({100 * done / total:.0f}%)", end="", flush=True)
        print()


def fetch_entry(entry, force=False):
    """Download, verify and (if an archive) extract one manifest entry."""
    target = REPO_ROOT / entry["target"]
    status = check_entry(entry, verbose=False)
    if status == "ok" and not force:
        print(f"  ok         {entry['name']} (already present)")
        return True

    print(f"  fetching   {entry['name']}  ({_human(entry.get('size', 0))})")
    try:
        _download(entry, target)
    except Exception as exc:
        print(f"  FAILED     {entry['name']}: {exc}")
        return False

    if entry.get("sha256") and target.exists():
        if _sha256(target) != entry["sha256"]:
            print(f"  FAILED     {entry['name']}: SHA-256 mismatch after download")
            return False

    if entry.get("archive"):
        dest = REPO_ROOT / entry.get("extract_to", "data")
        print(f"  extracting {entry['name']} -> {dest}")
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(target) as tar:
            tar.extractall(dest, filter="data")
        target.unlink()
        print(f"  removed    {target.name} (archive no longer needed)")

    print(f"  done       {entry['name']}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Fetch the large FAST_ML inputs hosted outside git",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--all", action="store_true", help="Fetch every group")
    parser.add_argument("--single-track", action="store_true",
                        help="2024 North Atlantic ERA5 storm archive")
    parser.add_argument("--ensemble", action="store_true",
                        help="Per-member chi/S files for the ventilation panels")
    parser.add_argument("--verify", action="store_true",
                        help="Report what is present, download nothing")
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    args = parser.parse_args()

    manifest = load_manifest()
    entries = manifest["files"]

    if args.verify:
        print("Checking local data:\n")
        statuses = [check_entry(e) for e in entries]
        missing = sum(1 for s in statuses if s != "ok")
        print(f"\n{len(statuses) - missing}/{len(statuses)} entries present and intact.")
        if missing:
            print("Fetch the rest with: python scripts/download_data.py --all")
        return 0 if missing == 0 else 1

    groups = set()
    if args.all:
        groups = {e["group"] for e in entries}
    if args.single_track:
        groups.add("single_track")
    if args.ensemble:
        groups.add("ensemble")
    if not groups:
        parser.print_help()
        print("\nNothing selected. Use --all, --single-track, --ensemble or --verify.")
        return 1

    selected = [e for e in entries if e["group"] in groups]
    print(f"Fetching {len(selected)} entries "
          f"(~{_human(sum(e.get('size', 0) for e in selected))})\n")

    ok = sum(fetch_entry(e, force=args.force) for e in selected)
    print(f"\n{ok}/{len(selected)} entries ready.")
    if ok < len(selected):
        print(f"See the manual download links in {REPO_ROOT / 'README.md'}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
