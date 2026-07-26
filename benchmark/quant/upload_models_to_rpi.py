#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent


OLLAMA_URL = os.environ.get("QUANT_OLLAMA_URL", "http://mill-56-rpi.local:11434").rstrip("/")
OLLAMA_HOSTNAME = urlparse(OLLAMA_URL).hostname or "mill-56-rpi.local"
MODELS_DIR = Path(os.environ.get("QUANT_MODELS_DIR", str(ROOT / "models")))
RPI_SSH = os.environ.get("QUANT_RPI_SSH", f"pi@{OLLAMA_HOSTNAME}")
REMOTE_MODELS_DIR = os.environ.get("QUANT_REMOTE_MODELS_DIR", "/home/pi/quant/models").rstrip("/")

MODELS = ["gemma3-4b", "gemma4-e2b", "bielik-4.5b"]
QUANT_VARIANTS = ["Q8_0", "Q5_0", "Q5_K_M", "Q4_0", "Q4_K_M", "Q2_K"]


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload quantized GGUF files to the Raspberry Pi over SSH/rsync."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the SSH/rsync operations without uploading files.",
    )
    return parser.parse_args()


def run_checked(command: list[str], *, dry_run: bool = False) -> None:
    if dry_run:
        log("[upload] DRY-RUN " + " ".join(command))
        return
    subprocess.run(command, check=True)


def local_model_path(model_id: str, variant: str) -> Path:
    return MODELS_DIR / model_id / f"{model_id}-{variant}.gguf"


def remote_model_path(model_id: str, variant: str) -> str:
    return f"{REMOTE_MODELS_DIR}/{model_id}/{model_id}-{variant}.gguf"


def main() -> None:
    args = parse_args()

    missing: list[Path] = []
    for model_id in MODELS:
        for variant in QUANT_VARIANTS:
            path = local_model_path(model_id, variant)
            if not path.exists():
                missing.append(path)

    if missing:
        log("[upload] ERROR: missing local GGUF files:")
        for path in missing:
            log(f"  - {path}")
        raise SystemExit(1)

    log(f"[upload] target={RPI_SSH}:{REMOTE_MODELS_DIR}")
    run_checked(["ssh", RPI_SSH, f"mkdir -p {REMOTE_MODELS_DIR}"], dry_run=args.dry_run)

    for model_id in MODELS:
        remote_dir = f"{REMOTE_MODELS_DIR}/{model_id}"
        log(f"[upload] mkdir {RPI_SSH}:{remote_dir}")
        run_checked(["ssh", RPI_SSH, f"mkdir -p {remote_dir}"], dry_run=args.dry_run)
        for variant in QUANT_VARIANTS:
            src = local_model_path(model_id, variant)
            dst = remote_model_path(model_id, variant)
            log(f"[upload] {src} -> {RPI_SSH}:{dst}")
            run_checked(
                ["rsync", "-avh", "--partial", "--progress", str(src), f"{RPI_SSH}:{dst}"],
                dry_run=args.dry_run,
            )

    log("[upload] verifying remote files...")
    for model_id in MODELS:
        for variant in QUANT_VARIANTS:
            dst = remote_model_path(model_id, variant)
            run_checked(["ssh", RPI_SSH, f"test -s '{dst}'"], dry_run=args.dry_run)
            log(f"[upload] OK {dst}")

    if args.dry_run:
        log("[upload] dry-run done")
    else:
        log("[upload] done")


if __name__ == "__main__":
    main()
