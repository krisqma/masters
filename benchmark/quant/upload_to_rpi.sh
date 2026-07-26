#!/usr/bin/env bash
set -euo pipefail

export QUANT_RPI_SSH="${QUANT_RPI_SSH:-krisqma@192.168.0.77}"
export QUANT_REMOTE_MODELS_DIR="${QUANT_REMOTE_MODELS_DIR:-/home/krisqma/quant/models}"

python3 upload_models_to_rpi.py "$@"
