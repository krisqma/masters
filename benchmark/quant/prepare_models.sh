#!/usr/bin/env bash
set -euo pipefail

# Keep Hugging Face downloads on the classic HTTP/LFS path. hf-xet can stall on
# large GGUF files and is not needed here.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

MODELS_DIR="${QUANT_MODELS_DIR:-./models}"
QUANT_TYPES=("Q8_0" "Q5_0" "Q5_K_M" "Q4_0" "Q4_K_M" "Q2_K")
COMMAND="${1:-all}"
if [[ $# -gt 0 ]]; then
  shift
fi
SELECTED_MODELS_COUNT="$#"
SELECTED_MODELS=("$@")
HF_CLI_BIN=""
HF_CLI_USE_PYTHON_MODULE=0

log() {
  printf '[prepare] %s\n' "$*"
}

human_size() {
  if [[ -f "$1" ]]; then
    du -h "$1" | awk '{print $1}'
  else
    printf '-'
  fi
}

file_size_bytes() {
  if stat -f '%z' "$1" >/dev/null 2>&1; then
    stat -f '%z' "$1"
  else
    stat -c '%s' "$1"
  fi
}

usage() {
  cat <<'USAGE'
Usage:
  ./prepare_models.sh [all|download|quantize|summary|list] [model-id...]

Commands:
  all       Download missing source GGUFs, then create missing quantized files.
  download  Download only source GGUFs. Use this before going offline.
  quantize  Quantize existing source GGUFs only. No internet needed.
  summary   Print source and quantized GGUF files.
  list      Print configured model ids.

Model ids:
  gemma3-4b
  gemma4-e2b
  bielik-4.5b

Examples:
  ./prepare_models.sh download
  ./prepare_models.sh download bielik-4.5b
  ./prepare_models.sh quantize
  QUANT_FORCE=1 ./prepare_models.sh quantize gemma4-e2b
USAGE
}

is_known_model() {
  case "$1" in
    gemma3-4b|gemma4-e2b|bielik-4.5b) return 0 ;;
    *) return 1 ;;
  esac
}

model_selected() {
  local model_id="$1"

  if [[ "$SELECTED_MODELS_COUNT" -eq 0 ]]; then
    return 0
  fi

  local selected
  for selected in "${SELECTED_MODELS[@]}"; do
    if [[ "$selected" == "$model_id" ]]; then
      return 0
    fi
  done

  return 1
}

validate_command() {
  case "$COMMAND" in
    all|download|quantize|summary|list) ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      printf 'ERROR: unknown command: %s\n' "$COMMAND" >&2
      exit 1
      ;;
  esac
}

validate_selected_models() {
  if [[ "$SELECTED_MODELS_COUNT" -eq 0 ]]; then
    return
  fi

  local selected
  for selected in "${SELECTED_MODELS[@]}"; do
    if ! is_known_model "$selected"; then
      usage >&2
      printf 'ERROR: unknown model id: %s\n' "$selected" >&2
      exit 1
    fi
  done
}

require_llama_quantize() {
  if command -v llama-quantize >/dev/null 2>&1; then
    log "llama-quantize found: $(command -v llama-quantize)"
    return
  fi

  log "llama-quantize not found; installing llama.cpp via Homebrew..."
  brew install llama.cpp

  if ! command -v llama-quantize >/dev/null 2>&1; then
    printf 'ERROR: llama-quantize is still unavailable after brew install llama.cpp\n' >&2
    exit 1
  fi
}

require_huggingface_cli() {
  if command -v huggingface-cli >/dev/null 2>&1; then
    HF_CLI_BIN="$(command -v huggingface-cli)"
    HF_CLI_USE_PYTHON_MODULE=0
    log "huggingface-cli found: $HF_CLI_BIN"
    return
  fi

  local user_base
  local user_cli
  user_base="$(python3 -m site --user-base 2>/dev/null || true)"
  user_cli="$user_base/bin/huggingface-cli"
  if [[ -n "$user_base" && -x "$user_cli" ]]; then
    HF_CLI_BIN="$user_cli"
    HF_CLI_USE_PYTHON_MODULE=0
    log "huggingface-cli found in Python user base: $HF_CLI_BIN"
    return
  fi

  if python3 -m huggingface_hub.commands.huggingface_cli --help >/dev/null 2>&1; then
    HF_CLI_USE_PYTHON_MODULE=1
    log "huggingface-cli module found: python3 -m huggingface_hub.commands.huggingface_cli"
    return
  fi

  log "huggingface-cli not found; installing huggingface_hub..."
  if ! python3 -m pip install --user huggingface_hub; then
    log "pip install --user failed; retrying with --break-system-packages..."
    python3 -m pip install --user huggingface_hub --break-system-packages
  fi

  user_base="$(python3 -m site --user-base 2>/dev/null || true)"
  user_cli="$user_base/bin/huggingface-cli"
  if [[ -n "$user_base" && -x "$user_cli" ]]; then
    HF_CLI_BIN="$user_cli"
    HF_CLI_USE_PYTHON_MODULE=0
    log "huggingface-cli installed in Python user base: $HF_CLI_BIN"
    return
  fi

  if python3 -m huggingface_hub.commands.huggingface_cli --help >/dev/null 2>&1; then
    HF_CLI_USE_PYTHON_MODULE=1
    log "huggingface-cli module installed: python3 -m huggingface_hub.commands.huggingface_cli"
    return
  fi

  if [[ "$HF_CLI_USE_PYTHON_MODULE" != "1" && -z "$HF_CLI_BIN" ]]; then
    printf 'ERROR: huggingface-cli is still unavailable after installing huggingface_hub\n' >&2
    exit 1
  fi
}

hf_cli() {
  if [[ "$HF_CLI_USE_PYTHON_MODULE" == "1" ]]; then
    python3 -m huggingface_hub.commands.huggingface_cli "$@"
  else
    "$HF_CLI_BIN" "$@"
  fi
}

hf_login_if_needed() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    log "HF_TOKEN detected; logging in to Hugging Face..."
    hf_cli login --token "$HF_TOKEN"
  else
    log "HF_TOKEN not set; continuing without explicit login."
  fi
}

download_if_missing() {
  local repo="$1"
  local filename="$2"
  local target_dir="$3"
  local target_path="$target_dir/$filename"

  mkdir -p "$target_dir"

  if [[ -f "$target_path" ]]; then
    log "source exists, skipping download: $target_path ($(human_size "$target_path"))"
    return
  fi

  log "downloading $repo / $filename -> $target_dir"
  hf_cli download "$repo" "$filename" \
    --local-dir "$target_dir" \
    --local-dir-use-symlinks False

  if [[ ! -f "$target_path" ]]; then
    printf 'ERROR: expected source file was not downloaded: %s\n' "$target_path" >&2
    exit 1
  fi

  log "downloaded: $target_path ($(human_size "$target_path"))"
}

quantize_if_missing() {
  local src="$1"
  local dst="$2"
  local quant_type="$3"
  local tmp_dst="$dst.incomplete"

  if [[ -f "$dst" ]]; then
    local src_bytes
    local dst_bytes
    local min_expected_bytes
    src_bytes="$(file_size_bytes "$src")"
    dst_bytes="$(file_size_bytes "$dst")"
    min_expected_bytes=$((src_bytes / 10))
    if [[ "$min_expected_bytes" -lt 1 ]]; then
      min_expected_bytes=1
    fi

    if [[ "${QUANT_FORCE:-0}" != "1" && "$dst_bytes" -ge "$min_expected_bytes" ]]; then
      log "quantized file exists, skipping: $dst ($(human_size "$dst"))"
      return
    fi

    if [[ "${QUANT_FORCE:-0}" == "1" ]]; then
      log "QUANT_FORCE=1; rebuilding existing quantized file: $dst ($(human_size "$dst"))"
    else
      log "existing quantized file looks partial; rebuilding: $dst ($(human_size "$dst"))"
    fi
  fi

  if [[ -f "$tmp_dst" ]]; then
    log "removing stale partial quantized file: $tmp_dst ($(human_size "$tmp_dst"))"
    rm -f "$tmp_dst"
  fi

  if [[ ! -f "$src" ]]; then
    printf 'ERROR: missing source file for quantization: %s\n' "$src" >&2
    exit 1
  fi

  log "quantizing: $src -> $dst ($quant_type)"
  local start_ts
  local end_ts
  start_ts="$(date +%s)"
  llama-quantize "$src" "$tmp_dst" "$quant_type"
  mv "$tmp_dst" "$dst"
  end_ts="$(date +%s)"
  log "done: $dst size=$(human_size "$dst") time=$((end_ts - start_ts))s"
}

prepare_model() {
  local model_id="$1"
  local repo="$2"
  local src_file="$3"
  local model_dir="$MODELS_DIR/$model_id"
  local src_path="$model_dir/$src_file"

  if ! model_selected "$model_id"; then
    return
  fi

  log "=== $model_id ==="

  if [[ "$COMMAND" == "all" || "$COMMAND" == "download" ]]; then
    download_if_missing "$repo" "$src_file" "$model_dir"
  fi

  if [[ "$COMMAND" == "all" || "$COMMAND" == "quantize" ]]; then
    if [[ ! -f "$src_path" ]]; then
      printf 'ERROR: missing source file: %s\n' "$src_path" >&2
      printf 'Run first: ./prepare_models.sh download %s\n' "$model_id" >&2
      exit 1
    fi

    for quant_type in "${QUANT_TYPES[@]}"; do
      quantize_if_missing "$src_path" "$model_dir/$model_id-$quant_type.gguf" "$quant_type"
    done
  fi
}

print_summary() {
  log "generated/source GGUF files:"
  printf '%-22s %-18s %s\n' "MODEL" "SIZE" "FILE"
  find "$MODELS_DIR" -type f -name '*.gguf' -print | sort | while IFS= read -r file; do
    model="$(basename "$(dirname "$file")")"
    printf '%-22s %-18s %s\n' "$model" "$(human_size "$file")" "$file"
  done
}

print_models() {
  printf '%s\n' \
    "gemma3-4b" \
    "gemma4-e2b" \
    "bielik-4.5b"
}

prepare_configured_models() {
  prepare_model "gemma3-4b" "bartowski/google_gemma-3-4b-it-GGUF" "google_gemma-3-4b-it-bf16.gguf"
  prepare_model "gemma4-e2b" "bartowski/google_gemma-4-E2B-it-GGUF" "google_gemma-4-E2B-it-bf16.gguf"
  prepare_model "bielik-4.5b" "speakleash/Bielik-4.5B-v3.0-Instruct-GGUF" "Bielik-4.5B-v3.0-Instruct-fp16.gguf"
}

main() {
  validate_command
  validate_selected_models

  if [[ "$COMMAND" == "list" ]]; then
    print_models
    exit 0
  fi

  if [[ "$COMMAND" == "summary" ]]; then
    print_summary
    exit 0
  fi

  if [[ "$COMMAND" == "all" || "$COMMAND" == "quantize" ]]; then
    require_llama_quantize
  fi

  if [[ "$COMMAND" == "all" || "$COMMAND" == "download" ]]; then
    require_huggingface_cli
    hf_login_if_needed
  fi

  prepare_configured_models

  print_summary
}

main "$@"
