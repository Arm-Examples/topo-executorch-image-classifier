#!/usr/bin/env bash
#
# Originally vendored from: https://gist.github.com/padeoe/697678ab8e528b85a2a7bddafea1fa4f
# Author: padeoe
# Retrieved: 2026-08-21
# Revision: dcc19a0
# License: none stated at time of vendoring
#
trap 'printf "\nInterrupted.\n"; exit 130' INT

human() {
    awk -v b="${1:-0}" 'BEGIN{u="B KB MB GB TB PB";n=split(u,a," ");i=1;
        while(b>=1000&&i<n){b/=1000;i++} printf (i==1?"%d%s":"%.2f%s"),b,a[i]}'
}

display_help() {
    cat << EOF
Usage:
  hfd <REPO_ID> [--include pattern ...] [--exclude pattern ...] [-x threads] [-j jobs] [--local-dir path] [--revision rev]

Description:
  Downloads a model from Hugging Face using aria2c.

Options:
  --include       File patterns to include.
  --exclude       File patterns to exclude.
  -x              Connections per server (default: 4, maximum: 10).
  -j              Concurrent downloads (default: 5, maximum: 10).
  --local-dir     Destination directory (default: the repository name).
  --revision      Model revision (default: main).

Examples:
  hfd gpt2
  hfd bigscience/bloom-560m --exclude '*.safetensors'
EOF
    exit 1
}

[[ -z "$1" || "$1" =~ ^-h || "$1" =~ ^--help ]] && display_help

REPO_ID=$1
shift

THREADS=4
CONCURRENT=5
HF_ENDPOINT=${HF_ENDPOINT:-"https://huggingface.co"}
INCLUDE_PATTERNS=()
EXCLUDE_PATTERNS=()
REVISION="main"

validate_number() {
    [[ "$2" =~ ^[1-9][0-9]*$ && "$2" -le "$3" ]] || {
        printf "[Error] %s must be 1-%s.\n" "$1" "$3"
        exit 1
    }
}

require_value() {
    (($# >= 2)) || {
        printf "[Error] %s requires a value.\n" "$1" >&2
        exit 1
    }
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --include)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do INCLUDE_PATTERNS+=("$1"); shift; done
            ;;
        --exclude)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do EXCLUDE_PATTERNS+=("$1"); shift; done
            ;;
        -x) require_value "$@"; validate_number "threads (-x)" "$2" 10; THREADS="$2"; shift 2 ;;
        -j) require_value "$@"; validate_number "concurrent downloads (-j)" "$2" 10; CONCURRENT="$2"; shift 2 ;;
        --local-dir) require_value "$@"; LOCAL_DIR="$2"; shift 2 ;;
        --revision) require_value "$@"; REVISION="$2"; shift 2 ;;
        *) display_help ;;
    esac
done

for command in curl jq aria2c; do
    command -v "$command" &>/dev/null || {
        printf "[Error] %s is not installed.\n" "$command"
        exit 1
    }
done

LOCAL_DIR="${LOCAL_DIR:-${REPO_ID#*/}}"
HF_ENDPOINT=${HF_ENDPOINT%/}
METADATA_PATH="models/$REPO_ID"
[[ "$REVISION" != "main" ]] && METADATA_PATH="$METADATA_PATH/revision/$REVISION"
API_URL="$HF_ENDPOINT/api/$METADATA_PATH?blobs=true"

mkdir -p "$LOCAL_DIR/.hfd"
METADATA_FILE="$LOCAL_DIR/.hfd/repo_metadata.json"
MANIFEST_FILE="$LOCAL_DIR/.hfd/manifest"

CURL_AUTH=()
[[ -n "${HF_TOKEN:-}" ]] && CURL_AUTH=(-H "Authorization: Bearer $HF_TOKEN")

printf "%s (%s)\n" "$REPO_ID" "$REVISION"
printf "Fetching metadata...\n"
status_code=$(curl --location-trusted -sS -w "%{http_code}" -o "$METADATA_FILE" \
    "${CURL_AUTH[@]}" "$API_URL")

if [[ "$status_code" != "200" ]]; then
    printf "[Error] Failed to fetch metadata from %s. HTTP status code: %s.\n" \
        "$API_URL" "$status_code" >&2
    cat "$METADATA_FILE" >&2
    exit 1
fi

if [[ "$(jq -r '.gated // false' "$METADATA_FILE")" != "false" && -z "${HF_TOKEN:-}" ]]; then
    printf "[Error] This model requires HF_TOKEN.\n"
    exit 1
fi

matches_any() {
    local path=$1 pattern
    shift
    for pattern in "$@"; do
        [[ "$path" == $pattern ]] && return 0
    done
    return 1
}

: > "$MANIFEST_FILE"
while IFS=$'\t' read -r size path; do
    [[ -z "$path" ]] && continue
    if ((${#INCLUDE_PATTERNS[@]})) && ! matches_any "$path" "${INCLUDE_PATTERNS[@]}"; then
        continue
    fi
    if ((${#EXCLUDE_PATTERNS[@]})) && matches_any "$path" "${EXCLUDE_PATTERNS[@]}"; then
        continue
    fi
    printf '%s\t%s\n' "${size:-0}" "$path" >> "$MANIFEST_FILE"
done < <(jq -r '(.siblings // [])[] | "\(.size // 0)\t\(.rfilename)"' "$METADATA_FILE")

TOTAL_FILES=$(wc -l < "$MANIFEST_FILE")
TOTAL_SIZE=$(awk -F'\t' '{total += $1} END {print total + 0}' "$MANIFEST_FILE")
FILE_NOUN="files"; ((TOTAL_FILES == 1)) && FILE_NOUN="file"

if ((TOTAL_FILES == 0)); then
    printf "No files matched the requested patterns.\n"
    rm -rf "$LOCAL_DIR/.hfd"
    exit 0
fi

printf "Downloading %d %s (%s) to %s.\n" \
    "$TOTAL_FILES" "$FILE_NOUN" "$(human "$TOTAL_SIZE")" "$LOCAL_DIR"

cd "$LOCAL_DIR" || exit 1
ARIA2_INPUT=".hfd/aria2_urls.txt"
: > "$ARIA2_INPUT"

while IFS=$'\t' read -r _ path; do
    dir=${path%/*}
    [[ "$dir" == "$path" ]] && dir="."
    printf '%s/%s/resolve/%s/%s\n dir=%s\n out=%s\n' \
        "$HF_ENDPOINT" "$REPO_ID" "$REVISION" "$path" "$dir" "${path##*/}" >> "$ARIA2_INPUT"
    [[ -n "${HF_TOKEN:-}" ]] && printf ' header=Authorization: Bearer %s\n' "$HF_TOKEN" >> "$ARIA2_INPUT"
    printf '\n' >> "$ARIA2_INPUT"
done < .hfd/manifest

aria2c --console-log-level=notice --summary-interval=5 --download-result=hide \
    --file-allocation=none --auto-file-renaming=false \
    -x "$THREADS" -j "$CONCURRENT" -s "$THREADS" -k 1M -i "$ARIA2_INPUT"
download_status=$?

if ((download_status != 0)); then
    printf "[Error] Download failed.\n"
    exit "$download_status"
fi

rm -rf .hfd
printf "Done. %d %s, %s in %s.\n" \
    "$TOTAL_FILES" "$FILE_NOUN" "$(human "$TOTAL_SIZE")" "$LOCAL_DIR"
