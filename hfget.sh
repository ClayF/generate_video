#!/bin/bash
# hfget <url> <dest-file>
#
# Fastest available download for a model file, used by the Dockerfile and
# entrypoint.sh. Hugging Face URLs go through the hub client with the Xet /
# hf_transfer parallel backends (many connections, saturates the worker's
# link — typically 5-10x wget on RunPod). Anything else, or a hub failure,
# falls back to aria2c (16 connections), then wget.
#
#   HF_TOKEN      picked up automatically by the hub client for gated repos
#   HFGET_DRY=1   print what would be done and exit
set -o pipefail
url="$1"; dest="$2"
if [ -z "$url" ] || [ -z "$dest" ]; then echo "usage: hfget <url> <dest-file>" >&2; exit 2; fi
mkdir -p "$(dirname "$dest")"

hub_download() {
    # https://huggingface.co/<owner>/<repo>/resolve/<rev>/<path> -> hub client
    [[ "$url" =~ ^https://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/([^?]+) ]] || return 1
    local repo="${BASH_REMATCH[1]}" rev="${BASH_REMATCH[2]}" file="${BASH_REMATCH[3]}"
    file="$(python3 -c 'import sys,urllib.parse; print(urllib.parse.unquote(sys.argv[1]))' "$file")"
    # stable staging dir next to the destination so an interrupted download resumes
    local stage="${dest}.hfget"
    if [ "$HFGET_DRY" = "1" ]; then echo "hub: repo=$repo rev=$rev file=$file -> $dest"; return 0; fi
    mkdir -p "$stage"
    # HF_XET_CACHE inside the staging dir: the Xet chunk cache would otherwise
    # persist under ~/.cache and bloat the image layer by the size of the model
    HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1 HF_XET_CACHE="$stage/.xet" HF_HUB_DISABLE_PROGRESS_BARS=1 \
    python3 - "$repo" "$rev" "$file" "$stage" <<'PY' || return 1
import sys
from huggingface_hub import hf_hub_download
repo, rev, file, stage = sys.argv[1:]
p = hf_hub_download(repo_id=repo, filename=file, revision=rev, local_dir=stage)
print(p)
PY
    mv -f "$stage/$file" "$dest" || return 1
    rm -rf "$stage"
}

fast_http() {
    if [ "$HFGET_DRY" = "1" ]; then echo "http: $url -> $dest"; return 0; fi
    if command -v aria2c >/dev/null 2>&1; then
        aria2c -x16 -s16 -k1M -c --file-allocation=none --console-log-level=warn --summary-interval=0 \
            ${HF_TOKEN:+--header="Authorization: Bearer $HF_TOKEN"} \
            -d "$(dirname "$dest")" -o "$(basename "$dest")" "$url" && return 0
        echo "aria2c failed, trying wget" >&2
    fi
    wget -nv -c --tries=5 ${HF_TOKEN:+--header="Authorization: Bearer $HF_TOKEN"} "$url" -O "$dest"
}

start=$(date +%s)
if hub_download || { echo "hub download failed for $url, falling back to direct HTTP" >&2; fast_http; }; then
    [ "$HFGET_DRY" = "1" ] || echo "hfget: $(basename "$dest") $(du -h "$dest" | cut -f1) in $(( $(date +%s) - start ))s"
    exit 0
fi
echo "hfget: FAILED $url" >&2
exit 1
