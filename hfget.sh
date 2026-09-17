#!/bin/bash
# hfget <url> <dest-file>
#
# Fastest available download for a model file, used by the Dockerfile and
# entrypoint.sh. Hugging Face URLs go through the hub client with the Xet /
# hf_transfer parallel backends (many connections, saturates the worker's
# link — typically 5-10x wget on RunPod). Anything else, or a hub failure,
# falls back to aria2c (16 connections), then wget. Every path downloads to
# a staging file and renames on completion, so $dest is either complete or
# absent — never 0 bytes or truncated.
#
#   HF_TOKEN      picked up automatically by the hub client for gated repos
#   HFGET_DRY=1   print what would be done and exit
set -o pipefail
url="$1"; dest="$2"
if [ -z "$url" ] || [ -z "$dest" ]; then echo "usage: hfget <url> <dest-file>" >&2; exit 2; fi
mkdir -p "$(dirname "$dest")"
# a 0-byte leftover from an earlier failed attempt is not a partial download
[ -e "$dest" ] && [ ! -s "$dest" ] && rm -f "$dest"

# Interpreter that actually has huggingface_hub (base images often have a
# system python3 next to the conda/venv `python` that pip installed into).
PY=""
for cand in python python3; do
    command -v "$cand" >/dev/null 2>&1 || continue
    if "$cand" -c 'import huggingface_hub' >/dev/null 2>&1; then PY="$cand"; break; fi
done

probe() {  # one line in the log that says whether this host can reach the server at all
    local host code
    host="$(printf '%s' "$url" | sed -E 's#^https?://([^/]+).*#\1#')"
    if code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 15 -I "https://$host/" 2>&1)"; then
        echo "hfget: probe https://$host/ -> HTTP $code"
    else
        echo "hfget: probe https://$host/ FAILED: $code (no DNS/egress from this worker?)"
    fi
}

hub_download() {
    # https://huggingface.co/<owner>/<repo>/resolve/<rev>/<path> -> hub client
    [[ "$url" =~ ^https://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/([^?]+) ]] || return 1
    local repo="${BASH_REMATCH[1]}" rev="${BASH_REMATCH[2]}" file="${BASH_REMATCH[3]}"
    file="$(python3 -c 'import sys,urllib.parse; print(urllib.parse.unquote(sys.argv[1]))' "$file" 2>/dev/null || printf '%s' "$file")"
    # stable staging dir next to the destination so an interrupted download resumes
    local stage="${dest}.hfget"
    if [ "$HFGET_DRY" = "1" ]; then echo "hub: repo=$repo rev=$rev file=$file -> $dest"; return 0; fi
    if [ -z "$PY" ]; then echo "hfget: no python with huggingface_hub found (tried python, python3); skipping hub client" >&2; return 1; fi
    mkdir -p "$stage"
    echo "hfget: hub client ($PY) repo=$repo file=$file"
    # HF_XET_CACHE inside the staging dir: the Xet chunk cache would otherwise
    # persist under ~/.cache and bloat the image layer by the size of the model
    HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1 HF_XET_CACHE="$stage/.xet" HF_HUB_DISABLE_PROGRESS_BARS=1 \
    "$PY" - "$repo" "$rev" "$file" "$stage" <<'PY' || return 1
import os, sys
from huggingface_hub import hf_hub_download
repo, rev, file, stage = sys.argv[1:]
# hf_transfer is optional: only keep it enabled if importable, otherwise the hub client refuses to run
try:
    import hf_transfer  # noqa: F401
except Exception:
    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
try:
    p = hf_hub_download(repo_id=repo, filename=file, revision=rev, local_dir=stage)
except Exception as e:  # make the reason visible in the worker log
    print(f"hfget: hub client failed: {type(e).__name__}: {e}", file=sys.stderr)
    raise SystemExit(1)
print(p)
PY
    [ -s "$stage/$file" ] && mv -f "$stage/$file" "$dest" || return 1
    rm -rf "$stage"
}

fast_http() {
    if [ "$HFGET_DRY" = "1" ]; then echo "http: $url -> $dest"; return 0; fi
    local part="$dest.part"
    if command -v aria2c >/dev/null 2>&1; then
        echo "hfget: aria2c x16"
        aria2c -x16 -s16 -k1M -c --file-allocation=none --console-log-level=warn --summary-interval=0 \
            ${HF_TOKEN:+--header="Authorization: Bearer $HF_TOKEN"} \
            -d "$(dirname "$part")" -o "$(basename "$part")" "$url" \
            && [ -s "$part" ] && mv -f "$part" "$dest" && return 0
        echo "hfget: aria2c failed, trying wget" >&2
    else
        echo "hfget: aria2c not installed, using wget"
    fi
    wget -nv -c --tries=5 ${HF_TOKEN:+--header="Authorization: Bearer $HF_TOKEN"} "$url" -O "$part" \
        && [ -s "$part" ] && mv -f "$part" "$dest" && return 0
    echo "hfget: wget failed" >&2
    [ -s "$part" ] || rm -f "$part"   # keep a real partial for resume, not an empty one
    return 1
}

start=$(date +%s)
[ "$HFGET_DRY" = "1" ] || probe
if hub_download || { echo "hfget: hub download failed for $url, falling back to direct HTTP" >&2; fast_http; }; then
    [ "$HFGET_DRY" = "1" ] || echo "hfget: $(basename "$dest") $(du -h "$dest" | cut -f1) in $(( $(date +%s) - start ))s"
    exit 0
fi
rm -f "$dest"   # belt and braces: nothing half-written under the final name
echo "hfget: FAILED $url" >&2
exit 1
