#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Optional: DashScope key for cloud Qwen prompt-expansion models. Local GGUF
# models (the default) do not need it.
MM_NODE_DIR=/ComfyUI/custom_nodes/ComfyUI-MultiModal-Prompt-Nodes
if [ -n "$DASHSCOPE_API_KEY" ] && [ -d "$MM_NODE_DIR" ]; then
    printf '%s' "$DASHSCOPE_API_KEY" > "$MM_NODE_DIR/api_key.txt"
    echo "DashScope API key written for prompt expansion."
fi

# ---------------------------------------------------------------------------
# Prompt-expansion model: fetch once onto the Network Volume (or the image's
# LLM folder when it was baked in / no volume is attached).
#   PROMPT_LLM_URL / PROMPT_MMPROJ_URL   what to fetch (set in the Dockerfile,
#                                        override on the endpoint to use another
#                                        quant or Qwen family)
#   PROMPT_LLM_AUTO_DOWNLOAD=0           skip entirely (e.g. cloud models only)
# ---------------------------------------------------------------------------
llm_have() {  # any Qwen GGUF (not an mmproj) already present?
    for d in /ComfyUI/models/LLM /runpod-volume/LLM; do
        [ -d "$d" ] && find "$d" -iname '*qwen*.gguf' ! -iname 'mmproj*' -size +100M 2>/dev/null | grep -q . && return 0
    done
    return 1
}
if [ "${PROMPT_LLM_AUTO_DOWNLOAD:-1}" != "0" ] && [ -n "$PROMPT_LLM_URL" ] && ! llm_have; then
    if [ -d /runpod-volume ]; then
        LLM_DEST=/runpod-volume/LLM
    else
        LLM_DEST=/ComfyUI/models/LLM
        echo "WARNING: no Network Volume at /runpod-volume — the prompt-expansion model will be" \
             "downloaded into the container disk and lost when this worker stops. Attach a" \
             "Network Volume to the endpoint so it is fetched once, or build with" \
             "--build-arg BAKE_PROMPT_LLM=1."
    fi
    mkdir -p "$LLM_DEST"
    for url in "$PROMPT_LLM_URL" "$PROMPT_MMPROJ_URL"; do
        [ -n "$url" ] || continue
        f="$LLM_DEST/$(basename "$url")"
        echo "Fetching prompt-expansion model: $url -> $f"
        # download to .part (resumable with -c) and rename only when complete, so a
        # start that gets interrupted never leaves a truncated .gguf behind
        if wget -nv -c --tries=5 "$url" -O "$f.part"; then
            mv -f "$f.part" "$f"
        else
            echo "WARNING: download failed for $url — prompt expansion will not work until it succeeds" \
                 "(the worker still serves normal generations; the partial file is kept for resume)."
            break
        fi
    done
elif llm_have; then
    echo "Prompt-expansion model present."
fi

# Start ComfyUI in the background
echo "Starting ComfyUI in the background..."
python /ComfyUI/main.py --listen --use-sage-attention &

# Wait for ComfyUI to be ready
echo "Waiting for ComfyUI to be ready..."
max_wait=120  # 최대 2분 대기
wait_count=0
while [ $wait_count -lt $max_wait ]; do
    if curl -s http://127.0.0.1:8188/ > /dev/null 2>&1; then
        echo "ComfyUI is ready!"
        break
    fi
    echo "Waiting for ComfyUI... ($wait_count/$max_wait)"
    sleep 2
    wait_count=$((wait_count + 2))
done

if [ $wait_count -ge $max_wait ]; then
    echo "Error: ComfyUI failed to start within $max_wait seconds"
    exit 1
fi

# Start the handler in the foreground
# 이 스크립트가 컨테이너의 메인 프로세스가 됩니다.
echo "Starting the handler..."
exec python handler.py