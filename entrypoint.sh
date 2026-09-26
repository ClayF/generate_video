#!/bin/bash
# generate_video — MiniMax-H3 build
set -e

# ---------------------------------------------------------------------------
# Models: models_minimax.txt → the Network Volume (fetch_models.py).
#   MODELS_DOWNLOAD=background   (default) start serving right away; jobs that
#                                need a file still on its way say so
#   MODELS_DOWNLOAD=foreground   finish every download before serving
#   MODELS_DOWNLOAD=off          you manage the files yourself
# Civitai files need CIVITAI_TOKEN (or CIVITAI_API_KEY) on the endpoint.
# ---------------------------------------------------------------------------
DL_LOG=/tmp/model-download.log
MODE="${MODELS_DOWNLOAD:-background}"
echo "=== entrypoint $(date -u +%Y-%m-%dT%H:%M:%SZ) build $(cat /build-commit 2>/dev/null || echo unknown) — MiniMax-H3 ===" | tee -a "$DL_LOG"
if [ ! -d /runpod-volume ]; then
    echo "WARNING: no Network Volume at /runpod-volume — MiniMax-H3 needs 50+ GB of models and" \
         "they would be re-downloaded into the container disk on every cold start. Attach a volume." | tee -a "$DL_LOG"
fi
if [ -z "${CIVITAI_TOKEN}${CIVITAI_API_TOKEN}${CIVITAI_API_KEY}${CIVITAI_KEY}" ]; then
    echo "WARNING: no CIVITAI_TOKEN set — the Civitai models and LoRAs cannot be downloaded." | tee -a "$DL_LOG"
fi
case "$MODE" in
    off)        echo "Model download disabled (MODELS_DOWNLOAD=off)." | tee -a "$DL_LOG" ;;
    foreground) python -u /fetch_models.py 2>&1 | tee -a "$DL_LOG" || echo "Some downloads failed — see $DL_LOG" ;;
    *)          echo "Fetching models in the background (log: $DL_LOG)"
                nohup python -u /fetch_models.py >> "$DL_LOG" 2>&1 & ;;
esac

# Start ComfyUI in the background
echo "Starting ComfyUI in the background..."
python /ComfyUI/main.py --listen ${COMFY_ARGS:---use-sage-attention} &

# Wait for ComfyUI to be ready
echo "Waiting for ComfyUI to be ready..."
max_wait=180
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
echo "Starting the handler..."
exec python handler.py
