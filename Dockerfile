# generate_video — MiniMax-H3 build
#
# Nothing large is baked in: MiniMax-H3's diffusion model, 32B text encoder and
# the Civitai files come to well over 50 GB, far past RunPod's build limit. The
# image carries ComfyUI and the downloader; entrypoint.sh fetches the models in
# models_minimax.txt onto the Network Volume on first start (see README).
FROM wlsdml1114/engui_genai-base_blackwell:1.1 as runtime

# Fast model downloads: HF hub client (Xet / hf_transfer) + aria2c for Civitai
RUN pip install -U "huggingface_hub[hf_transfer,hf_xet]" && \
    (apt-get update && apt-get install -y --no-install-recommends aria2 && rm -rf /var/lib/apt/lists/*) \
    || echo "aria2 not installed (apt unavailable); downloads fall back to wget"
COPY hfget.sh /usr/local/bin/hfget
RUN chmod +x /usr/local/bin/hfget
RUN pip install runpod websocket-client

WORKDIR /

# MiniMax-H3's nodes (MiniMaxH3ImageToVideo, CLIPLoader type "minimax",
# VAEDecodeAudio, CreateVideo/SaveVideo) are core ComfyUI, so recent master is
# all that is needed; requirements.txt brings comfy-kitchen for the int8/nvfp4
# weights. Pin with --build-arg COMFYUI_REF=<commit> for reproducible builds.
ARG COMFYUI_REF=master
RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd /ComfyUI && git checkout "$COMFYUI_REF" && \
    pip install -r requirements.txt && \
    python -c "import ast; ast.parse(open('/ComfyUI/comfy_extras/nodes_minimax_h3.py').read()); print('ComfyUI has MiniMax-H3 nodes')"

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Comfy-Org/ComfyUI-Manager.git && \
    cd ComfyUI-Manager && \
    pip install -r requirements.txt

COPY . .
COPY extra_model_paths.yaml /ComfyUI/extra_model_paths.yaml
# Which commit this image was built from — shown by {"diagnostics": true} and at handler start
RUN (git -C / rev-parse --short HEAD 2>/dev/null || echo unknown) > /build-commit && \
    date -u +%Y-%m-%dT%H:%M:%SZ > /build-date && echo "build $(cat /build-commit) $(cat /build-date)"
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
