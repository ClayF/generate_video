# Use specific version of  nvidia cuda image
# FROM wlsdml1114/my-comfy-models:v1 as model_provider
# FROM wlsdml1114/multitalk-base:1.7 as runtime
FROM wlsdml1114/engui_genai-base_blackwell:1.1 as runtime

# Model downloads: the hub client with its parallel Xet / hf_transfer backends
# (many connections per file) plus aria2c as a fallback — several times faster
# than the single-stream wget the image used to rely on. See hfget.sh.
RUN pip install -U "huggingface_hub[hf_transfer,hf_xet]" && \
    (apt-get update && apt-get install -y --no-install-recommends aria2 && rm -rf /var/lib/apt/lists/*) \
    || echo "aria2 not installed (apt unavailable); hfget will use the hub client / wget"
COPY hfget.sh /usr/local/bin/hfget
COPY mmproj-name.sh /usr/local/bin/mmproj-name
RUN chmod +x /usr/local/bin/hfget /usr/local/bin/mmproj-name
RUN pip install runpod websocket-client

WORKDIR /

RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd /ComfyUI && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Comfy-Org/ComfyUI-Manager.git && \
    cd ComfyUI-Manager && \
    pip install -r requirements.txt
    
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/city96/ComfyUI-GGUF && \
    cd ComfyUI-GGUF && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-KJNodes && \
    cd ComfyUI-KJNodes && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite && \
    cd ComfyUI-VideoHelperSuite && \
    pip install -r requirements.txt
    
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kael558/ComfyUI-GGUF-FantasyTalking && \
    cd ComfyUI-GGUF-FantasyTalking && \
    pip install -r requirements.txt
    
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/orssorbit/ComfyUI-wanBlockswap

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper && \
    cd ComfyUI-WanVideoWrapper && \
    pip install -r requirements.txt

    
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/eddyhhlure1Eddy/IntelligentVRAMNode && \
    git clone https://github.com/eddyhhlure1Eddy/auto_wan2.2animate_freamtowindow_server && \
    git clone https://github.com/eddyhhlure1Eddy/ComfyUI-AdaptiveWindowSize && \
    cd ComfyUI-AdaptiveWindowSize/ComfyUI-AdaptiveWindowSize && \
    mv * ../

# ---------------------------------------------------------------------------
# Prompt expansion: ComfyUI-MultiModal-Prompt-Nodes (Wan Video Prompt Generator)
# + llama-cpp-python with CUDA so the Qwen3-VL GGUF runs on the worker GPU.
# The JamePeng fork ships prebuilt cu128 wheels for Python 3.10-3.14; the tag is
# picked from the base image's interpreter at build time.
# ---------------------------------------------------------------------------
ARG LLAMA_CPP_PYTHON_RELEASE=v0.3.49-cu128-linux-20260831
ARG LLAMA_CPP_PYTHON_WHEEL_VERSION=0.3.49%2Bcu128
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kantan-kanto/ComfyUI-MultiModal-Prompt-Nodes && \
    pip install dashscope pillow numpy
RUN PYTAG=$(python -c 'import sys; print("cp%d%d" % sys.version_info[:2])') && \
    pip install "https://github.com/JamePeng/llama-cpp-python/releases/download/${LLAMA_CPP_PYTHON_RELEASE}/llama_cpp_python-${LLAMA_CPP_PYTHON_WHEEL_VERSION}-${PYTAG}-${PYTAG}-linux_x86_64.whl" && \
    python -c "import llama_cpp; print('llama-cpp-python', llama_cpp.__version__)"

# Local vision LLM for prompt expansion: Qwen3-VL-8B-Instruct **abliterated v2**
# (prithivMLmods) Q8_0 + mmproj, ~9.4 GB. Refusal-free, so it will describe any
# frame; swap PROMPT_LLM_URL/PROMPT_MMPROJ_URL for the stock model if preferred:
#   https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main/Qwen3VL-8B-Instruct-Q8_0.gguf
#   https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf
# It is NOT baked into the image by default because the download
# pushes RunPod's build past its time limit; entrypoint.sh fetches it into
# /runpod-volume/LLM on the first start instead (see PROMPT_LLM_URL below).
# To bake it in anyway (own registry, no build timeout):
#   --build-arg BAKE_PROMPT_LLM=1
ARG BAKE_PROMPT_LLM=0
ENV PROMPT_LLM_URL=https://huggingface.co/prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF/resolve/main/Qwen3-VL-8B-Instruct-abliterated-v2.Q8_0.gguf \
    PROMPT_MMPROJ_URL=https://huggingface.co/prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF/resolve/main/Qwen3-VL-8B-Instruct-abliterated-v2.mmproj-Q8_0.gguf
# The node only recognises projectors named mmproj-*.gguf, so the mmproj is
# stored under a normalised name (entrypoint.sh applies the same rule).
RUN mkdir -p /ComfyUI/models/LLM && \
    if [ "$BAKE_PROMPT_LLM" = "1" ]; then \
        hfget "${PROMPT_LLM_URL}" "/ComfyUI/models/LLM/$(basename "${PROMPT_LLM_URL}")" && \
        hfget "${PROMPT_MMPROJ_URL}" "/ComfyUI/models/LLM/$(/usr/local/bin/mmproj-name "${PROMPT_MMPROJ_URL}")"; \
    fi

# Tiny output node that publishes the expanded prompt to the job history
COPY custom_nodes/generate_video_prompt_nodes /ComfyUI/custom_nodes/generate_video_prompt_nodes

RUN hfget https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/I2V/Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors /ComfyUI/models/diffusion_models/Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors
RUN hfget https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/I2V/Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors /ComfyUI/models/diffusion_models/Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors

RUN hfget https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors /ComfyUI/models/loras/high_noise_model.safetensors
RUN hfget https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors /ComfyUI/models/loras/low_noise_model.safetensors

RUN hfget https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors /ComfyUI/models/clip_vision/clip_vision_h.safetensors
RUN hfget https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors /ComfyUI/models/text_encoders/umt5-xxl-enc-bf16.safetensors
RUN hfget https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors /ComfyUI/models/vae/Wan2_1_VAE_bf16.safetensors

COPY . .
COPY extra_model_paths.yaml /ComfyUI/extra_model_paths.yaml
# Which commit this image was built from — shown by {"diagnostics": true} and at handler start
RUN (git -C / rev-parse --short HEAD 2>/dev/null || echo unknown) > /build-commit && \
    date -u +%Y-%m-%dT%H:%M:%SZ > /build-date && echo "build $(cat /build-commit) $(cat /build-date)"
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
