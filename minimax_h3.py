"""
MiniMax-H3 image-to-video graph, built from ComfyUI's own template.

Source of truth: Comfy-Org/workflow_templates  templates/video_minimax_h3_i2v.json
(the "Image to Video (MiniMax H3)" subgraph), flattened to API format:

    UNETLoader ─┬─ [LoraLoaderModelOnly …] ─┬─ BasicScheduler ─┐
                │                           └─ BasicGuider ────┤
    CLIPLoader(type=minimax) ─┐                                 │
    VAELoader (video) ────────┼─ MiniMaxH3ImageToVideo ─ cond ──┘
    LoadImage first/last ─────┘          └─ AV latent ─ SamplerCustomAdvanced
    RandomNoise, KSamplerSelect(res_multistep) ────────────────┘
    SamplerCustomAdvanced ─ VAEDecode (video VAE) ──────┐
                          └ VAEDecodeAudio (audio VAE) ─┴─ CreateVideo(24 fps) ─ SaveVideo

Differences from the template, all additive:
  * any number of user LoRAs are chained after the UNET (the template has one
    slot, used for the optional turbo LoRA — kept as `turbo`)
  * first_frame / last_frame are optional, so the same graph does T2V, I2V and
    first+last-frame (MiniMaxH3ImageToVideo supports all three natively)
  * `audio: false` drops the audio decode and makes a silent clip

This module has no ComfyUI dependency so it can be unit tested anywhere.
"""

import os
import re

FPS = 24                     # MiniMaxH3ImageToVideo generates at 24 fps
SIZE_STEP = 32               # width/height step on the node
MIN_FRAMES = 5
TRAINED_MAX_FRAMES = 362     # node tooltip: trained range ~124-362 frames
HARD_MAX_FRAMES = 3600       # node's own max

# The Wan GUI sends `length` in frames at 16 fps. Converting through seconds
# keeps its duration readout honest (81 frames ≈ 5 s → 124 H3 frames ≈ 5.2 s).
LEGACY_LENGTH_FPS = float(os.getenv("LEGACY_LENGTH_FPS", "16"))

DEFAULTS = {
    "unet": os.getenv("MINIMAX_UNET", ""),   # empty = pick automatically
    "clip": os.getenv("MINIMAX_CLIP", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
    "video_vae": os.getenv("MINIMAX_VIDEO_VAE", "minimax_h3_video_vae_int8_convrot.safetensors"),
    "audio_vae": os.getenv("MINIMAX_AUDIO_VAE", "minimax_h3_audio_vae_fp32.safetensors"),
    "turbo_lora": os.getenv("MINIMAX_TURBO_LORA", "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"),
    "base_unet": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "sampler": os.getenv("MINIMAX_SAMPLER", "res_multistep"),
    "scheduler": os.getenv("MINIMAX_SCHEDULER", "simple"),
    "steps": int(os.getenv("MINIMAX_STEPS", "20")),          # template: 20 without turbo
    "turbo_steps": int(os.getenv("MINIMAX_TURBO_STEPS", "8")),  # template: 8 with turbo
    "width": 1344,
    "height": 768,
    "frames": 124,           # node default, ≈ 5 s
}

N = {  # fixed node ids — stable so logs and errors are readable
    "unet": "1", "clip": "2", "video_vae": "3", "audio_vae": "4",
    "first": "30", "last": "31", "i2v": "40", "noise": "41", "sampler": "42",
    "sigmas": "43", "guider": "44", "sample": "45", "decode": "46",
    "decode_audio": "47", "create": "48", "save": "49",
}
LORA_BASE_ID = 100           # user LoRAs are 100, 101, …; turbo is 199


AUTO_TURBO_MAX_STEPS = int(os.getenv("MINIMAX_AUTO_TURBO_MAX_STEPS", "10"))
_DISTILL_HINT = re.compile(r"(turbo|lightx2v|lightning|distill|\d+\s*-?\s*steps?)", re.I)


def wants_turbo(job, steps, loras):
    """Whether to add the turbo LoRA.

    Explicit wins: {"turbo": true|false}. Otherwise ("auto", the default) it goes
    on when the request asks for few steps — H3 needs ~20 steps without it, and
    the Wan GUI sends 8 by default, which would otherwise come back as an
    under-sampled smear. It stays off if the caller already chose a distill
    LoRA of their own, so two turbo adapters are never stacked.

    Returns (on, reason) so the decision can be reported back.
    """
    t = job.get("turbo", "auto")
    if isinstance(t, str) and t.strip().lower() in ("true", "1", "yes", "on"):
        t = True
    elif isinstance(t, str) and t.strip().lower() in ("false", "0", "no", "off"):
        t = False
    if t is True:
        return True, "requested"
    if t is False:
        return False, "disabled"
    own = [n for n, _ in loras if _DISTILL_HINT.search(os.path.basename(n))]
    if own:
        return False, f"auto: already using {own[0]}"
    if steps is not None and steps <= AUTO_TURBO_MAX_STEPS:
        return True, f"auto: {steps} steps"
    return False, "auto: enough steps"


class RequestError(ValueError):
    """A request the graph cannot be built from (reported back to the client)."""


# ---------------------------------------------------------------------------
# request normalisation
# ---------------------------------------------------------------------------
def align_frames(n):
    """ComfyUI's align_frame_count: round up onto the 17k+5 grid (min 5)."""
    n = max(MIN_FRAMES, int(n))
    return n + (5 - n % 17) % 17


def resolve_frames(job):
    """Frame count from, in order of preference:
         duration  seconds (the template's own control)
         frames    H3 frames at 24 fps
         length    the Wan GUI's frames at LEGACY_LENGTH_FPS (16)
    """
    warnings = []
    if job.get("duration") not in (None, ""):
        try:
            seconds = float(job["duration"])
        except (TypeError, ValueError):
            raise RequestError("duration must be a number of seconds")
        raw = round(seconds * FPS)
    elif job.get("frames") not in (None, ""):
        raw = _int(job["frames"], "frames")
    elif job.get("length") not in (None, ""):
        raw = round(_int(job["length"], "length") / LEGACY_LENGTH_FPS * FPS)
    else:
        raw = DEFAULTS["frames"]
    frames = min(align_frames(raw), HARD_MAX_FRAMES)
    if frames > TRAINED_MAX_FRAMES:
        warnings.append(f"{frames} frames is past the model's trained range (~{TRAINED_MAX_FRAMES}); expect drift")
    return frames, warnings


def round_to_step(value, step=SIZE_STEP):
    value = max(step, int(round(float(value) / step)) * step)
    return value


def resolve_size(job):
    try:
        w = round_to_step(job.get("width", DEFAULTS["width"]))
        h = round_to_step(job.get("height", DEFAULTS["height"]))
    except (TypeError, ValueError):
        raise RequestError("width and height must be numbers")
    return w, h


def resolve_loras(job):
    """User LoRAs as [(filename, strength)] from either schema:

         loras:      [{"name": "x.safetensors", "strength": 0.8}, …]   (native)
         lora_pairs: [{"high": "x", "high_weight": 0.8, "low": …}, …]  (Wan GUI)

    H3 is a single model, so a Wan pair contributes one LoRA: `high` if set,
    otherwise `low`, with the matching weight.
    """
    out = []
    for item in job.get("loras") or []:
        if isinstance(item, str):
            out.append((item, 1.0))
        elif isinstance(item, dict):
            name = item.get("name") or item.get("lora_name") or item.get("file")
            if name:
                out.append((name, _float(item.get("strength", item.get("weight", 1.0)), "lora strength")))
    for pair in job.get("lora_pairs") or []:
        if not isinstance(pair, dict):
            continue
        name, weight = pair.get("high"), pair.get("high_weight")
        if not _usable(name):
            name, weight = pair.get("low"), pair.get("low_weight")
        if _usable(name):
            out.append((name, _float(1.0 if weight is None else weight, "lora weight")))
    # drop placeholders, keep order, first occurrence wins
    seen, clean = set(), []
    for name, strength in out:
        if not _usable(name) or name in seen:
            continue
        seen.add(name)
        clean.append((name, strength))
    return clean


def _usable(name):
    return isinstance(name, str) and name.strip() and name.strip().lower() not in ("none", "null", "-")


def _int(v, what):
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        raise RequestError(f"{what} must be a number")


def _float(v, what):
    try:
        return float(v)
    except (TypeError, ValueError):
        raise RequestError(f"{what} must be a number")


# ---------------------------------------------------------------------------
# model resolution
# ---------------------------------------------------------------------------
def match_file(name, available):
    """Exact filename, else case-insensitive, else with .safetensors added, else
    by basename (so "sub/x.safetensors" and "x" both find x.safetensors)."""
    if not name:
        return None
    if name in available:
        return name
    low = {a.lower(): a for a in available}
    for cand in (name, name + ".safetensors"):
        if cand.lower() in low:
            return low[cand.lower()]
    base = os.path.basename(name).lower()
    for a in available:
        if os.path.basename(a).lower() in (base, base + ".safetensors"):
            return a
    return None


def pick_unet(requested, available, preferred=()):
    """Which diffusion model to load: the request, then MINIMAX_UNET, then the
    models fetched from Civitai (in manifest order), then ComfyUI's base file,
    then anything that looks like an H3 DiT."""
    for cand in (requested, DEFAULTS["unet"]):
        if cand:
            hit = match_file(cand, available)
            if not hit:
                raise RequestError(f"diffusion model {cand!r} is not on this worker (have: {', '.join(available) or 'none'})")
            return hit
    for cand in list(preferred) + [DEFAULTS["base_unet"]]:
        hit = match_file(cand, available)
        if hit:
            return hit
    h3 = [a for a in available if re.search(r"(minimax|h3)", a, re.I)]
    if h3:
        return sorted(h3)[0]
    if len(available) == 1:
        return available[0]
    raise RequestError("no MiniMax-H3 diffusion model found on this worker — the start-up download may still be running "
                       "(ask {\"input\":{\"models_status\":true}})")


# ---------------------------------------------------------------------------
# graph
# ---------------------------------------------------------------------------
def build_graph(*, prompt, width, height, frames, seed, steps, unet, clip, video_vae, audio_vae,
                loras=(), turbo_lora=None, first_image=None, last_image=None,
                sampler=None, scheduler=None, audio=True, fps=FPS, filename_prefix="video/MiniMax_H3"):
    g = {
        N["unet"]: {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"},
                    "_meta": {"title": "Load Diffusion Model"}},
        N["clip"]: {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "minimax", "device": "default"},
                    "_meta": {"title": "Load CLIP"}},
        N["video_vae"]: {"class_type": "VAELoader", "inputs": {"vae_name": video_vae}, "_meta": {"title": "Video VAE"}},
    }
    # model path: UNET → user LoRAs → turbo LoRA
    model = [N["unet"], 0]
    chain = list(loras) + ([(turbo_lora, 1.0)] if turbo_lora else [])
    for i, (name, strength) in enumerate(chain):
        nid = str(199 if (turbo_lora and i == len(chain) - 1) else LORA_BASE_ID + i)
        g[nid] = {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": model, "lora_name": name, "strength_model": float(strength)},
                  "_meta": {"title": "Turbo LoRA" if nid == "199" else f"LoRA {i + 1}"}}
        model = [nid, 0]

    i2v = {"clip": [N["clip"], 0], "vae": [N["video_vae"], 0], "prompt": prompt,
           "width": int(width), "height": int(height), "length": int(frames)}
    if first_image:
        g[N["first"]] = {"class_type": "LoadImage", "inputs": {"image": first_image}, "_meta": {"title": "First frame"}}
        i2v["first_frame"] = [N["first"], 0]
    if last_image:
        g[N["last"]] = {"class_type": "LoadImage", "inputs": {"image": last_image}, "_meta": {"title": "Last frame"}}
        i2v["last_frame"] = [N["last"], 0]
    g[N["i2v"]] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": i2v, "_meta": {"title": "MiniMax H3 Image to Video"}}

    g[N["noise"]] = {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}}
    g[N["sampler"]] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampler or DEFAULTS["sampler"]}}
    g[N["sigmas"]] = {"class_type": "BasicScheduler",
                      "inputs": {"model": model, "scheduler": scheduler or DEFAULTS["scheduler"], "steps": int(steps), "denoise": 1.0}}
    g[N["guider"]] = {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": [N["i2v"], 0]}}
    g[N["sample"]] = {"class_type": "SamplerCustomAdvanced",
                      "inputs": {"noise": [N["noise"], 0], "guider": [N["guider"], 0], "sampler": [N["sampler"], 0],
                                 "sigmas": [N["sigmas"], 0], "latent_image": [N["i2v"], 1]}}
    g[N["decode"]] = {"class_type": "VAEDecode", "inputs": {"samples": [N["sample"], 0], "vae": [N["video_vae"], 0]}}
    create = {"images": [N["decode"], 0], "fps": float(fps)}
    if audio:
        g[N["audio_vae"]] = {"class_type": "VAELoader", "inputs": {"vae_name": audio_vae}, "_meta": {"title": "Audio VAE"}}
        g[N["decode_audio"]] = {"class_type": "VAEDecodeAudio",
                                "inputs": {"samples": [N["sample"], 0], "vae": [N["audio_vae"], 0]}}
        create["audio"] = [N["decode_audio"], 0]
    g[N["create"]] = {"class_type": "CreateVideo", "inputs": create}
    # SaveVideo's format is a DynamicCombo: the choice plus its nested "format.codec"
    g[N["save"]] = {"class_type": "SaveVideo",
                    "inputs": {"video": [N["create"], 0], "filename_prefix": filename_prefix,
                               "format": "mp4", "format.codec": "auto"}}
    return g
