"""
generate_video — MiniMax-H3 build.

Same request/response contract as the Wan 2.2 build, so the existing GUI and
clients work unchanged:

    in : prompt, image_base64|image_url|image_path, end_image_*, width, height,
         length (GUI frames at 16 fps) | frames (H3, 24 fps) | duration (s),
         seed, steps, lora_pairs [{high, high_weight, low, low_weight}] | loras
    out: {"video": <base64 mp4 with audio>, ...}

Accepted and ignored because H3's template graph has no use for them:
negative_prompt, cfg (BasicGuider, no CFG), context_overlap.

Extra inputs: turbo (8-step LoRA), audio (false = silent clip), model (pick a
diffusion model by filename), sampler, scheduler.

Info jobs: {"models_status": true}, {"download_models": true}, {"diagnostics": true}.
"""
import base64
import binascii  # Base64 에러 처리를 위해 import
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

import runpod
import websocket

import minimax_h3 as h3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server_address = os.getenv("SERVER_ADDRESS", "127.0.0.1")
client_id = str(uuid.uuid4())

COMFY_INPUT_DIR = os.getenv("COMFY_INPUT_DIR", "/ComfyUI/input")
COMFY_OUTPUT_DIR = os.getenv("COMFY_OUTPUT_DIR", "/ComfyUI/output")
# same rule as fetch_models.py: the volume when one is mounted, else the image
MODELS_ROOT = os.getenv("MODELS_ROOT") or ("/runpod-volume/models" if os.path.isdir("/runpod-volume") else "/ComfyUI/models")
MODEL_DIRS = {   # mirrors extra_model_paths.yaml
    "diffusion_models": ["/ComfyUI/models/diffusion_models", "/ComfyUI/models/unet", f"{MODELS_ROOT}/diffusion_models"],
    "text_encoders": ["/ComfyUI/models/text_encoders", "/ComfyUI/models/clip", f"{MODELS_ROOT}/text_encoders"],
    "vae": ["/ComfyUI/models/vae", f"{MODELS_ROOT}/vae"],
    "loras": ["/ComfyUI/models/loras", f"{MODELS_ROOT}/loras", "/runpod-volume/loras"],
}
MODEL_EXTS = (".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".gguf")
PROGRESS_FILE = os.getenv("MODELS_PROGRESS_FILE", "/tmp/model-progress.json")
DOWNLOAD_LOG = os.getenv("MODELS_DOWNLOAD_LOG", "/tmp/model-download.log")
VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".mov")


class JobError(Exception):
    """Raised for user-facing problems (bad input, missing model, ComfyUI errors)."""


# ---------------------------------------------------------------------------
# input staging (unchanged from the Wan build)
# ---------------------------------------------------------------------------
def task_input_dir(task_id):
    """Per-job staging folder inside ComfyUI's input directory."""
    return os.path.join(COMFY_INPUT_DIR, task_id)


def comfy_image_ref(path):
    """What to put in a LoadImage node for a staged file: the path relative to the
    ComfyUI input directory (required by current ComfyUI). Anything outside it is
    passed through unchanged for older builds that still accept absolute paths."""
    path = os.path.abspath(path)
    root = os.path.abspath(COMFY_INPUT_DIR)
    if path.startswith(root + os.sep):
        return os.path.relpath(path, root).replace(os.sep, "/")
    return path


def process_input(input_data, task_id, output_filename, input_type):
    """입력 데이터를 처리하여 파일 경로를 반환하는 함수 (always a file under the ComfyUI input dir)"""
    temp_dir = task_input_dir(task_id)
    if input_type == "path":
        # 경로인 경우: copy into the input dir so ComfyUI's LoadImage will accept it
        logger.info(f"📁 경로 입력 처리: {input_data}")
        if not os.path.isfile(input_data):
            raise JobError(f"image path not found on the worker: {input_data}")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        shutil.copyfile(input_data, file_path)
        return file_path
    elif input_type == "url":
        # URL인 경우 다운로드
        logger.info(f"🌐 URL 입력 처리: {input_data}")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        return download_file_from_url(input_data, file_path)
    elif input_type == "base64":
        # Base64인 경우 디코딩하여 저장
        logger.info(f"🔢 Base64 입력 처리")
        return save_base64_to_file(input_data, temp_dir, output_filename)
    else:
        raise JobError(f"Unsupported input type: {input_type}")

        
def download_file_from_url(url, output_path):
    """URL에서 파일을 다운로드하는 함수"""
    try:
        # wget을 사용하여 파일 다운로드
        result = subprocess.run([
            'wget', '-O', output_path, '--no-verbose', url
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✅ URL에서 파일을 성공적으로 다운로드했습니다: {url} -> {output_path}")
            return output_path
        else:
            logger.error(f"❌ wget 다운로드 실패: {result.stderr}")
            raise JobError(f"URL download failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.error("❌ 다운로드 시간 초과")
        raise JobError("URL download timed out")
    except JobError:
        raise
    except Exception as e:
        logger.error(f"❌ 다운로드 중 오류 발생: {e}")
        raise JobError(f"URL download error: {e}")


def strip_data_url_prefix(base64_data):
    """Accept both raw base64 and data URLs ("data:image/png;base64,....")."""
    if isinstance(base64_data, str) and base64_data.startswith("data:"):
        comma = base64_data.find(",")
        if comma != -1:
            return base64_data[comma + 1:]
    return base64_data


def save_base64_to_file(base64_data, temp_dir, output_filename):
    """Base64 데이터를 파일로 저장하는 함수"""
    try:
        # Base64 문자열 디코딩
        decoded_data = base64.b64decode(strip_data_url_prefix(base64_data))
        
        # 디렉토리가 존재하지 않으면 생성
        os.makedirs(temp_dir, exist_ok=True)
        
        # 파일로 저장
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        with open(file_path, 'wb') as f:
            f.write(decoded_data)
        
        logger.info(f"✅ Base64 입력을 '{file_path}' 파일로 저장했습니다.")
        return file_path
    except (binascii.Error, ValueError) as e:
        logger.error(f"❌ Base64 디코딩 실패: {e}")
        raise JobError(f"Base64 decoding failed: {e}")


def queue_prompt(prompt):
    url = f"http://{server_address}:8188/prompt"
    logger.info(f"Queueing prompt to: {url}")
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(url, data=data)
    try:
        return json.loads(urllib.request.urlopen(req).read())
    except urllib.error.HTTPError as e:
        # ComfyUI answers 400 with a JSON body describing which node/input failed
        # validation (e.g. an LLM model name that is not in the node's list).
        body = e.read().decode("utf-8", "replace")
        raise JobError(f"ComfyUI rejected the workflow: {summarize_validation_error(body)}")


def summarize_validation_error(body):
    try:
        data = json.loads(body)
    except Exception:
        return body[:2000]
    parts = []
    err = data.get("error") or {}
    if isinstance(err, dict) and err.get("message"):
        parts.append(err["message"])
    for node_id, info in (data.get("node_errors") or {}).items():
        class_type = info.get("class_type", "?")
        for e in info.get("errors", []):
            msg = e.get("message", "")
            details = e.get("details", "")
            parts.append(f"node {node_id} ({class_type}): {msg} {details}".strip())
    return "; ".join(parts) or body[:2000]


def get_history(prompt_id):
    url = f"http://{server_address}:8188/history/{prompt_id}"
    logger.info(f"Getting history from: {url}")
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())


def read_output_file(item):
    """Bytes of one history output entry (VHS gives fullpath, core nodes don't)."""
    full = item.get("fullpath")
    if full and os.path.isfile(full):
        with open(full, "rb") as f:
            return f.read()
    path = os.path.join(COMFY_OUTPUT_DIR if item.get("type", "output") == "output" else "/ComfyUI/temp",
                        item.get("subfolder") or "", item.get("filename") or "")
    if os.path.isfile(path):
        with open(path, "rb") as f:
            return f.read()
    q = urllib.parse.urlencode({"filename": item.get("filename"), "subfolder": item.get("subfolder") or "",
                                "type": item.get("type") or "output"})
    with urllib.request.urlopen(f"http://{server_address}:8188/view?{q}") as r:
        return r.read()


def collect_videos(history):
    """Video files from a finished prompt, whichever node saved them.

    SaveVideo reports {"images": [...], "animated": [true]}, VHS reports {"gifs": [...]}.
    """
    found = []
    for node_id, out in (history.get("outputs") or {}).items():
        for key in ("gifs", "videos", "images"):
            for item in out.get(key) or []:
                name = str(item.get("filename", ""))
                if name.lower().endswith(VIDEO_EXTS):
                    found.append((node_id, item))
    return found


def run_workflow(ws, prompt):
    """Queue the graph, wait for it, and return (video_bytes | None, error | None)."""
    prompt_id = queue_prompt(prompt)["prompt_id"]
    error = None
    while True:
        out = ws.recv()
        if not isinstance(out, str):
            continue
        message = json.loads(out)
        mtype, data = message.get("type"), message.get("data") or {}
        if mtype == "executing":
            if data.get("node") is None and data.get("prompt_id") == prompt_id:
                break
        elif mtype == "execution_error" and data.get("prompt_id") == prompt_id:
            error = (f"node {data.get('node_id')} ({data.get('node_type')}): "
                     f"{data.get('exception_type', '')} {data.get('exception_message', '')}").strip()
            logger.error(f"ComfyUI execution error: {error}")
        elif mtype == "execution_interrupted" and data.get("prompt_id") == prompt_id:
            error = "execution interrupted"
    history = get_history(prompt_id).get(prompt_id, {})
    if error is None and (history.get("status") or {}).get("status_str") == "error":
        error = "execution failed (see worker logs)"
    for node_id, item in collect_videos(history):
        try:
            return read_output_file(item), error
        except Exception as e:  # noqa: BLE001
            error = error or f"could not read {item.get('filename')}: {e}"
    return None, error

def wait_for_comfyui():
    ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
    logger.info(f"Connecting to WebSocket: {ws_url}")
    
    # 먼저 HTTP 연결이 가능한지 확인
    http_url = f"http://{server_address}:8188/"
    logger.info(f"Checking HTTP connection to: {http_url}")
    
    # HTTP 연결 확인 (최대 1분)
    max_http_attempts = 180
    for http_attempt in range(max_http_attempts):
        try:
            response = urllib.request.urlopen(http_url, timeout=5)
            logger.info(f"HTTP 연결 성공 (시도 {http_attempt+1})")
            break
        except Exception as e:
            logger.warning(f"HTTP 연결 실패 (시도 {http_attempt+1}/{max_http_attempts}): {e}")
            if http_attempt == max_http_attempts - 1:
                raise Exception("ComfyUI 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
            time.sleep(1)
    
    ws = websocket.WebSocket()
    # 웹소켓 연결 시도 (최대 3분)
    max_attempts = int(180/5)  # 3분 (1초에 한 번씩 시도)
    for attempt in range(max_attempts):
        try:
            ws.connect(ws_url)
            logger.info(f"웹소켓 연결 성공 (시도 {attempt+1})")
            break
        except Exception as e:
            logger.warning(f"웹소켓 연결 실패 (시도 {attempt+1}/{max_attempts}): {e}")
            if attempt == max_attempts - 1:
                raise Exception("웹소켓 연결 시간 초과 (3분)")
            time.sleep(5)
    return ws


# ---------------------------------------------------------------------------
# models on this worker
# ---------------------------------------------------------------------------
def list_models(folder):
    """Filenames ComfyUI offers for a model folder (relative to each search dir)."""
    names = []
    for d in MODEL_DIRS[folder]:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if not f.lower().endswith(MODEL_EXTS):
                    continue
                p = os.path.join(root, f)
                try:
                    if os.path.getsize(p) < 1024 * 1024:      # a stub or an aborted download
                        continue
                except OSError:
                    continue
                names.append(os.path.relpath(p, d).replace(os.sep, "/"))
    return sorted(dict.fromkeys(names), key=str.lower)


def manifest_state():
    try:
        with open(os.path.join(MODELS_ROOT, ".manifest.json"), encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def preferred_unets():
    """Diffusion models fetched from Civitai, in manifest order — they win over
    ComfyUI's base file when several are present."""
    out = []
    for rec in manifest_state().values():
        if rec.get("folder") == "diffusion_models" and rec.get("source") == "civitai" and rec.get("file"):
            out.append(rec["file"])
    return out


def download_progress():
    try:
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def _missing(kind, name, available):
    prog = download_progress()
    hint = ""
    if prog and prog.get("state") == "running":
        hint = f" — models are still downloading ({len(prog.get('done') or [])}/{prog.get('total')} done)"
    elif prog and prog.get("failed"):
        hint = f" — the start-up download reported {len(prog['failed'])} failure(s); see {{\"models_status\":true}}"
    shown = ", ".join(available[:12]) + (" …" if len(available) > 12 else "")
    return JobError(f"{kind} {name!r} is not on this worker{hint}. Available: {shown or 'none'}")


def resolve_models(job):
    unets = list_models("diffusion_models")
    try:
        unet = h3.pick_unet(job.get("model") or job.get("unet_name"), unets, preferred_unets())
    except h3.RequestError as e:
        prog = download_progress()
        if prog and prog.get("state") == "running":
            raise JobError(f"{e} Models are still downloading ({len(prog.get('done') or [])}/{prog.get('total')}).")
        raise JobError(str(e))
    picked = {"unet": unet}
    for key, folder, wanted in (("clip", "text_encoders", job.get("clip_name") or h3.DEFAULTS["clip"]),
                                ("video_vae", "vae", job.get("video_vae") or h3.DEFAULTS["video_vae"]),
                                ("audio_vae", "vae", job.get("audio_vae") or h3.DEFAULTS["audio_vae"])):
        if key == "audio_vae" and job.get("audio") is False:
            continue
        have = list_models(folder)
        hit = h3.match_file(wanted, have)
        if not hit:
            raise _missing({"clip": "text encoder", "video_vae": "video VAE", "audio_vae": "audio VAE"}[key], wanted, have)
        picked[key] = hit
    return picked


def resolve_lora_names(requested, turbo):
    have = list_models("loras")
    out = []
    for name, strength in requested:
        hit = h3.match_file(name, have)
        if not hit:
            raise _missing("LoRA", name, have)
        out.append((hit, strength))
    turbo_name = None
    if turbo:
        turbo_name = h3.match_file(h3.DEFAULTS["turbo_lora"], have)
        if not turbo_name:
            raise _missing("turbo LoRA", h3.DEFAULTS["turbo_lora"], have)
    return out, turbo_name


# ---------------------------------------------------------------------------
# info jobs
# ---------------------------------------------------------------------------
CIVITAI_TOKEN_VARS = ("CIVITAI_TOKEN", "CIVITAI_API_TOKEN", "CIVITAI_API_KEY", "CIVITAI_KEY")


def _read(path, tail=None):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            data = f.read()
        return data if tail is None else "\n".join(data.splitlines()[-tail:])
    except OSError:
        return None


def _masked_env(names):
    for v in names:
        val = os.getenv(v)
        if val:
            return {"variable": v, "value": f"{val[:4]}… ({len(val)} chars)"}
    return None


def models_status():
    state = manifest_state()
    loras = []
    for rec in state.values():
        if rec.get("folder") == "loras":
            loras.append({"file": rec.get("file"), "name": rec.get("model_name") or None,
                          "trigger_words": rec.get("trained_words") or [], "base_model": rec.get("base_model") or None})
    return {
        "progress": download_progress(),
        "models": {f: list_models(f) for f in MODEL_DIRS},
        "default_unet": _safe(lambda: h3.pick_unet(None, list_models("diffusion_models"), preferred_unets())),
        "civitai_loras": loras,
        "download_log_tail": _read(DOWNLOAD_LOG, tail=40),
    }


def _safe(fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


def diagnostics():
    return {
        "build_commit": (_read("/build-commit") or "unknown").strip(),
        "build_date": (_read("/build-date") or "unknown").strip(),
        "variant": "minimax-h3",
        "volume_mounted": os.path.isdir("/runpod-volume"),
        "models_root": MODELS_ROOT,
        "civitai_token": _masked_env(CIVITAI_TOKEN_VARS),
        "hf_token": _masked_env(("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")),
        "env": {k: os.getenv(k) for k in ("MINIMAX_UNET", "MINIMAX_CLIP", "MINIMAX_STEPS", "MINIMAX_TURBO_STEPS",
                                          "LEGACY_LENGTH_FPS", "MODELS_DOWNLOAD", "MODELS_MANIFEST")},
        "tools": {t: shutil.which(t) for t in ("hfget", "aria2c", "wget", "ffmpeg")},
        "status": models_status(),
    }


def download_models():
    """Run the manifest download now, inside the job, and report what happened."""
    proc = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fetch_models.py")],
                          capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    try:
        with open(DOWNLOAD_LOG, "a", encoding="utf-8") as f:
            f.write(out)
    except OSError:
        pass
    return {"returncode": proc.returncode, "progress": download_progress(), "log": out[-8000:],
            "models": {f: list_models(f) for f in MODEL_DIRS}}


# ---------------------------------------------------------------------------
# the job
# ---------------------------------------------------------------------------
def stage_image(job_input, prefix, task_id):
    for key, kind in ((f"{prefix}_path", "path"), (f"{prefix}_url", "url"), (f"{prefix}_base64", "base64")):
        if job_input.get(key):
            return process_input(job_input[key], task_id, f"{prefix}.png", kind)
    return None


def handler(job):
    job_input = job.get("input", {}) or {}
    if isinstance(job_input, str):
        try:
            job_input = json.loads(job_input)
        except json.JSONDecodeError:
            return {"error": "input must be a JSON object"}

    if job_input.get("diagnostics"):
        return _info(diagnostics, "diagnostics")
    if job_input.get("models_status"):
        return _info(models_status, "models_status")
    if job_input.get("download_models") or job_input.get("download_prompt_model"):
        return _info(download_models, None)

    logger.info("Received job input: " + json.dumps(
        {k: (f"<{len(v)} chars>" if isinstance(v, str) and len(v) > 200 else v) for k, v in job_input.items()},
        ensure_ascii=False, default=str))

    if job_input.get("expand_only"):
        return {"error": "Prompt expansion is not part of the MiniMax-H3 build — use the Wan build "
                         "(branch feature/prompt-expansion) for prompt previews."}
    prompt_text = job_input.get("prompt")
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        return {"error": "'prompt' is required"}

    task_id = f"task_{uuid.uuid4()}"
    warnings = []
    try:
        try:
            frames, w1 = h3.resolve_frames(job_input)
            width, height = h3.resolve_size(job_input)
            loras = h3.resolve_loras(job_input)
        except h3.RequestError as e:
            raise JobError(str(e))
        warnings += w1
        if job_input.get("prompt_expansion"):
            warnings.append("prompt_expansion is not available on the MiniMax-H3 build; the prompt was used as written")

        turbo = bool(job_input.get("turbo"))
        audio = job_input.get("audio", True) is not False
        steps = job_input.get("steps")
        steps = int(steps) if steps not in (None, "") else (h3.DEFAULTS["turbo_steps"] if turbo else h3.DEFAULTS["steps"])
        if steps < 1:
            raise JobError("steps must be at least 1")
        try:
            seed = int(job_input.get("seed", 0)) % (2 ** 63)
        except (TypeError, ValueError):
            raise JobError("seed must be an integer")

        models = resolve_models(dict(job_input, audio=audio))
        lora_names, turbo_name = resolve_lora_names(loras, turbo)

        first = stage_image(job_input, "image", task_id)
        last = stage_image(job_input, "end_image", task_id)

        graph = h3.build_graph(
            prompt=prompt_text, width=width, height=height, frames=frames, seed=seed, steps=steps,
            unet=models["unet"], clip=models["clip"], video_vae=models["video_vae"],
            audio_vae=models.get("audio_vae"), loras=lora_names, turbo_lora=turbo_name,
            first_image=comfy_image_ref(first) if first else None,
            last_image=comfy_image_ref(last) if last else None,
            sampler=job_input.get("sampler"), scheduler=job_input.get("scheduler"), audio=audio,
        )
        mode = "flf2va" if first and last else "i2va" if first else "t2va"
        logger.info(f"MiniMax-H3 {mode}: {width}x{height}, {frames} frames, {steps} steps, unet={models['unet']}, "
                    f"loras={[n for n, _ in lora_names]}{', turbo' if turbo_name else ''}")

        ws = wait_for_comfyui()
        try:
            video, error = run_workflow(ws, graph)
        finally:
            ws.close()
        if not video:
            raise JobError(error or "Video not found in ComfyUI output")
        result = {
            "video": base64.b64encode(video).decode("utf-8"),
            "fps": h3.FPS, "frames": frames, "duration": round(frames / h3.FPS, 3),
            "width": width, "height": height, "seed": seed, "steps": steps, "mode": mode,
            "model": models["unet"], "loras": [{"name": n, "strength": s} for n, s in lora_names],
            "audio": audio,
        }
        if turbo_name:
            result["turbo"] = turbo_name
        if warnings:
            result["warnings"] = warnings
        return result
    except JobError as e:
        logger.error(f"Job failed: {e}")
        return {"error": str(e)}
    finally:
        shutil.rmtree(task_input_dir(task_id), ignore_errors=True)


def _info(fn, key):
    try:
        out = fn()
        return {key: out} if key else out
    except Exception as e:  # noqa: BLE001
        return {"error": f"{fn.__name__} failed: {type(e).__name__}: {e}"}


if __name__ == "__main__":
    logger.info(f"generate_video (MiniMax-H3) handler starting — build {(_read('/build-commit') or 'unknown').strip()}; "
                f"diffusion models: {list_models('diffusion_models')}")
    runpod.serverless.start({"handler": handler})
