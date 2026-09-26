#!/usr/bin/env python3
"""
fetch_models.py — put everything in models_minimax.txt onto the volume.

    python fetch_models.py                 fetch what is missing
    python fetch_models.py --check         report only, download nothing
    python fetch_models.py --manifest X    use another manifest

Hugging Face files go through hfget (parallel Xet/hf_transfer, aria2c, wget).
Civitai links (…/api/download/models/<version>?fileId=<id>) are resolved through
Civitai's API first, which gives the real filename, what kind of model it is
(so "auto" lines land in the right ComfyUI folder), its size and trigger words.
The download itself carries the key as ?token=… because Civitai redirects to a
signed storage URL that refuses a second Authorization header.

Everything is written as <name>.part and renamed when complete, and each file
is locked while it downloads, so several workers sharing one Network Volume
never fetch the same file twice or see half a model.

Environment:
    CIVITAI_TOKEN / CIVITAI_API_TOKEN / CIVITAI_API_KEY / CIVITAI_KEY
    HF_TOKEN (and the usual aliases, handled by hfget)
    MODELS_ROOT        default /runpod-volume/models if a volume is mounted,
                       else /ComfyUI/models
    MODELS_MANIFEST    default models_minimax.txt next to this file
    VERIFY_SHA256=1    hash Civitai files against the published SHA256 (slow)
"""

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDERS = ("diffusion_models", "text_encoders", "vae", "loras")
CIVITAI_TOKEN_VARS = ("CIVITAI_TOKEN", "CIVITAI_API_TOKEN", "CIVITAI_API_KEY", "CIVITAI_KEY")
PROGRESS_FILE = os.getenv("MODELS_PROGRESS_FILE", "/tmp/model-progress.json")
USER_AGENT = "generate_video-minimax/1.0"


def models_root():
    if os.getenv("MODELS_ROOT"):
        return os.getenv("MODELS_ROOT")
    return "/runpod-volume/models" if os.path.isdir("/runpod-volume") else "/ComfyUI/models"


def civitai_token():
    for v in CIVITAI_TOKEN_VARS:
        val = (os.getenv(v) or "").strip().strip('"').strip("'")
        if val:
            return re.sub(r"^bearer\s+", "", val, flags=re.I)
    return None


def log(msg):
    print(f"[models] {msg}", flush=True)


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------
def parse_manifest(path):
    entries = []
    for n, raw in enumerate(open(path, encoding="utf-8"), 1):
        line = re.sub(r"(^|\s)#.*$", "", raw).strip()     # comments: whole line or after whitespace
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"{path}:{n}: expected '<folder> <url> [if-no-dit]'")
        folder, url, flags = parts[0], parts[1], parts[2:]
        if folder not in FOLDERS + ("auto",):
            raise ValueError(f"{path}:{n}: unknown folder {folder!r}")
        entries.append({"folder": folder, "url": url, "if_no_dit": "if-no-dit" in flags, "line": n})
    return entries


def is_civitai(url):
    host = urllib.parse.urlparse(url).hostname or ""
    return "civitai" in host


def civitai_ids(url):
    m = re.search(r"/api/download/models/(\d+)", url)
    if not m:
        raise ValueError(f"not a Civitai download link: {url}")
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    return m.group(1), (q.get("fileId") or [None])[0]


# ---------------------------------------------------------------------------
# Civitai
# ---------------------------------------------------------------------------
def _get_json(url, token=None, timeout=30):
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def civitai_lookup(url, token=None):
    """What a Civitai download link points at, from the public API."""
    version_id, file_id = civitai_ids(url)
    host = urllib.parse.urlparse(url).hostname
    last = None
    for h in dict.fromkeys([host, "civitai.com"]):          # the link's own host first
        try:
            data = _get_json(f"https://{h}/api/v1/model-versions/{version_id}", token)
            break
        except Exception as e:  # noqa: BLE001
            last = e
    else:
        raise RuntimeError(f"Civitai API lookup failed for version {version_id}: {last}")
    files = data.get("files") or []
    f = next((x for x in files if str(x.get("id")) == str(file_id)), None) if file_id else None
    f = f or next((x for x in files if x.get("primary")), None) or (files[0] if files else None)
    if not f:
        raise RuntimeError(f"Civitai version {version_id} lists no files")
    model = data.get("model") or {}
    return {
        "name": f.get("name"),
        "file_type": f.get("type") or "",
        "model_type": model.get("type") or "",
        "model_name": model.get("name") or "",
        "version_name": data.get("name") or "",
        "base_model": data.get("baseModel") or "",
        "trained_words": data.get("trainedWords") or [],
        "size": int(float(f.get("sizeKB") or 0) * 1024) or None,
        "sha256": ((f.get("hashes") or {}).get("SHA256") or "").lower() or None,
    }


def folder_for(info):
    """ComfyUI folder for a Civitai file, from what Civitai says it is."""
    mt, ft = info.get("model_type", "").lower(), info.get("file_type", "").lower()
    if mt in ("lora", "locon", "dora", "lycoris"):
        return "loras"
    if mt == "vae" or ft == "vae":
        return "vae"
    if "text encoder" in ft or "textencoder" in mt.replace(" ", ""):
        return "text_encoders"
    return "diffusion_models"


def with_token(url, token):
    if not token:
        return url
    parts = urllib.parse.urlparse(url)
    q = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    q = [(k, v) for k, v in q if k != "token"] + [("token", token)]
    return urllib.parse.urlunparse(parts._replace(query=urllib.parse.urlencode(q)))


def redact(text, token):
    return text.replace(token, "***") if token and text else text


def http_download(url, dest, token=None, expected_size=None):
    """Fetch url → dest via <dest>.part (resumable). aria2c when present."""
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    src = with_token(url, token)
    if shutil.which("aria2c"):
        cmd = ["aria2c", "-x16", "-s16", "-k1M", "-c", "--file-allocation=none", "--console-log-level=warn",
               "--summary-interval=30", "--auto-file-renaming=false", "--allow-overwrite=true",
               f"--user-agent={USER_AGENT}", "-d", os.path.dirname(part), "-o", os.path.basename(part), src]
    else:
        cmd = ["wget", "-nv", "-c", "--tries=5", f"--user-agent={USER_AGENT}", "-O", part, src]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = redact((proc.stdout or "") + (proc.stderr or ""), token)
    if proc.returncode != 0 or not os.path.isfile(part) or os.path.getsize(part) == 0:
        if "401" in out or "403" in out:
            raise RuntimeError("Civitai refused the download (401/403) — check CIVITAI_TOKEN on the endpoint"
                               if is_civitai(url) else f"download refused: {out.strip()[-300:]}")
        raise RuntimeError(f"download failed ({proc.returncode}): {out.strip()[-400:]}")
    size = os.path.getsize(part)
    if expected_size and abs(size - expected_size) > max(1024 * 1024, expected_size * 0.01):
        raise RuntimeError(f"size mismatch: got {size} bytes, Civitai lists {expected_size}")
    os.replace(part, dest)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# bookkeeping
# ---------------------------------------------------------------------------
def load_state(root):
    try:
        return json.load(open(os.path.join(root, ".manifest.json"), encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def save_state(root, state):
    os.makedirs(root, exist_ok=True)
    tmp = os.path.join(root, f".manifest.json.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, os.path.join(root, ".manifest.json"))


def write_progress(**kw):
    kw["updated"] = time.time()
    try:
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(kw, f)
    except OSError:
        pass


@contextlib.contextmanager
def file_lock(dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest + ".lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)      # another worker downloading it? wait
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
    with contextlib.suppress(OSError):
        os.remove(dest + ".lock")


def present(path, min_bytes=1024 * 1024):
    return os.path.isfile(path) and os.path.getsize(path) >= min_bytes


def has_dit(root):
    for base in (root, "/ComfyUI/models"):
        for sub in ("diffusion_models", "unet"):
            d = os.path.join(base, sub)
            if os.path.isdir(d) and any(present(os.path.join(d, f)) for f in os.listdir(d)
                                        if f.endswith((".safetensors", ".gguf", ".sft"))):
                return True
    return False


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def fetch_entry(entry, root, state, token, check_only=False):
    url, folder = entry["url"], entry["folder"]
    rec = dict(state.get(url) or {})
    if is_civitai(url):
        try:
            info = civitai_lookup(url, token)
            rec.update(info)
        except Exception as e:  # noqa: BLE001
            if not rec.get("name"):
                raise RuntimeError(f"cannot resolve {url}: {e}")
            log(f"Civitai lookup failed, using the name recorded earlier ({rec['name']}): {e}")
        folder = folder if folder != "auto" else folder_for(rec)
        name = rec["name"]
    else:
        name = os.path.basename(urllib.parse.urlparse(url).path)
        if folder == "auto":
            raise RuntimeError(f"'auto' only works for Civitai links: {url}")
    dest = os.path.join(root, folder, name)
    rec.update({"file": name, "folder": folder, "path": dest, "source": "civitai" if is_civitai(url) else "hf"})
    if present(dest):
        rec["status"] = "present"
        return rec
    if check_only:
        rec["status"] = "missing"
        return rec
    with file_lock(dest):
        if present(dest):                   # another worker finished it while we waited
            rec["status"] = "present"
            return rec
        t0 = time.time()
        log(f"fetching {folder}/{name} …")
        if is_civitai(url):
            if not token:
                raise RuntimeError("Civitai downloads need CIVITAI_TOKEN (or CIVITAI_API_KEY) set on the endpoint")
            http_download(url, dest, token, rec.get("size"))
            if os.getenv("VERIFY_SHA256") == "1" and rec.get("sha256"):
                got = sha256_of(dest)
                if got != rec["sha256"]:
                    os.remove(dest)
                    raise RuntimeError(f"SHA256 mismatch for {name}")
        else:
            hfget = shutil.which("hfget") or "/usr/local/bin/hfget"
            proc = subprocess.run([hfget, url, dest], capture_output=True, text=True)
            if proc.returncode != 0 or not present(dest):
                raise RuntimeError(f"hfget failed: {(proc.stdout + proc.stderr).strip()[-400:]}")
        rec["status"] = "downloaded"
        rec["seconds"] = round(time.time() - t0)
        log(f"done {folder}/{name} ({os.path.getsize(dest) / 1e9:.2f} GB in {rec['seconds']}s)")
    return rec


def run(manifest, root, check_only=False):
    entries = parse_manifest(manifest)
    token = civitai_token()
    if any(is_civitai(e["url"]) for e in entries) and not token:
        log("WARNING: no CIVITAI_TOKEN on this endpoint — Civitai files will fail")
    state = load_state(root)
    ok, failed = [], []
    write_progress(state="running", root=root, done=[], failed=[], total=len(entries), started=time.time())
    first = [e for e in entries if not e["if_no_dit"]]
    later = [e for e in entries if e["if_no_dit"]]
    for group in (first, later):
        for e in group:
            if e["if_no_dit"] and has_dit(root):
                log(f"skip {e['url'].rsplit('/', 1)[-1]} — a diffusion model is already present")
                continue
            try:
                rec = fetch_entry(e, root, state, token, check_only)
                state[e["url"]] = {k: v for k, v in rec.items() if k != "path"}
                ok.append(f"{rec['folder']}/{rec['file']}:{rec['status']}")
            except Exception as ex:  # noqa: BLE001 - keep going, report at the end
                msg = redact(str(ex), token)
                log(f"FAILED {e['url'].split('?')[0]}: {msg}")
                failed.append({"url": e["url"].split("?")[0], "line": e["line"], "error": msg})
            if not check_only:
                save_state(root, state)
            write_progress(state="running", root=root, done=ok, failed=failed, total=len(entries))
    write_progress(state="failed" if failed else "done", root=root, done=ok, failed=failed, total=len(entries))
    return ok, failed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", default=os.getenv("MODELS_MANIFEST", os.path.join(HERE, "models_minimax.txt")))
    ap.add_argument("--root", default=models_root())
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    log(f"manifest {a.manifest} → {a.root}")
    ok, failed = run(a.manifest, a.root, a.check)
    for line in ok:
        log(f"  {line}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
