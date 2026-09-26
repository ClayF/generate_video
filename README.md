# generate_video — MiniMax-H3 build (`minimax-h3` branch)

A RunPod serverless worker that runs **MiniMax-H3** image-to-video — video
**with synchronized audio** — through ComfyUI's own template
(`video_minimax_h3_i2v`), with LoRA support. It keeps the request and response
shape of the Wan 2.2 build, so the existing GUI (`wan22-i2v.html`) and
`generate_video_client.py` work against it unchanged.

```
handler.py            RunPod handler: request → graph → ComfyUI → base64 mp4
minimax_h3.py         the graph (ComfyUI's template, flattened) + input rules
fetch_models.py       downloads models_minimax.txt onto the Network Volume
models_minimax.txt    what gets downloaded (HF + Civitai)
hfget.sh              fast Hugging Face downloader
tests/test_minimax.py offline tests (no GPU, no ComfyUI)
```

## Licence and where the endpoint may run

The weights are fetched by your worker, so your use falls under the
[MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/LICENSE),
whose territory **excludes the European Union, the United Kingdom, the Republic
of Korea and the United States**. RunPod schedules serverless workers across
its data centres by default, including US and EU ones — restrict the endpoint's
allowed data centres to regions outside those (for example Canada) so no worker
ever loads the model somewhere the licence doesn't cover.

## The graph

Straight from `Comfy-Org/workflow_templates` → `templates/video_minimax_h3_i2v.json`:

`UNETLoader` → LoRAs → `BasicScheduler` + `BasicGuider` · `CLIPLoader (type minimax)` +
video VAE → `MiniMaxH3ImageToVideo` · `RandomNoise` + `KSamplerSelect (res_multistep)` →
`SamplerCustomAdvanced` → `VAEDecode` + `VAEDecodeAudio` → `CreateVideo (24 fps)` → `SaveVideo (mp4)`

Additions: any number of LoRAs chained after the diffusion model; first and last
frame are both optional, so the same graph does text-to-video, image-to-video
and first+last-frame; the template's optional 8-step turbo LoRA is `turbo` (auto by default, see below).

## Request

Everything the Wan GUI sends is accepted:

| Field | Meaning here |
| --- | --- |
| `prompt` | required |
| `image_base64` / `image_url` / `image_path` | first frame (optional — omit for text-to-video) |
| `end_image_base64` / `_url` / `_path` | last frame (optional) |
| `width`, `height` | rounded to multiples of 32 |
| `length` | **the GUI's frames at 16 fps**, converted through seconds: 81 → 124 H3 frames (≈5.2 s) |
| `seed` | as before |
| `steps` | default 20 (the template's), 8 with `turbo`. **The GUI sends 8**, which switches turbo on automatically |
| `lora_pairs` | `[{high, high_weight, low, low_weight}]` — H3 is one model, so each pair contributes `high` (or `low` if `high` is empty) at its weight |
| `negative_prompt`, `cfg`, `context_overlap` | accepted and ignored — the template samples with `BasicGuider`, which has no negative/CFG |
| `prompt_expansion` | ignored with a warning; `expand_only` returns an error (that feature lives on the Wan build) |

Extra fields for clients that want them:

| Field | |
| --- | --- |
| `duration` | seconds (wins over `length`) |
| `frames` | H3 frames at 24 fps, snapped up to the model's 17k+5 grid |
| `loras` | `[{"name": "file.safetensors", "strength": 0.8}]` — native form |
| `turbo` | `"auto"` (default): on when `steps` ≤ 10, since plain H3 needs ~20 steps and the GUI's default of 8 would come back under-sampled; off if you already chose a turbo/lightning/distill LoRA. `true`/`false` force it. The reply's `turbo_reason` says which applied |
| `audio` | `false` for a silent clip |
| `model` | pick a diffusion model by filename (see `models_status`) |
| `sampler`, `scheduler` | override `res_multistep` / `simple` |

## Response

```json
{"video": "<base64 mp4 with audio>", "fps": 24, "frames": 124, "duration": 5.167,
 "width": 480, "height": 832, "seed": 1, "steps": 20, "mode": "i2va",
 "model": "…", "loras": [{"name": "…", "strength": 0.8}], "warnings": ["…"]}
```

`video` is what the GUI reads; the rest is extra. On failure: `{"error": "…"}` —
a missing model or LoRA names what *is* on the worker, and says so if the
start-up download is still running.

## Models

Nothing large is in the image. On first start `entrypoint.sh` runs
`fetch_models.py` in the background, which downloads `models_minimax.txt` to
`/runpod-volume/models/<folder>/` — so **attach a Network Volume** (plan on
well over 50 GB) and set:

| Endpoint env var | |
| --- | --- |
| `CIVITAI_TOKEN` | required for the Civitai models and LoRAs (`CIVITAI_API_KEY` also works) |
| `HF_TOKEN` | recommended — Hugging Face throttles anonymous downloads from datacenter IPs |
| `MODELS_DOWNLOAD` | `background` (default), `foreground` (serve only when complete), `off` |
| `MINIMAX_UNET` | force a diffusion model by filename |
| `LEGACY_LENGTH_FPS` | how `length` is read (default 16, the GUI's rate) |

What is fetched:

- from **Comfy-Org/MiniMax-H3** (the template's own files): the Qwen3-VL-32B
  text encoder (nvfp4) and the video and audio VAEs;
- the template's optional **turbo LoRA** (lightx2v);
- your **two Civitai models** (version 3294059) — `fetch_models.py` asks
  Civitai's API what each file is and files it in the matching ComfyUI folder;
- your **seven Civitai LoRAs**, with their trigger words recorded;
- ComfyUI's base FL2VA diffusion model **only if** none of the Civitai files
  turned out to be one.

The key is sent as `?token=` rather than a header: Civitai redirects to signed
storage URLs that reject a second `Authorization` header. Files download as
`.part` and are renamed when complete, each under a lock, so several workers
sharing one volume never fetch the same file twice. When more than one diffusion
model is present, the Civitai ones win, in manifest order.

Info jobs (no rendering):

```bash
curl -s -X POST https://api.runpod.ai/v2/$ENDPOINT_ID/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"input":{"models_status":true}}'
```

- `{"models_status": true}` — download progress, every model/LoRA file present,
  the diffusion model that would be used, and each Civitai LoRA's trigger words
- `{"download_models": true}` — run the download now and return its log
- `{"diagnostics": true}` — build commit, volume, token presence, tools, all of the above

## GUI notes

The Wan GUI works as-is. Two cosmetic mismatches, both harmless: its frame
scrubber steps in 1/16 s (H3 is 24 fps), and its Workflow tab still shows the
Wan graph. LoRAs go in the GUI's high-noise field; the low-noise field is only
used when high is empty.

## Tests

```bash
python3 -m unittest discover -s tests
```

The suite checks the graph against the template's node types and wiring, the
GUI's exact payload end to end with ComfyUI stubbed, frame/size rules, LoRA
mapping, and the downloader with the network faked (type-based filing, resume,
the if-no-dit fallback, token handling).
