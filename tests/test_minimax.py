"""Offline tests for the MiniMax-H3 build: python3 -m unittest discover -s tests"""

import base64
import importlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# the handler imports runpod/websocket at module level; neither is needed offline
sys.modules.setdefault("runpod", types.SimpleNamespace(serverless=types.SimpleNamespace(start=lambda *_: None)))
sys.modules.setdefault("websocket", types.SimpleNamespace(WebSocket=object))

import minimax_h3 as h3  # noqa: E402
import fetch_models as fm  # noqa: E402

# What the Wan GUI (wan22-i2v.html) actually posts, trimmed of the image.
GUI_PAYLOAD = {
    "prompt": "slow push-in as she looks up",
    "negative_prompt": "bright tones, overexposed, static",
    "seed": 8812093, "cfg": 2, "width": 480, "height": 832, "length": 81,
    "steps": 8, "context_overlap": 48,
    "lora_pairs": [
        {"high": "style_a.safetensors", "low": "style_a.safetensors", "high_weight": 0.8, "low_weight": 0.8},
        {"high": "", "low": "only_low.safetensors", "high_weight": 1, "low_weight": 0.5},
    ],
}


def links(g):
    return {(nid, k): tuple(v) for nid, n in g.items() for k, v in n["inputs"].items() if isinstance(v, list)}


class TemplateFidelity(unittest.TestCase):
    """The graph is ComfyUI's video_minimax_h3_i2v subgraph, flattened."""

    def build(self, **kw):
        base = dict(prompt="p", width=1344, height=768, frames=124, seed=1, steps=20,
                    unet="u.safetensors", clip="c.safetensors", video_vae="v.safetensors", audio_vae="a.safetensors")
        base.update(kw)
        return h3.build_graph(**base)

    def test_node_types_match_the_template(self):
        g = self.build(first_image="t/first.png")
        classes = sorted(n["class_type"] for n in g.values())
        self.assertEqual(classes, sorted([
            "UNETLoader", "CLIPLoader", "VAELoader", "VAELoader", "LoadImage", "MiniMaxH3ImageToVideo",
            "RandomNoise", "KSamplerSelect", "BasicScheduler", "BasicGuider", "SamplerCustomAdvanced",
            "VAEDecode", "VAEDecodeAudio", "CreateVideo", "SaveVideo"]))

    def test_template_values(self):
        g = self.build()
        self.assertEqual(g["2"]["inputs"]["type"], "minimax")
        self.assertEqual(g["42"]["inputs"]["sampler_name"], "res_multistep")
        self.assertEqual(g["43"]["inputs"]["scheduler"], "simple")
        self.assertEqual(g["43"]["inputs"]["denoise"], 1.0)
        self.assertEqual(g["48"]["inputs"]["fps"], 24.0)
        self.assertEqual(g["49"]["inputs"]["format"], "mp4")
        self.assertEqual(g["49"]["inputs"]["format.codec"], "auto", "SaveVideo's nested DynamicCombo key")

    def test_wiring_matches_the_template(self):
        L = links(self.build(first_image="f.png", last_image="l.png"))
        self.assertEqual(L[("40", "clip")], ("2", 0))
        self.assertEqual(L[("40", "vae")], ("3", 0), "the I2V node encodes keyframes with the video VAE")
        self.assertEqual(L[("40", "first_frame")], ("30", 0))
        self.assertEqual(L[("40", "last_frame")], ("31", 0))
        self.assertEqual(L[("44", "conditioning")], ("40", 0))
        self.assertEqual(L[("45", "latent_image")], ("40", 1))
        self.assertEqual(L[("46", "samples")], ("45", 0), "video decode takes the sampler output directly")
        self.assertEqual(L[("47", "samples")], ("45", 0), "…and so does audio decode")
        self.assertEqual(L[("46", "vae")], ("3", 0))
        self.assertEqual(L[("47", "vae")], ("4", 0))
        self.assertEqual(L[("48", "images")], ("46", 0))
        self.assertEqual(L[("48", "audio")], ("47", 0))

    def test_every_link_points_at_a_node(self):
        g = self.build(first_image="f.png", last_image="l.png", loras=[("a", 1), ("b", 0.5)], turbo_lora="t")
        for (nid, key), (src, _) in links(g).items():
            self.assertIn(src, g, f"#{nid}.{key} → #{src}")

    def test_t2v_i2v_flf2v_are_the_same_graph_with_optional_frames(self):
        t2v = self.build()
        self.assertNotIn("30", t2v)
        self.assertNotIn("first_frame", t2v["40"]["inputs"])
        i2v = self.build(first_image="f.png")
        self.assertNotIn("last_frame", i2v["40"]["inputs"])
        self.assertEqual(i2v["30"]["inputs"]["image"], "f.png")

    def test_loras_chain_after_the_unet_and_feed_scheduler_and_guider(self):
        g = self.build(loras=[("a.safetensors", 0.8), ("b.safetensors", 0.5)], turbo_lora="turbo.safetensors")
        self.assertEqual(g["100"]["inputs"]["model"], ["1", 0])
        self.assertEqual(g["101"]["inputs"]["model"], ["100", 0])
        self.assertEqual(g["199"]["inputs"]["model"], ["101", 0], "turbo goes last, as in the template")
        self.assertEqual(g["43"]["inputs"]["model"], ["199", 0])
        self.assertEqual(g["44"]["inputs"]["model"], ["199", 0])
        self.assertEqual(g["101"]["inputs"]["strength_model"], 0.5)
        self.assertEqual(g["100"]["class_type"], "LoraLoaderModelOnly")

    def test_without_loras_the_unet_feeds_sampling_directly(self):
        g = self.build()
        self.assertEqual(g["43"]["inputs"]["model"], ["1", 0])
        self.assertFalse([n for n in g.values() if n["class_type"] == "LoraLoaderModelOnly"])

    def test_silent_clip_drops_the_audio_branch(self):
        g = self.build(audio=False)
        self.assertNotIn("4", g)
        self.assertNotIn("47", g)
        self.assertNotIn("audio", g["48"]["inputs"])


class Requests(unittest.TestCase):
    def test_frame_grid_matches_comfyui(self):
        for n, want in [(1, 5), (5, 5), (6, 22), (22, 22), (120, 124), (124, 124), (125, 141)]:
            self.assertEqual(h3.align_frames(n), want, n)

    def test_gui_length_is_read_as_16fps_frames(self):
        frames, _ = h3.resolve_frames({"length": 81})          # ≈ 5.06 s in the GUI
        self.assertEqual(frames, 124)                          # ≈ 5.17 s at 24 fps
        self.assertEqual(h3.resolve_frames({"length": 49})[0], 90)

    def test_duration_and_frames_are_honoured(self):
        self.assertEqual(h3.resolve_frames({"duration": 5})[0], 124)   # the template's own default
        self.assertEqual(h3.resolve_frames({"frames": 200})[0], 209)
        self.assertEqual(h3.resolve_frames({"duration": 5, "length": 999})[0], 124, "duration wins")
        self.assertEqual(h3.resolve_frames({})[0], 124)

    def test_long_clips_warn(self):
        frames, warns = h3.resolve_frames({"duration": 20})
        self.assertEqual(frames, 481)                          # 480 → next 17k+5
        self.assertTrue(warns and "trained range" in warns[0])

    def test_size_rounds_to_32(self):
        self.assertEqual(h3.resolve_size({"width": 480, "height": 832}), (480, 832))
        self.assertEqual(h3.resolve_size({"width": 500, "height": 830}), (512, 832))
        self.assertEqual(h3.resolve_size({"width": 5, "height": 5}), (32, 32))
        self.assertEqual(h3.resolve_size({}), (1344, 768))
        with self.assertRaises(h3.RequestError):
            h3.resolve_size({"width": "wide"})

    def test_wan_lora_pairs_become_single_loras(self):
        self.assertEqual(h3.resolve_loras(GUI_PAYLOAD),
                         [("style_a.safetensors", 0.8), ("only_low.safetensors", 0.5)])

    def test_native_lora_list_and_dedup(self):
        got = h3.resolve_loras({"loras": [{"name": "a", "strength": 0.3}, "b", {"name": "a", "strength": 1}],
                                "lora_pairs": [{"high": "None", "low": ""}]})
        self.assertEqual(got, [("a", 0.3), ("b", 1.0)])

    def test_file_matching(self):
        have = ["Foo.safetensors", "sub/bar.safetensors"]
        self.assertEqual(h3.match_file("foo.safetensors", have), "Foo.safetensors")
        self.assertEqual(h3.match_file("Foo", have), "Foo.safetensors")
        self.assertEqual(h3.match_file("bar.safetensors", have), "sub/bar.safetensors")
        self.assertIsNone(h3.match_file("baz", have))

    def test_unet_choice_order(self):
        have = ["minimax_h3_fl2va_pruned_int8_convrot.safetensors", "civitai_h3_tune.safetensors"]
        self.assertEqual(h3.pick_unet(None, have, ["civitai_h3_tune.safetensors"]), "civitai_h3_tune.safetensors")
        self.assertEqual(h3.pick_unet(None, have, []), "minimax_h3_fl2va_pruned_int8_convrot.safetensors")
        self.assertEqual(h3.pick_unet("civitai_h3_tune", have, []), "civitai_h3_tune.safetensors")
        with self.assertRaises(h3.RequestError):
            h3.pick_unet("missing.safetensors", have, [])
        with self.assertRaises(h3.RequestError):
            h3.pick_unet(None, [], [])


class Manifest(unittest.TestCase):
    def test_shipped_manifest_parses(self):
        entries = fm.parse_manifest(os.path.join(ROOT, "models_minimax.txt"))
        civ = [e for e in entries if fm.is_civitai(e["url"])]
        self.assertEqual(len(civ), 9, "2 models + 7 LoRAs")
        self.assertEqual(sum(1 for e in civ if e["folder"] == "loras"), 7)
        self.assertEqual(sum(1 for e in civ if e["folder"] == "auto"), 2)
        self.assertEqual([e for e in entries if e["if_no_dit"]][0]["folder"], "diffusion_models")
        for e in entries:
            self.assertNotIn("token=", e["url"], "no keys in the manifest")

    def test_civitai_ids(self):
        self.assertEqual(fm.civitai_ids("https://civitai.red/api/download/models/3294059?fileId=3178732"),
                         ("3294059", "3178732"))

    def test_folder_from_civitai_metadata(self):
        self.assertEqual(fm.folder_for({"model_type": "LORA"}), "loras")
        self.assertEqual(fm.folder_for({"model_type": "Checkpoint", "file_type": "Model"}), "diffusion_models")
        self.assertEqual(fm.folder_for({"model_type": "Checkpoint", "file_type": "VAE"}), "vae")
        self.assertEqual(fm.folder_for({"model_type": "Other", "file_type": "Text Encoder"}), "text_encoders")

    def test_token_goes_in_the_query_and_is_redacted(self):
        u = fm.with_token("https://civitai.red/api/download/models/1?fileId=2", "SECRET")
        self.assertIn("fileId=2", u)
        self.assertIn("token=SECRET", u)
        self.assertEqual(fm.redact(f"GET {u} 401", "SECRET").count("SECRET"), 0)

    def test_token_variable_names(self):
        with mock.patch.dict(os.environ, {"CIVITAI_API_KEY": ' "Bearer abc123" '}, clear=False):
            for v in ("CIVITAI_TOKEN", "CIVITAI_API_TOKEN", "CIVITAI_KEY"):
                os.environ.pop(v, None)
            self.assertEqual(fm.civitai_token(), "abc123")


class Fetching(unittest.TestCase):
    """The downloader end to end, with the network faked."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.join(self.tmp.name, "models")
        self.manifest = os.path.join(self.tmp.name, "m.txt")
        with open(self.manifest, "w") as f:
            f.write("auto  https://civitai.red/api/download/models/10?fileId=11\n"
                    "loras https://civitai.red/api/download/models/20?fileId=21\n"
                    "diffusion_models https://huggingface.co/x/y/resolve/main/base.safetensors if-no-dit\n")
        p = mock.patch.object(fm, "PROGRESS_FILE", os.path.join(self.tmp.name, "progress.json"))
        p.start(); self.addCleanup(p.stop)

    def fake_lookup(self, url, token=None):
        vid, fid = fm.civitai_ids(url)
        if vid == "10":
            return {"name": "h3_tune_int8.safetensors", "file_type": "Model", "model_type": "Checkpoint",
                    "model_name": "H3 tune", "base_model": "MiniMax H3", "trained_words": [], "size": 2 * 1024 * 1024, "sha256": None}
        return {"name": "cool_lora.safetensors", "file_type": "Model", "model_type": "LORA",
                "model_name": "Cool", "base_model": "MiniMax H3", "trained_words": ["c00l"], "size": 2 * 1024 * 1024, "sha256": None}

    def fake_download(self, url, dest, token=None, expected_size=None):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(b"\0" * (2 * 1024 * 1024))
        self.calls.append((url, dest, token))

    def run_fetch(self, token="tok"):
        self.calls = []
        with mock.patch.object(fm, "civitai_lookup", side_effect=self.fake_lookup), \
             mock.patch.object(fm, "http_download", side_effect=self.fake_download), \
             mock.patch.object(fm, "civitai_token", return_value=token), \
             mock.patch.object(fm.subprocess, "run") as hf:
            ok, failed = fm.run(self.manifest, self.root)
        return ok, failed, hf

    def test_auto_files_land_by_type_and_base_dit_is_skipped(self):
        ok, failed, hf = self.run_fetch()
        self.assertEqual(failed, [])
        self.assertTrue(os.path.isfile(os.path.join(self.root, "diffusion_models", "h3_tune_int8.safetensors")))
        self.assertTrue(os.path.isfile(os.path.join(self.root, "loras", "cool_lora.safetensors")))
        hf.assert_not_called()   # a DiT came from Civitai, so the base one is not fetched
        state = json.load(open(os.path.join(self.root, ".manifest.json")))
        rec = state["https://civitai.red/api/download/models/20?fileId=21"]
        self.assertEqual(rec["trained_words"], ["c00l"])
        self.assertTrue(all(c[2] == "tok" for c in self.calls))

    def test_second_run_downloads_nothing(self):
        self.run_fetch()
        ok, failed, _ = self.run_fetch()
        self.assertEqual(self.calls, [])
        self.assertTrue(all(s.endswith(":present") for s in ok))

    def test_missing_token_fails_civitai_but_reports_it(self):
        ok, failed, hf = self.run_fetch(token=None)
        civ = [f for f in failed if "civitai" in f["url"]]
        self.assertEqual(len(civ), 2)
        self.assertTrue(all("CIVITAI_TOKEN" in f["error"] for f in civ))
        # with no DiT from Civitai, ComfyUI's base model is tried as the fallback
        self.assertTrue(hf.called and "base.safetensors" in " ".join(hf.call_args[0][0]))
        prog = json.load(open(fm.PROGRESS_FILE))
        self.assertEqual(prog["state"], "failed")


class HandlerEndToEnd(unittest.TestCase):
    """The handler with ComfyUI replaced by a stub, fed the GUI's exact payload."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.models = os.path.join(self.tmp.name, "models")
        for folder, names in {
            "diffusion_models": ["h3_tune_int8.safetensors", "minimax_h3_fl2va_pruned_int8_convrot.safetensors"],
            "text_encoders": [h3.DEFAULTS["clip"]],
            "vae": [h3.DEFAULTS["video_vae"], h3.DEFAULTS["audio_vae"]],
            "loras": ["style_a.safetensors", "only_low.safetensors", h3.DEFAULTS["turbo_lora"]],
        }.items():
            os.makedirs(os.path.join(self.models, folder))
            for n in names:
                with open(os.path.join(self.models, folder, n), "wb") as f:
                    f.write(b"\0" * (1024 * 1024 + 1))
        with open(os.path.join(self.models, ".manifest.json"), "w") as f:
            json.dump({"u": {"folder": "diffusion_models", "source": "civitai", "file": "h3_tune_int8.safetensors"}}, f)
        import handler
        self.h = importlib.reload(handler)
        self.h.MODELS_ROOT = self.models
        self.h.MODEL_DIRS = {k: [os.path.join(self.models, k)] for k in ("diffusion_models", "text_encoders", "vae", "loras")}
        self.h.COMFY_INPUT_DIR = os.path.join(self.tmp.name, "input")
        self.h.PROGRESS_FILE = os.path.join(self.tmp.name, "p.json")
        self.sent = []

        def fake_run(ws, graph):
            self.sent.append(graph)
            return b"MP4DATA", None
        self.patches = [mock.patch.object(self.h, "run_workflow", side_effect=fake_run),
                        mock.patch.object(self.h, "wait_for_comfyui", return_value=mock.Mock())]
        for p in self.patches:
            p.start(); self.addCleanup(p.stop)

    def payload(self, **kw):
        d = dict(GUI_PAYLOAD, image_base64=base64.b64encode(b"png").decode())
        d.update(kw)
        return {"input": d}

    def test_gui_payload_renders_and_answers_in_the_gui_shape(self):
        out = self.h.handler(self.payload())
        self.assertNotIn("error", out, out)
        self.assertEqual(base64.b64decode(out["video"]), b"MP4DATA", "the GUI reads output.video")
        self.assertEqual((out["fps"], out["frames"], out["mode"]), (24, 124, "i2va"))
        g = self.sent[0]
        self.assertEqual(g["1"]["inputs"]["unet_name"], "h3_tune_int8.safetensors", "the Civitai model wins")
        self.assertEqual(g["40"]["inputs"]["width"], 480)
        self.assertEqual(g["40"]["inputs"]["length"], 124)
        self.assertEqual(g["41"]["inputs"]["noise_seed"], 8812093)
        self.assertEqual(g["43"]["inputs"]["steps"], 8)
        self.assertEqual([g[i]["inputs"]["lora_name"] for i in ("100", "101")],
                         ["style_a.safetensors", "only_low.safetensors"])
        self.assertTrue(g["30"]["inputs"]["image"].endswith("/image.png"), "image staged inside ComfyUI/input")
        self.assertFalse(os.path.isabs(g["30"]["inputs"]["image"]))

    def test_end_frame_makes_it_first_last(self):
        out = self.h.handler(self.payload(end_image_base64=base64.b64encode(b"png2").decode()))
        self.assertEqual(out["mode"], "flf2va")
        self.assertIn("last_frame", self.sent[0]["40"]["inputs"])

    def test_turbo_defaults_to_8_steps(self):
        d = self.payload(turbo=True)
        del d["input"]["steps"]
        out = self.h.handler(d)
        self.assertEqual(out["steps"], 8)
        self.assertIn("199", self.sent[0])

    def test_gui_default_8_steps_turns_turbo_on(self):
        """The Wan GUI sends steps=8 by default; plain H3 at 8 steps is badly
        under-sampled, so auto mode adds the turbo LoRA the template pairs with
        low step counts."""
        out = self.h.handler(self.payload())            # GUI_PAYLOAD carries steps 8
        g = self.sent[0]
        self.assertIn("199", g, "turbo LoRA chained in")
        self.assertEqual(g["199"]["inputs"]["lora_name"], h3.DEFAULTS["turbo_lora"])
        self.assertEqual(g["199"]["inputs"]["model"], ["101", 0], "after the user's LoRAs")
        self.assertEqual(g["43"]["inputs"]["model"], ["199", 0])
        self.assertEqual(out["turbo_reason"], "auto: 8 steps")

    def test_enough_steps_or_explicit_off_keeps_turbo_out(self):
        out = self.h.handler(self.payload(steps=20))
        self.assertNotIn("199", self.sent[-1])
        self.assertFalse(out["turbo"])
        out = self.h.handler(self.payload(steps=6, turbo=False))
        self.assertNotIn("199", self.sent[-1], "explicit false is respected")
        self.assertEqual(out["turbo_reason"], "disabled")
        out = self.h.handler(self.payload(steps=6, turbo="off"))
        self.assertNotIn("199", self.sent[-1], "string flags from form fields work too")

    def test_own_distill_lora_is_not_doubled(self):
        os.rename(os.path.join(self.models, "loras", "style_a.safetensors"),
                  os.path.join(self.models, "loras", "my_h3_lightning_4step.safetensors"))
        out = self.h.handler(self.payload(lora_pairs=[{"high": "my_h3_lightning_4step.safetensors", "high_weight": 1}]))
        self.assertNotIn("error", out, out)
        self.assertNotIn("199", self.sent[-1])
        self.assertIn("already using", out["turbo_reason"])

    def test_auto_turbo_without_the_file_runs_plain_and_warns(self):
        os.remove(os.path.join(self.models, "loras", h3.DEFAULTS["turbo_lora"]))
        out = self.h.handler(self.payload())
        self.assertNotIn("error", out, out)
        self.assertNotIn("199", self.sent[-1])
        self.assertTrue(any("turbo LoRA" in w for w in out.get("warnings", [])))
        out = self.h.handler(self.payload(turbo=True))
        self.assertIn("turbo LoRA", out["error"], "an explicit request still fails loudly")

    def test_missing_lora_names_whats_available(self):
        out = self.h.handler(self.payload(lora_pairs=[{"high": "nope.safetensors", "high_weight": 1}]))
        self.assertIn("nope.safetensors", out["error"])
        self.assertIn("style_a.safetensors", out["error"])
        self.assertEqual(self.sent, [])

    def test_still_downloading_is_said_plainly(self):
        os.remove(os.path.join(self.models, "vae", h3.DEFAULTS["audio_vae"]))
        with open(self.h.PROGRESS_FILE, "w") as f:
            json.dump({"state": "running", "done": ["a", "b"], "total": 12}, f)
        out = self.h.handler(self.payload())
        self.assertIn("still downloading (2/12", out["error"])

    def test_expand_only_is_refused_clearly_and_expansion_flag_only_warns(self):
        self.assertIn("not part of the MiniMax-H3 build", self.h.handler(self.payload(expand_only=True))["error"])
        out = self.h.handler(self.payload(prompt_expansion={"language": "en"}))
        self.assertIn("prompt_expansion", out["warnings"][0])

    def test_prompt_required_and_staging_cleaned_up(self):
        self.assertIn("'prompt' is required", self.h.handler({"input": {"prompt": "  "}})["error"])
        self.h.handler(self.payload())
        self.assertEqual(os.listdir(self.h.COMFY_INPUT_DIR), [], "per-job input folder removed")

    def test_models_status_job(self):
        out = self.h.handler({"input": {"models_status": True}})["models_status"]
        self.assertEqual(out["default_unet"], "h3_tune_int8.safetensors")
        self.assertIn("style_a.safetensors", out["models"]["loras"])


class OutputCollection(unittest.TestCase):
    def test_savevideo_and_vhs_outputs_are_both_found(self):
        import handler
        hist = {"outputs": {
            "49": {"images": [{"filename": "MiniMax_H3_00001_.mp4", "subfolder": "video", "type": "output"}], "animated": [True]},
            "9": {"images": [{"filename": "preview.png", "subfolder": "", "type": "temp"}]},
            "131": {"gifs": [{"filename": "w.mp4", "fullpath": "/x/w.mp4"}]},
        }}
        names = [i["filename"] for _, i in handler.collect_videos(hist)]
        self.assertEqual(sorted(names), ["MiniMax_H3_00001_.mp4", "w.mp4"])


if __name__ == "__main__":
    unittest.main()
