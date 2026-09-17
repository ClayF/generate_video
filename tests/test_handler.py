"""Offline tests for handler.py (no ComfyUI / RunPod needed).

Run with:  python -m unittest discover -s tests -v
"""
import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _stub_runtime_modules():
    """handler.py imports runpod and websocket at module level; stub them."""
    if "runpod" not in sys.modules:
        runpod = types.ModuleType("runpod")
        serverless = types.ModuleType("runpod.serverless")
        utils = types.ModuleType("runpod.serverless.utils")
        utils.rp_upload = types.SimpleNamespace()
        serverless.utils = utils
        serverless.start = lambda *a, **k: None
        runpod.serverless = serverless
        sys.modules["runpod"] = runpod
        sys.modules["runpod.serverless"] = serverless
        sys.modules["runpod.serverless.utils"] = utils
    if "websocket" not in sys.modules:
        sys.modules["websocket"] = types.SimpleNamespace(WebSocket=object)


_stub_runtime_modules()
os.environ["WORKFLOW_DIR"] = ROOT
sys.path.insert(0, ROOT)
handler = importlib.import_module("handler")


def links_in(graph):
    for nid, node in graph.items():
        for k, v in node["inputs"].items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                yield nid, k, v[0]


class PromptExpansionSettings(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.llm_dir = os.path.join(self.tmp.name, "LLM")
        os.makedirs(self.llm_dir)
        for name in ("Qwen3VL-8B-Instruct-Q8_0.gguf", "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf", "other-model.gguf"):
            open(os.path.join(self.llm_dir, name), "wb").close()
        self.node_dir = os.path.join(self.tmp.name, "mm-node")
        os.makedirs(self.node_dir)
        self.patches = [
            mock.patch.object(handler, "LLM_DIR", self.llm_dir),
            mock.patch.object(handler, "VOLUME_LLM_DIR", os.path.join(self.tmp.name, "no-volume")),
            mock.patch.object(handler, "MULTIMODAL_NODE_DIR", self.node_dir),
            mock.patch.dict(os.environ, {}, clear=False),
        ]
        for p in self.patches:
            p.start()
        for k in ("PROMPT_EXPANSION_MODEL", "PROMPT_EXPANSION_LANGUAGE", "PROMPT_EXPANSION_DEVICE"):
            os.environ.pop(k, None)

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_off_by_default(self):
        self.assertIsNone(handler.resolve_prompt_expansion({"prompt": "x"}))
        self.assertIsNone(handler.resolve_prompt_expansion({"prompt": "x", "prompt_expansion": False}))
        self.assertIsNone(handler.resolve_prompt_expansion({"prompt": "x", "prompt_expansion": {"enabled": False}}))

    def test_bool_and_shorthand_use_bundled_model(self):
        for job in ({"prompt_expansion": True}, {"expand_prompt": True}, {"expand_only": True}):
            e = handler.resolve_prompt_expansion(job)
            self.assertEqual(e["model"], "Local: LLM/Qwen3VL-8B-Instruct-Q8_0.gguf")
            self.assertEqual(e["language"], "zh")
            self.assertEqual(e["device"], "GPU")
            self.assertEqual(e["mmproj"], "(Auto-detect)")

    def test_local_discovery_skips_mmproj_and_non_qwen(self):
        self.assertEqual(handler.list_local_llm_models(self.llm_dir), ["LLM/Qwen3VL-8B-Instruct-Q8_0.gguf"])

    def test_volume_models_are_listed_by_absolute_path_like_the_node(self):
        # outside the models dir, like a real /runpod-volume mount
        vol_root = tempfile.TemporaryDirectory(); self.addCleanup(vol_root.cleanup)
        vol = os.path.join(vol_root.name, "LLM")
        os.makedirs(vol)
        open(os.path.join(vol, "Qwen3VL-4B-Instruct-Q4_K_M.gguf"), "wb").close()
        with mock.patch.object(handler, "VOLUME_LLM_DIR", vol):
            names = handler.list_local_llm_models()
            self.assertEqual(names, [f"{vol}/Qwen3VL-4B-Instruct-Q4_K_M.gguf", "LLM/Qwen3VL-8B-Instruct-Q8_0.gguf"])
            # a bare filename resolves to wherever the file actually is
            self.assertEqual(handler.normalize_llm_model("Qwen3VL-4B-Instruct-Q4_K_M.gguf"), f"Local: {vol}/Qwen3VL-4B-Instruct-Q4_K_M.gguf")
            self.assertEqual(handler.normalize_llm_model("Qwen3VL-8B-Instruct-Q8_0.gguf"), "Local: LLM/Qwen3VL-8B-Instruct-Q8_0.gguf")
        # image dir empty, only the volume has a model -> it becomes the default
        for f in os.listdir(self.llm_dir):
            os.remove(os.path.join(self.llm_dir, f))
        with mock.patch.object(handler, "VOLUME_LLM_DIR", vol):
            self.assertEqual(handler.default_llm_model(), f"Local: {vol}/Qwen3VL-4B-Instruct-Q4_K_M.gguf")

    def test_object_form_and_normalisation(self):
        e = handler.resolve_prompt_expansion({"prompt_expansion": {
            "language": "EN", "device": "cpu", "model": "Qwen3VL-8B-Instruct-Q8_0.gguf", "max_retries": 99}})
        self.assertEqual(e["language"], "en")
        self.assertEqual(e["device"], "CPU")
        self.assertEqual(e["model"], "Local: LLM/Qwen3VL-8B-Instruct-Q8_0.gguf")
        self.assertEqual(e["max_retries"], 10)
        self.assertEqual(handler.normalize_llm_model("Local: LLM/x.gguf"), "Local: LLM/x.gguf")
        self.assertEqual(handler.normalize_llm_model("sub/x.gguf"), "Local: sub/x.gguf")
        self.assertEqual(handler.normalize_llm_model("/runpod-volume/LLM/x.gguf"), "Local: /runpod-volume/LLM/x.gguf")
        self.assertEqual(handler.normalize_llm_model("unknown.gguf"), "Local: LLM/unknown.gguf")
        self.assertEqual(handler.normalize_llm_model("qwen3.7-plus"), "qwen3.7-plus")

    def test_env_overrides(self):
        os.environ["PROMPT_EXPANSION_LANGUAGE"] = "en"
        os.environ["PROMPT_EXPANSION_DEVICE"] = "CPU"
        os.environ["PROMPT_EXPANSION_MODEL"] = "MyQwen.gguf"
        e = handler.resolve_prompt_expansion({"prompt_expansion": True})
        self.assertEqual((e["language"], e["device"], e["model"]), ("en", "CPU", "Local: LLM/MyQwen.gguf"))

    def test_invalid_values(self):
        with self.assertRaises(handler.JobError):
            handler.resolve_prompt_expansion({"prompt_expansion": {"language": "fr"}})
        with self.assertRaises(handler.JobError):
            handler.resolve_prompt_expansion({"prompt_expansion": {"device": "TPU"}})
        with self.assertRaises(handler.JobError):
            handler.resolve_prompt_expansion({"prompt_expansion": "yes"})

    def test_cloud_model_needs_api_key(self):
        with self.assertRaises(handler.JobError):
            handler.resolve_prompt_expansion({"prompt_expansion": {"model": "qwen3.7-plus"}})
        with open(os.path.join(self.node_dir, "api_key.txt"), "w") as f:
            f.write("sk-test\n")
        e = handler.resolve_prompt_expansion({"prompt_expansion": {"model": "qwen3.7-plus"}})
        self.assertEqual(e["model"], "qwen3.7-plus")

    def test_no_local_model_is_a_clear_error(self):
        with mock.patch.object(handler, "LLM_DIR", os.path.join(self.tmp.name, "missing")):
            with self.assertRaises(handler.JobError) as cm:
                handler.resolve_prompt_expansion({"prompt_expansion": True})
        self.assertIn("No local prompt model", str(cm.exception))


class WorkflowBuilding(unittest.TestCase):
    BASE = {"prompt": "the camera pushes in", "seed": 7, "cfg": 2.5, "width": 500, "height": 830,
            "length": 81, "context_overlap": 48,
            "lora_pairs": [{"high": "a_high.safetensors", "low": "a_low.safetensors", "high_weight": 0.8, "low_weight": 0.9}]}
    EXP = {"language": "zh", "model": "Local: LLM/Qwen3VL-8B-Instruct-Q8_0.gguf", "mmproj": "(Auto-detect)",
           "device": "GPU", "max_retries": 3, "save_tokens": True}

    def test_plain_graph_matches_previous_behaviour(self):
        g = handler.build_workflow(self.BASE, "/tmp/in.jpg", None)
        self.assertEqual(len(g), 27)
        self.assertNotIn("900", g)
        self.assertEqual(g["135"]["inputs"]["positive_prompt"], "the camera pushes in")
        self.assertEqual(g["244"]["inputs"]["image"], "/tmp/in.jpg")
        self.assertEqual((g["235"]["inputs"]["value"], g["236"]["inputs"]["value"]), (496, 832))
        self.assertEqual(g["220"]["inputs"]["seed"], 7)
        self.assertEqual(g["540"]["inputs"]["cfg"], 2.5)
        self.assertEqual(g["279"]["inputs"]["lora_1"], "a_high.safetensors")
        self.assertEqual(g["553"]["inputs"]["strength_1"], 0.9)
        self.assertEqual(g["498"]["inputs"]["context_frames"], 81)

    def test_defaults_when_optional_fields_missing(self):
        g = handler.build_workflow({"prompt": "p"}, "/tmp/in.jpg", None)
        self.assertEqual((g["235"]["inputs"]["value"], g["236"]["inputs"]["value"]), (480, 832))
        self.assertEqual(g["220"]["inputs"]["seed"], 42)
        self.assertEqual(g["540"]["inputs"]["cfg"], 2.0)
        with self.assertRaises(handler.JobError):
            handler.build_workflow({}, "/tmp/in.jpg", None)

    def test_expansion_wiring(self):
        g = handler.build_workflow(self.BASE, "/tmp/in.jpg", None, self.EXP)
        self.assertEqual(len(g), 29)
        self.assertEqual(g["900"]["class_type"], "WanVideoPromptGenerator")
        self.assertEqual(g["901"]["class_type"], "GVPromptOutput")
        self.assertEqual(g["900"]["inputs"]["prompt"], "the camera pushes in")
        self.assertEqual(g["900"]["inputs"]["image"], ["171", 0])
        self.assertEqual(g["900"]["inputs"]["task_type"], "Image-to-Video")
        self.assertEqual(g["900"]["inputs"]["llm_model"], self.EXP["model"])
        self.assertEqual(g["901"]["inputs"]["text"], ["900", 0])
        self.assertEqual(g["135"]["inputs"]["positive_prompt"], ["901", 0])
        # the raw user prompt must not leak into the encoder alongside the link
        self.assertNotIn("the camera pushes in", json.dumps(g["135"]))

    def test_flf2v_with_expansion(self):
        g = handler.build_workflow(self.BASE, "/tmp/in.jpg", "/tmp/end.jpg", self.EXP)
        self.assertEqual(len(g), 31)
        self.assertEqual(g["617"]["inputs"]["image"], "/tmp/end.jpg")
        self.assertEqual(g["135"]["inputs"]["positive_prompt"], ["901", 0])

    def test_expand_only_graph_is_self_contained(self):
        g = handler.expand_only_graph(handler.build_workflow(self.BASE, "/tmp/in.jpg", "/tmp/end.jpg", self.EXP))
        self.assertEqual(set(g), {"244", "171", "235", "236", "900", "901"})
        for nid, key, target in links_in(g):
            self.assertIn(target, g, f"{nid}.{key} points outside the expand-only graph")
        self.assertNotIn("122", g)  # no diffusion model loads
        self.assertNotIn("617", g)  # end frame is irrelevant for expansion

    def test_all_links_resolve_in_every_workflow(self):
        for name in handler.WORKFLOW_FILES.values():
            with open(os.path.join(ROOT, name)) as f:
                g = json.load(f)
            for nid, key, target in links_in(g):
                self.assertIn(target, g, f"{name}: {nid}.{key} -> {target}")

    def test_expand_workflows_are_generated_from_base(self):
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        build = importlib.import_module("build_expand_workflows")
        for src, dst in build.PAIRS:
            with open(os.path.join(ROOT, src)) as f:
                base = json.load(f)
            with open(os.path.join(ROOT, dst)) as f:
                self.assertEqual(build.build(base), json.load(f), dst)


class Helpers(unittest.TestCase):
    def test_strip_data_url_prefix(self):
        self.assertEqual(handler.strip_data_url_prefix("data:image/png;base64,AAAA"), "AAAA")
        self.assertEqual(handler.strip_data_url_prefix("AAAA"), "AAAA")

    def test_validation_error_summary(self):
        body = json.dumps({"error": {"message": "Prompt outputs failed validation"},
                           "node_errors": {"900": {"class_type": "WanVideoPromptGenerator", "errors": [
                               {"message": "Value not in list", "details": "llm_model: 'Local: nope.gguf' not in [...]"}]}}})
        s = handler.summarize_validation_error(body)
        self.assertIn("node 900 (WanVideoPromptGenerator)", s)
        self.assertIn("Value not in list", s)

    def test_prompt_output_node(self):
        sys.path.insert(0, os.path.join(ROOT, "custom_nodes"))
        mod = importlib.import_module("generate_video_prompt_nodes")
        node = mod.NODE_CLASS_MAPPINGS["GVPromptOutput"]()
        out = node.run("expanded text")
        self.assertEqual(out, {"ui": {"text": ["expanded text"]}, "result": ("expanded text",)})
        self.assertTrue(mod.GVPromptOutput.OUTPUT_NODE)


@unittest.skipUnless(os.environ.get("MULTIMODAL_NODE_SRC"), "set MULTIMODAL_NODE_SRC=/path/to/ComfyUI-MultiModal-Prompt-Nodes")
class NodeContract(unittest.TestCase):
    """Check node 900's inputs against the real WanVideoPromptGenerator.INPUT_TYPES."""

    def test_inputs_match_node_definition(self):
        src = os.environ["MULTIMODAL_NODE_SRC"]
        fake_fp = types.SimpleNamespace(models_dir="/models",
                                        get_folder_paths=lambda k: (_ for _ in ()).throw(KeyError(k)))
        stubs = {"dashscope": types.SimpleNamespace(base_http_api_url=""),
                 "folder_paths": fake_fp,
                 "comfy_execution": types.ModuleType("comfy_execution"),
                 "comfy_execution.graph": types.SimpleNamespace(ExecutionBlocker=object)}
        with mock.patch.dict(sys.modules, stubs):
            sys.path.insert(0, src)
            wan_nodes = importlib.import_module("wan_nodes")
            with mock.patch.object(wan_nodes, "discover_local_gguf_models", return_value=["LLM/Qwen3VL-8B-Instruct-Q8_0.gguf"]), \
                 mock.patch.object(wan_nodes, "discover_local_mmproj_files", return_value=["LLM/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"]):
                spec = wan_nodes.WanVideoPromptGenerator.INPUT_TYPES()
        allowed = {**spec["required"], **spec.get("optional", {})}
        g = handler.build_workflow({"prompt": "p"}, "/x.jpg", None, WorkflowBuilding.EXP)
        node = g["900"]["inputs"]
        self.assertEqual(set(spec["required"]), set(node) - {"image"})
        for key, value in node.items():
            self.assertIn(key, allowed)
            kind = allowed[key][0]
            if isinstance(kind, list) and not isinstance(value, list):
                self.assertIn(value, kind, key)
        self.assertEqual(wan_nodes.WanVideoPromptGenerator.RETURN_TYPES, ("STRING",))


class FakeWS:
    def __init__(self, messages):
        self.messages = list(messages)

    def recv(self):
        return json.dumps(self.messages.pop(0))


class RunWorkflow(unittest.TestCase):
    def test_collects_video_text_and_error(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"\x00\x01")
            vid = f.name
        history = {"pid": {"status": {"status_str": "success"}, "outputs": {
            "131": {"gifs": [{"fullpath": vid}]},
            "901": {"text": ["expanded!"]}}}}
        ws = FakeWS([{"type": "executing", "data": {"node": "900", "prompt_id": "pid"}},
                     {"type": "executing", "data": {"node": None, "prompt_id": "pid"}}])
        with mock.patch.object(handler, "queue_prompt", return_value={"prompt_id": "pid"}), \
             mock.patch.object(handler, "get_history", return_value=history):
            videos, texts, error = handler.run_workflow(ws, {})
        self.assertIsNone(error)
        self.assertEqual(texts, {"901": ["expanded!"]})
        self.assertEqual(videos, {"131": ["AAE="]})
        os.unlink(vid)

    def test_execution_error_is_reported(self):
        ws = FakeWS([{"type": "execution_error", "data": {"prompt_id": "pid", "node_id": "900",
                                                        "node_type": "WanVideoPromptGenerator",
                                                        "exception_type": "RuntimeError",
                                                        "exception_message": "mmproj not specified"}},
                     {"type": "executing", "data": {"node": None, "prompt_id": "pid"}}])
        history = {"pid": {"status": {"status_str": "error"}, "outputs": {}}}
        with mock.patch.object(handler, "queue_prompt", return_value={"prompt_id": "pid"}), \
             mock.patch.object(handler, "get_history", return_value=history):
            videos, texts, error = handler.run_workflow(ws, {})
        self.assertEqual(videos, {})
        self.assertIn("mmproj not specified", error)


class HandlerEndToEnd(unittest.TestCase):
    """Drive handler() with ComfyUI mocked out."""

    def _run(self, job_input, texts, videos=None, error=None):
        with mock.patch.object(handler, "wait_for_comfyui", return_value=mock.Mock()), \
             mock.patch.object(handler, "run_workflow", return_value=(videos or {}, texts, error)) as rw, \
             mock.patch.object(handler, "resolve_prompt_expansion", wraps=handler.resolve_prompt_expansion), \
             mock.patch.object(handler, "default_llm_model", return_value="Local: LLM/Qwen3VL-8B-Instruct-Q8_0.gguf"):
            out = handler.handler({"input": job_input})
            graph = rw.call_args[0][1]
        return out, graph

    def test_expand_only_returns_text_without_video(self):
        out, graph = self._run({"prompt": "walk", "image_path": "/x.jpg", "expand_only": True},
                               texts={"901": ["a long prompt"]})
        self.assertEqual(out, {"expanded_prompt": "a long prompt", "prompt": "walk"})
        self.assertEqual(set(graph), {"244", "171", "235", "236", "900", "901"})

    def test_generation_with_expansion_returns_both(self):
        out, graph = self._run({"prompt": "walk", "image_path": "/x.jpg", "prompt_expansion": {"language": "en"}},
                               texts={"901": ["a long prompt"]}, videos={"131": ["QUJD"]})
        self.assertEqual(out, {"video": "QUJD", "expanded_prompt": "a long prompt", "prompt": "walk"})
        self.assertEqual(graph["900"]["inputs"]["target_language"], "en")

    def test_generation_without_expansion_is_unchanged(self):
        out, graph = self._run({"prompt": "walk", "image_path": "/x.jpg"}, texts={}, videos={"131": ["QUJD"]})
        self.assertEqual(out, {"video": "QUJD"})
        self.assertNotIn("900", graph)

    def test_errors_surface_in_output(self):
        out, _ = self._run({"prompt": "walk", "image_path": "/x.jpg", "prompt_expansion": True},
                           texts={}, error="node 900 (WanVideoPromptGenerator): RuntimeError boom")
        self.assertIn("boom", out["error"])
        out = handler.handler({"input": {"image_path": "/x.jpg"}})
        self.assertIn("'prompt' is required", out["error"])


if __name__ == "__main__":
    unittest.main()
