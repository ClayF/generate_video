#!/usr/bin/env python3
"""Regenerate the *_expand_api.json workflows from the base workflows.

The expanded variants are the base Wan2.2 graphs plus two nodes:

  900  WanVideoPromptGenerator  (ComfyUI-MultiModal-Prompt-Nodes)
       Sees the resized start frame (node 171) and the user's motion prompt and
       writes a full Wan2.2 image-to-video prompt.
  901  GVPromptOutput           (custom_nodes/generate_video_prompt_nodes)
       Passes the expanded prompt to the text encoder and publishes it to the
       job history so handler.py can return it.

Node 135 (WanVideo TextEncode) then takes its positive prompt from 901 instead
of a literal string. Run this after editing new_Wan22_api.json or
new_Wan22_flf2v_api.json so the four workflow files stay in sync:

    python tools/build_expand_workflows.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROMPT_NODE_ID = "900"
PROMPT_OUTPUT_NODE_ID = "901"
TEXT_ENCODE_NODE_ID = "135"
RESIZED_START_IMAGE = ["171", 0]

DEFAULT_LLM_MODEL = "Local: LLM/Qwen3VL-8B-Instruct-Q8_0.gguf"

PAIRS = [
    ("new_Wan22_api.json", "new_Wan22_expand_api.json"),
    ("new_Wan22_flf2v_api.json", "new_Wan22_flf2v_expand_api.json"),
]


def expansion_nodes(user_prompt):
    return {
        PROMPT_NODE_ID: {
            "inputs": {
                "prompt": user_prompt,
                "task_type": "Image-to-Video",
                "target_language": "zh",
                "llm_model": DEFAULT_LLM_MODEL,
                "mmproj": "(Auto-detect)",
                "max_retries": 3,
                "device": "GPU",
                "save_tokens": True,
                "image": list(RESIZED_START_IMAGE),
            },
            "class_type": "WanVideoPromptGenerator",
            "_meta": {"title": "Wan Video Prompt Generator"},
        },
        PROMPT_OUTPUT_NODE_ID: {
            "inputs": {"text": [PROMPT_NODE_ID, 0]},
            "class_type": "GVPromptOutput",
            "_meta": {"title": "Prompt Output (generate_video)"},
        },
    }


def build(base):
    wf = json.loads(json.dumps(base))  # deep copy
    for nid in (PROMPT_NODE_ID, PROMPT_OUTPUT_NODE_ID):
        if nid in wf:
            raise SystemExit(f"node id {nid} already used in base workflow")
    user_prompt = wf[TEXT_ENCODE_NODE_ID]["inputs"]["positive_prompt"]
    wf.update(expansion_nodes(user_prompt))
    wf[TEXT_ENCODE_NODE_ID]["inputs"]["positive_prompt"] = [PROMPT_OUTPUT_NODE_ID, 0]
    return wf


def main(check=False):
    changed = False
    for src, dst in PAIRS:
        with open(os.path.join(ROOT, src), "r", encoding="utf-8") as f:
            base = json.load(f)
        out = json.dumps(build(base), indent=2, ensure_ascii=False) + "\n"
        path = os.path.join(ROOT, dst)
        current = open(path, encoding="utf-8").read() if os.path.exists(path) else None
        if current != out:
            changed = True
            if check:
                print(f"{dst} is out of date")
            else:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(out)
                print(f"wrote {dst}")
    if check and changed:
        sys.exit(1)
    if not changed:
        print("workflows up to date")


if __name__ == "__main__":
    main(check="--check" in sys.argv)
