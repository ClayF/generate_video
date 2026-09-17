"""Tiny helper nodes for the generate_video RunPod worker.

ComfyUI only exposes data in the job history for nodes that are flagged as
OUTPUT_NODE and return a ``ui`` payload. The Wan Video Prompt Generator from
ComfyUI-MultiModal-Prompt-Nodes returns a plain STRING, so the handler cannot
read the expanded prompt back unless something downstream publishes it.

``GVPromptOutput`` sits between the prompt generator and the text encoder: it
passes the string through unchanged and also records it in the history under
``outputs[<node_id>]["text"]`` where handler.py picks it up.
"""


class GVPromptOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True, "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "generate_video"
    DESCRIPTION = "Passes a string through and publishes it to the job history so the RunPod handler can return it."

    def run(self, text):
        text = "" if text is None else str(text)
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {"GVPromptOutput": GVPromptOutput}
NODE_DISPLAY_NAME_MAPPINGS = {"GVPromptOutput": "Prompt Output (generate_video)"}
