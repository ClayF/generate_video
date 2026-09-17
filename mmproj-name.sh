#!/bin/bash
# mmproj-name <url-or-filename>
# Filename to store a projector under. ComfyUI-MultiModal-Prompt-Nodes only
# treats files named mmproj-*.gguf as projectors (anything else is listed as a
# model, and auto-detect never finds it), but quantisers name them differently:
#   Qwen/…:          mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf            (fine as is)
#   prithivMLmods/…: Qwen3-VL-8B-Instruct-abliterated-v2.mmproj-Q8_0.gguf
#                 -> mmproj-Qwen3-VL-8B-Instruct-abliterated-v2-Q8_0.gguf
# Keeping the model's own name after the prefix is what lets the node's family
# match (qwen3-vl / qwen3vl) pair it with the right model.
base="$(basename "${1%%\?*}")"
if [[ "$base" == mmproj-* ]]; then
    echo "$base"
else
    stripped="$(printf '%s' "$base" | sed -E 's/[._-]?mmproj[._-]?/-/I; s/^-//; s/--+/-/g')"
    echo "mmproj-$stripped"
fi
