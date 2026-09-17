#!/bin/bash
# Projector file naming for ComfyUI-MultiModal-Prompt-Nodes.
#
# The node only treats files named mmproj-*.gguf as projectors: anything else
# is listed as a *model* and mmproj auto-detect never finds it. Quantisers name
# projectors differently, e.g.
#   Qwen/…:          mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf            (fine as is)
#   prithivMLmods/…: Qwen3-VL-8B-Instruct-abliterated-v2.mmproj-Q8_0.gguf
#                 -> mmproj-Qwen3-VL-8B-Instruct-abliterated-v2-Q8_0.gguf
# Keeping the model's own name after the prefix is what lets the node's family
# match (qwen3-vl / qwen3vl) pair the projector with the right model.
#
# Usage:  mmproj-name <url-or-filename>      prints the normalised name
#         mmproj-name --fix <dir>...         renames mis-named projectors in place
#         source mmproj-name                 defines mmproj_name / mmproj_fix_dir

mmproj_name() {
    local base="$(basename "${1%%\?*}")"
    if [[ "$base" == mmproj-* ]]; then
        echo "$base"
    else
        local stripped="$(printf '%s' "$base" | sed -E 's/[._-]?mmproj[._-]?/-/I; s/^-//; s/--+/-/g')"
        echo "mmproj-$stripped"
    fi
}

mmproj_fix_dir() {  # rename every *mmproj*.gguf that does not start with mmproj-
    local d f new
    for d in "$@"; do
        [ -d "$d" ] || continue
        for f in "$d"/*.gguf; do
            [ -e "$f" ] || continue
            case "$(basename "$f")" in
                mmproj-*) ;;
                *[mM][mM][pP][rR][oO][jJ]*)
                    new="$d/$(mmproj_name "$f")"
                    if [ -e "$new" ]; then rm -f "$f"; echo "mmproj: removed duplicate $(basename "$f") ($(basename "$new") exists)"
                    else mv -f "$f" "$new"; echo "mmproj: renamed $(basename "$f") -> $(basename "$new")"; fi ;;
            esac
        done
    done
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    if [ "$1" = "--fix" ]; then shift; mmproj_fix_dir "$@"
    else mmproj_name "$1"; fi
fi
