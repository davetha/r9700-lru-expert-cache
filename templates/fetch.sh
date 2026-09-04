#!/usr/bin/env bash
# Fetch the community "fixed" Qwen chat template (froggeric/Qwen-Fixed-Chat-Templates on
# Hugging Face) into this directory. It is not vendored here — its license is the author's.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
URL=${URL:-https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/resolve/main/chat_template.jinja}
curl -sfL -o "$HERE/qwen_fixed_chat_template.jinja" "$URL"
grep -q "template_version" "$HERE/qwen_fixed_chat_template.jinja"
echo "fetched $(wc -c < "$HERE/qwen_fixed_chat_template.jinja") bytes: $(grep -o 'template_version = "[^"]*"' "$HERE/qwen_fixed_chat_template.jinja")"
