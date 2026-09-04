#!/usr/bin/env bash
# Copy the prebuilt gfx1201 kernels into build/kernels/ (where the launcher expects them).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEST=${1:-$HERE/../build/kernels}
(cd "$HERE" && sha256sum -c SHA256SUMS)
mkdir -p "$DEST"
cp "$HERE"/librlu.so "$HERE"/libhcqfp8sk.so "$HERE"/cold_gather.so "$DEST"/
echo "installed to $DEST"; ls -la "$DEST"
