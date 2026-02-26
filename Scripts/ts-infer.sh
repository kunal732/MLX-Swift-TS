#!/bin/bash
# ts-infer: Build and run the TSInfer CLI tool for testing MLX time series models.
#
# Usage:
#   ./Scripts/ts-infer.sh --hf-path kunal732/Toto-Open-Base-1.0-MLX
#   ./Scripts/ts-infer.sh --hf-path kunal732/chronos-t5-base-mlx --prediction-length 16
#   ./Scripts/ts-infer.sh --mlx-path ./mlx_model

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Building TSInfer..."
xcodebuild -scheme TSInfer \
  -destination 'platform=macOS,arch=arm64' \
  -configuration Debug \
  build \
  -quiet 2>&1 | grep -E "error:|BUILD SUCCEEDED|BUILD FAILED" || true

# Find the built binary
BINARY=$(find "$HOME/Library/Developer/Xcode/DerivedData" \
  -name "TSInfer" -type f \
  -path "*/Debug/TSInfer" 2>/dev/null | head -1)

if [ -z "$BINARY" ]; then
  echo "Error: TSInfer binary not found after build"
  exit 1
fi

echo "Running: $BINARY $@"
echo ""
"$BINARY" "$@"
