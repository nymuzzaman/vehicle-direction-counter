#!/usr/bin/env bash
# Downloads the sample videos and trained YOLO weights into ./data
set -euo pipefail

# Repository root (this script lives in <root>/scripts)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/data"
mkdir -p "$DATA"

if ! command -v gdown >/dev/null 2>&1; then
    echo "gdown not found. Install it with: pip install gdown" >&2
    exit 1
fi

# Sample traffic videos
gdown -O "$DATA/vehiclesTraffic1.mp4" "https://drive.google.com/uc?id=1vhWHMxg0pe-FXYOssJ3jsIWkRqcq8cid"
gdown -O "$DATA/vehiclesTraffic2.mov" "https://drive.google.com/uc?id=1qadBd7lgpediafCpL_yedGjQPk-FLK-W"

# Trained YOLO weights
gdown -O "$DATA/vehiclesTraffic.pt" "https://drive.google.com/uc?id=1y-IfToCjRXa3ZdC1JpnKRopC7mcQW-5z"
gdown -O "$DATA/vehiclesTraffic_overtrained.pt" "https://drive.google.com/uc?id=1dajaD2vw32v2pfPNgIW2SP1RgNOe3Krv"

echo "Files downloaded to $DATA"
