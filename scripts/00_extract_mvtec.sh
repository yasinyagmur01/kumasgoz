#!/usr/bin/env bash
# Extract MVTec AD into data/mvtec (Anomalib layout: <root>/<category>/...).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_ARCHIVE="${HOME}/Downloads/mvtec_anomaly_detection.tar.xz"
ARCHIVE="${1:-$DEFAULT_ARCHIVE}"
CATEGORY="${2:-carpet}"
DEST="${ROOT_DIR}/data/mvtec"

if [[ ! -f "$ARCHIVE" ]]; then
  # Fallback: common filename typo / variants
  for candidate in \
    "${HOME}/Downloads/mvtec_anomaly_detection.ion.tar.xz" \
    "${HOME}/Downloads/mvtec_anomaly_detection.tar.xz" \
    "${HOME}/downloads/mvtec_anomaly_detection.tar.xz"; do
    if [[ -f "$candidate" ]]; then
      ARCHIVE="$candidate"
      break
    fi
  done
fi

if [[ ! -f "$ARCHIVE" ]]; then
  echo "ERROR: MVTec archive not found. Pass path as first argument." >&2
  exit 1
fi

echo "Archive: $ARCHIVE"
echo "Category: $CATEGORY"
echo "Destination: $DEST"
mkdir -p "$DEST"

# Archive root is category folders (carpet/, bottle/, ...), not mvtec_anomaly_detection/
if [[ -d "${DEST}/${CATEGORY}/train/good" ]]; then
  echo "Already extracted: ${DEST}/${CATEGORY}"
  exit 0
fi

echo "Extracting ${CATEGORY}/ ..."
tar -xJf "$ARCHIVE" -C "$DEST" "${CATEGORY}"

if [[ ! -d "${DEST}/${CATEGORY}/train/good" ]]; then
  echo "ERROR: Expected ${DEST}/${CATEGORY}/train/good after extract." >&2
  exit 1
fi

echo "OK: $(find "${DEST}/${CATEGORY}/train/good" -type f | wc -l) train/good images"
echo "OK: $(find "${DEST}/${CATEGORY}/test" -type f | wc -l) test images"
