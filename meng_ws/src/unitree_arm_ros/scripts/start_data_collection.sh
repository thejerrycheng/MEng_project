#!/usr/bin/env bash
set -e

DATA_DIR="$HOME/Desktop/MEng_project/rosbag_data"
mkdir -p "$DATA_DIR"

# -----------------------------
# Default prefix
# -----------------------------
PREFIX="bag"   # default if user does not pass -O

# -----------------------------
# Parse arguments
# -----------------------------
while [[ $# -gt 0 ]]; do
  case $1 in
    -O|--prefix)
      PREFIX="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: record_data.sh [-O PREFIX]"
      exit 1
      ;;
  esac
done

# -----------------------------
# Build filename: PREFIX_DATE
# -----------------------------
DATE_TAG=$(date +"%Y%m%d_%H%M%S")
FINAL_NAME="${PREFIX}_${DATE_TAG}"

echo "=========================================="
echo " IRIS Data Collection"
echo " Saving to: $DATA_DIR/${FINAL_NAME}.bag"
echo " Press Ctrl+C to stop recording"
echo "=========================================="

cd "$DATA_DIR"

# -----------------------------
# Record topics
# -----------------------------
rosbag record \
  -O "$FINAL_NAME" \
  --lz4 \
  /arm/command \
  /joint_states \
  /tf \
  /tf_static \
  /camera/color/image_raw \
  /camera/color/camera_info \
  /camera/depth/image_rect_raw \
  /camera/depth/camera_info \
  /camera/extrinsics/depth_to_color
