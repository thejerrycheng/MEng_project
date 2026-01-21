#!/usr/bin/env bash
set -e

# =============================
# External SSD Configuration
# =============================

SSD_MOUNT="/media/jerry/SSD"
DATA_DIR="$SSD_MOUNT/rosbag_data"

# -----------------------------
# Check SSD is mounted
# -----------------------------
if [[ ! -d "$SSD_MOUNT" ]]; then
  echo "❌ ERROR: External SSD not found at: $SSD_MOUNT"
  exit 1
fi

mkdir -p "$DATA_DIR"

# -----------------------------
# Parse arguments
# -----------------------------
BAG_NAME=""
OBSTACLE=false

while [[ $# -gt 0 ]]; do
  case $1 in
    -O|--name)
      BAG_NAME="$2"
      shift 2
      ;;
    --obstacle)
      OBSTACLE=true
      shift 1
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: ./record_data.sh -O <NAME> [--obstacle]"
      exit 1
      ;;
  esac
done

# -----------------------------
# Check Name provided
# -----------------------------
if [[ -z "$BAG_NAME" ]]; then
  echo "❌ ERROR: You must provide a filename using -O NAME"
  exit 1
fi

# -----------------------------
# Build filename
# -----------------------------
DATE_TAG=$(date +"%Y%m%d_%H%M%S")

# Append 'obstacle' tag to the name if the flag is present
if [[ "$OBSTACLE" == true ]]; then
  FINAL_PREFIX="${BAG_NAME}_obstacle_${DATE_TAG}"
else
  FINAL_PREFIX="${BAG_NAME}_${DATE_TAG}"
fi

echo "=========================================="
echo " IRIS Data Collection (Auto-Chunked)"
echo " Base Name: $BAG_NAME"
if [[ "$OBSTACLE" == true ]]; then
  echo " Environment: WITH obstacle"
else
  echo " Environment: NO obstacle"
fi
echo " Saving to: $DATA_DIR/"
echo " format: ${FINAL_PREFIX}_#.bag"
echo " Chunk length: 100 seconds per bag"
echo " Press Ctrl+C to stop recording"
echo "=========================================="

cd "$DATA_DIR"

# -----------------------------
# Record topics with auto split
# -----------------------------
# Note: -O specifies the base name. 
# --split automatically appends _0, _1, etc. to the end.

rosbag record \
  --lz4 \
  --split \
  --duration=100 \
  -O "$FINAL_PREFIX" \
  /joint_commands_calibrated \
  /joint_states_calibrated \
  /tf \
  /tf_static \
  /camera/color/image_raw \
  /camera/color/camera_info \
  /camera/depth/image_rect_raw \
  /camera/depth/camera_info \
  /camera/extrinsics/depth_to_color \