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
GOAL=""
OBSTACLE=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --goal)
      GOAL="$2_$3_$4"
      shift 4
      ;;
    --obstacle)
      OBSTACLE=true
      shift 1
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: record_data.sh --goal X Y Z [--obstacle]"
      exit 1
      ;;
  esac
done

# -----------------------------
# Check goal provided
# -----------------------------
if [[ -z "$GOAL" ]]; then
  echo "❌ ERROR: Goal must be provided."
  exit 1
fi

# -----------------------------
# Build filename prefix
# -----------------------------
DATE_TAG=$(date +"%Y%m%d_%H%M%S")

if [[ "$OBSTACLE" == true ]]; then
  FINAL_NAME="${GOAL}_goal_obstacle_${DATE_TAG}"
else
  FINAL_NAME="${GOAL}_goal_${DATE_TAG}"
fi

echo "=========================================="
echo " IRIS Data Collection (Auto-Chunked)"
echo " Goal (end-effector target): $GOAL"
if [[ "$OBSTACLE" == true ]]; then
  echo " Environment: WITH obstacle"
else
  echo " Environment: NO obstacle"
fi
echo " Saving to: $DATA_DIR/"
echo " File prefix: ${FINAL_NAME}_#.bag"
echo " Chunk length: 100 seconds per bag"
echo " Press Ctrl+C to stop recording"
echo "=========================================="

cd "$DATA_DIR"

# -----------------------------
# Record topics with auto split
# -----------------------------
rosbag record \
  --lz4 \
  --split \
  --duration=100 \
  -O "$FINAL_NAME" \
  /arm/command \
  /joint_states \
  /tf \
  /tf_static \
  /camera/color/image_raw \
  /camera/color/camera_info \
  /camera/depth/image_rect_raw \
  /camera/depth/camera_info \
  /camera/extrinsics/depth_to_color \
