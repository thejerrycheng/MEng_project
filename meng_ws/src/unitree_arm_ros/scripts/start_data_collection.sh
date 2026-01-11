#!/usr/bin/env bash
set -e

DATA_DIR="$HOME/Desktop/MEng_project/rosbag_data"
mkdir -p "$DATA_DIR"

# -----------------------------
# Parse arguments
# -----------------------------
GOAL=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --goal)
      GOAL="$2_$3_$4"
      shift 4
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: record_data.sh --goal X Y Z"
      exit 1
      ;;
  esac
done

# -----------------------------
# Check goal provided
# -----------------------------
if [[ -z "$GOAL" ]]; then
  echo "❌ ERROR: Goal must be provided."
  echo "Usage: record_data.sh --goal X Y Z"
  exit 1
fi

# -----------------------------
# Build filename: GOAL_DATE
# -----------------------------
DATE_TAG=$(date +"%Y%m%d_%H%M%S")
FINAL_NAME="${GOAL}_goal_${DATE_TAG}"

echo "=========================================="
echo " IRIS Data Collection"
echo " Goal (end-effector target): $GOAL"
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
