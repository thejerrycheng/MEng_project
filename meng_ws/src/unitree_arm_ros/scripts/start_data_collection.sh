#!/usr/bin/env bash
set -e

# ======================================================
# IRIS ROSBAG DATA COLLECTION SCRIPT
# Records synchronized RGB-D + robot state to external SSD
# Auto-splits bags into fixed-duration chunks
# ======================================================

# -----------------------------
# External SSD Auto-Detection
# -----------------------------
USER_NAME=$(whoami)

SSD_CANDIDATES=(
  "/media/${USER_NAME}/PortableSSD"
  "/media/${USER_NAME}/SSD"
)

SSD_MOUNT=""

for path in "${SSD_CANDIDATES[@]}"; do
  if [[ -d "$path" ]]; then
    SSD_MOUNT="$path"
    break
  fi
done

if [[ -z "$SSD_MOUNT" ]]; then
  echo "❌ ERROR: No external SSD found."
  echo "Tried:"
  for path in "${SSD_CANDIDATES[@]}"; do
    echo "  - $path"
  done
  exit 1
fi

DATA_DIR="${SSD_MOUNT}/rosbag_data"
mkdir -p "$DATA_DIR"

echo "✔ External SSD detected at: $SSD_MOUNT"

# -----------------------------
# Parse arguments
# -----------------------------
BAG_NAME=""
OBSTACLE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -O|--name)
      BAG_NAME="$2"
      shift 2
      ;;
    --obstacle)
      OBSTACLE=true
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: ./record_data.sh -O <NAME> [--obstacle]"
      exit 1
      ;;
  esac
done

# -----------------------------
# Validate name
# -----------------------------
if [[ -z "$BAG_NAME" ]]; then
  echo "❌ ERROR: Please provide a filename using -O <NAME>"
  exit 1
fi

# -----------------------------
# Build filename prefix
# -----------------------------
DATE_TAG=$(date +"%Y%m%d_%H%M%S")

if [[ "$OBSTACLE" == true ]]; then
  FINAL_PREFIX="${BAG_NAME}_obstacle_${DATE_TAG}"
  ENV_TAG="WITH obstacle"
else
  FINAL_PREFIX="${BAG_NAME}_${DATE_TAG}"
  ENV_TAG="NO obstacle"
fi

# -----------------------------
# Display session info
# -----------------------------
echo "=================================================="
echo "   IRIS Data Collection — ROSBAG Recorder"
echo "--------------------------------------------------"
echo " Base Name     : ${BAG_NAME}"
echo " Environment   : ${ENV_TAG}"
echo " Save Directory: ${DATA_DIR}"
echo " File Pattern  : ${FINAL_PREFIX}_#.bag"
echo " Chunk Length  : 100 seconds per bag"
echo " Compression   : LZ4"
echo " Buffer Size   : 4 GB"
echo "--------------------------------------------------"
echo " Press Ctrl+C to stop recording"
echo "=================================================="

cd "$DATA_DIR"

# -----------------------------
# Start recording
# -----------------------------
rosbag record \
  --lz4 \
  --split \
  --duration=100 \
  -b 4096 \
  -O "$FINAL_PREFIX" \
  /joint_commands_calibrated \
  /joint_states_calibrated \
  /tf \
  /tf_static \
  /camera/color/image_rect_raw \
  /camera/color/camera_info \
  /camera/depth/image_rect_raw \
  /camera/depth/camera_info \
  /camera/extrinsics/depth_to_color
