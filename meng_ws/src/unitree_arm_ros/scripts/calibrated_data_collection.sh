#!/usr/bin/env bash
set -e

# =============================
# External SSD Configuration
# =============================
SSD_MOUNT="/media/jerry/SSD"
DATA_DIR="$SSD_MOUNT/rosbag_data"

# =============================
# ROS Calibration Node
# =============================
CALIB_NODE="unitree_arm_ros calibrate_joint_states.py"

# -----------------------------
# Check SSD is mounted
# -----------------------------
if [[ ! -d "$SSD_MOUNT" ]]; then
  echo "❌ ERROR: External SSD not found at: $SSD_MOUNT"
  exit 1
fi

mkdir -p "$DATA_DIR"

# -----------------------------
# Default prefix
# -----------------------------
PREFIX="bag"

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
# Build filename
# -----------------------------
DATE_TAG=$(date +"%Y%m%d_%H%M%S")
FINAL_NAME="${PREFIX}_${DATE_TAG}"

echo "=========================================="
echo " IRIS Data Collection (Calibrated Joints)"
echo " Saving to: $DATA_DIR"
echo " File name: ${FINAL_NAME}_#.bag"
echo "=========================================="

# -----------------------------
# Start Calibration Relay Node
# -----------------------------
echo "▶ Starting joint calibration relay..."
rosrun unitree_arm_ros calibrate_joint_states.py &
CALIB_PID=$!

# Allow ROS node to initialize
sleep 2

# Ensure calibration node stops on exit
cleanup() {
    echo ""
    echo "⏹ Stopping calibration relay..."
    kill $CALIB_PID 2>/dev/null || true
}
trap cleanup EXIT

# -----------------------------
# Start Recording
# -----------------------------
cd "$DATA_DIR"

echo "▶ Recording rosbag..."
rosbag record \
  --lz4 \
  --split \
  --duration=100 \
  -O "$FINAL_NAME" \
  /arm/command \
  /joint_states_calibrated \
  /tf \
  /tf_static \
  /camera/color/image_raw \
  /camera/color/camera_info \
  /camera/depth/image_rect_raw \
  /camera/depth/camera_info \
  /camera/extrinsics/depth_to_color
