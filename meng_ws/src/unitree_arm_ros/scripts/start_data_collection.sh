#!/usr/bin/env bash

set -e

DATA_DIR="$HOME/Desktop/MEng_project/data"
mkdir -p "$DATA_DIR"

DATE_TAG=$(date +"%Y%m%d_%H%M%S")
BAG_NAME="GAG_NAME_${DATE_TAG}.bag"

echo "=========================================="
echo " IRIS Data Collection"
echo " Saving to: $DATA_DIR/$BAG_NAME"
echo " Press Ctrl+C to stop recording"
echo "=========================================="

cd "$DATA_DIR"

rosbag record \
/arm/command \
/joint_states \
/tf \
/tf_static \
/camera/color/image_raw \
/camera/color/camera_info \
/camera/depth/image_rect_raw \
/camera/depth/camera_info \
/camera/depth/color/points \
/camera/extrinsics/depth_to_color \
/diagnostics \
/rosout \
--lz4 \
-O "$BAG_NAME"
