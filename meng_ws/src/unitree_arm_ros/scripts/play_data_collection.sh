#!/usr/bin/env bash
set -e

DATA_DIR="$HOME/Desktop/MEng_project/data"

# --- Bag selection ---
if [ "$1" == "--bag" ]; then
    BAG_PATH="$2"
else
    BAG_PATH=$(ls -t $DATA_DIR/GAG_NAME_*.bag | head -n 1)
fi

if [ ! -f "$BAG_PATH" ]; then
    echo "Bag not found: $BAG_PATH"
    exit 1
fi

echo "Playing bag: $BAG_PATH"

# --- Use ROS simulated time ---
rosparam set /use_sim_time true

# --- Start image viewer ---
rosrun image_view image_view image:=/camera/color/image_raw &

IMG_PID=$!
sleep 1

# --- Play rosbag ---
rosbag play "$BAG_PATH" --clock

# --- Cleanup ---
kill $IMG_PID 2>/dev/null || true
echo "Done."
