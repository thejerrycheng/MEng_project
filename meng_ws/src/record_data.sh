#!/bin/bash
# record_data.sh
# This script records the specified topics into a rosbag.
# Make sure that roscore is running and your devices are publishing the topics.

# --- Check and Source ROS Environment ---
if [ -f /opt/ros/noetic/setup.bash ]; then
    source /opt/ros/noetic/setup.bash
else
    echo "[ERROR] ROS environment file /opt/ros/noetic/setup.bash not found."
    exit 1
fi

# --- Check and Source Your Workspace ---
WORKSPACE_SETUP=~/meng_ws/devel/setup.bash
if [ -f "$WORKSPACE_SETUP" ]; then
    source "$WORKSPACE_SETUP"
else
    echo "[ERROR] Workspace setup file $WORKSPACE_SETUP not found. Build your workspace first."
    exit 1
fi

# --- Check that ROS master is running ---
if ! rostopic list >/dev/null 2>&1; then
    echo "[ERROR] ROS master is not running. Please run 'roscore' before executing this script."
    exit 1
fi

# --- Create a Unique Bag File Name ---
BAGFILE="data_$(date +%Y%m%d_%H%M%S).bag"
echo "Recording topics to ${BAGFILE}..."
echo "Press Ctrl+C to stop recording."

# --- Define Topics to Record ---
TOPICS=(
    "/ufactory/joint_states"          # Robot joint states
    "/tf"                             # Transform frames
    "/camera/depth/camera_info"         # Depth camera info
    "/camera/depth/image_rect_raw"      # Depth images
    "/camera/color/image_raw"           # RGB images
    "/camera/extrinsics/depth_to_color" # Extrinsics data
    "/tf_static"                      # Static transforms
)

# --- Start Recording ---
rosbag record -O "${BAGFILE}" "${TOPICS[@]}"
