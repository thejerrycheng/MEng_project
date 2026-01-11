import rosbag
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
import cv2
import argparse
import os
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Image utilities
# ------------------------------------------------------------

def read_images_from_rosbag(bag_file, topic):
    bridge = CvBridge()
    images = []
    timestamps = []

    with rosbag.Bag(bag_file, 'r') as bag:
        for _, msg, t in bag.read_messages(topics=[topic]):
            try:
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                images.append(cv_image)
                timestamps.append(t.to_sec())
            except Exception as e:
                print(f"Image conversion error: {e}")

    return images, timestamps


def normalize_depth_image(image):
    """For visualization only"""
    image = np.clip(image, 0, 10000)
    image = (image / 10000.0) * 255
    image = np.uint8(image)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def save_images(images, timestamps, folder, image_type):
    os.makedirs(folder, exist_ok=True)

    for img, ts in zip(images, timestamps):
        filename = os.path.join(folder, f"{ts:.6f}.png")
        if image_type == "depth":
            img = normalize_depth_image(img)
        cv2.imwrite(filename, img)

    print(f"Saved {len(images)} {image_type} images to {folder}")


# ------------------------------------------------------------
# Joint utilities
# ------------------------------------------------------------

def read_joint_topic(bag_file, topic):
    timestamps = []
    positions = []

    with rosbag.Bag(bag_file, 'r') as bag:
        for _, msg, t in bag.read_messages(topics=[topic]):
            timestamps.append(t.to_sec())
            positions.append(list(msg.position))

    return timestamps, positions


def save_joint_csv(timestamps, positions, filename):
    df = pd.DataFrame({
        "timestamp": timestamps,
        "positions": positions
    })
    df.to_csv(filename, index=False)
    print(f"Saved: {filename}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, help="Path to rosbag")
    parser.add_argument("--out", required=True, help="Output folder")
    args = parser.parse_args()

    bag_file = args.bag
    out_dir = args.out

    bag_name = os.path.splitext(os.path.basename(bag_file))[0]

    # ---- Topics for IRIS ----
    rgb_topic       = "/camera/color/image_raw"
    depth_topic     = "/camera/depth/image_rect_raw"   # change to aligned if needed
    joint_state_tp  = "/joint_states"
    joint_cmd_tp    = "/arm/command"

    # ---- Output folders ----
    rgb_out   = os.path.join(out_dir, bag_name, "rgb")
    depth_out= os.path.join(out_dir, bag_name, "depth")
    robot_out= os.path.join(out_dir, bag_name, "robot")

    os.makedirs(robot_out, exist_ok=True)

    # ---- Read data ----
    print("Reading RGB images...")
    rgb_images, rgb_ts = read_images_from_rosbag(bag_file, rgb_topic)

    print("Reading depth images...")
    depth_images, depth_ts = read_images_from_rosbag(bag_file, depth_topic)

    print("Reading joint feedback...")
    js_ts, js_pos = read_joint_topic(bag_file, joint_state_tp)

    print("Reading joint commands...")
    cmd_ts, cmd_pos = read_joint_topic(bag_file, joint_cmd_tp)

    # ---- Save ----
    save_images(rgb_images, rgb_ts, rgb_out, "rgb")
    save_images(depth_images, depth_ts, depth_out, "depth")

    save_joint_csv(js_ts, js_pos, os.path.join(robot_out, "joint_states.csv"))
    save_joint_csv(cmd_ts, cmd_pos, os.path.join(robot_out, "joint_commands.csv"))

    print("\nExtraction complete.")
    print(f"Dataset saved under: {os.path.join(out_dir, bag_name)}")


if __name__ == "__main__":
    main()
