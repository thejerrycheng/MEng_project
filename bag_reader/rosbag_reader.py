import rosbag
from sensor_msgs.msg import Image, CameraInfo, JointState
from cv_bridge import CvBridge
import cv2
import argparse
import os
import numpy as np
import pandas as pd

def read_images_from_rosbag(bag_file, topic):
    bridge = CvBridge()
    images = []
    timestamps = []
    
    with rosbag.Bag(bag_file, 'r') as bag:        
        for topic, msg, t in bag.read_messages(topics=[topic]):
            try:
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                images.append(cv_image)
                timestamps.append(t.to_sec())
            except Exception as e:
                print(f"Error converting image: {e}")
    return images, timestamps

def normalize_depth_image(image):
    normalized_image = np.clip(image, 0, 10000)  # Clip values to [0, 10000]
    normalized_image = (normalized_image / 10000.0) * 255  # Scale to [0, 255]
    normalized_image = np.uint8(normalized_image)
    return cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)

def save_images(images, timestamps, folder, image_type):
    os.makedirs(folder, exist_ok=True)
    for i, (image, ts) in enumerate(zip(images, timestamps)):
        filename = os.path.join(folder, f"{ts:.6f}.png")
        if image_type == "depth":
            image = normalize_depth_image(image)
        cv2.imwrite(filename, image)
        print(f"Saved {image_type} image: {filename}")

def read_joint_states_from_rosbag(bag_file, topic):
    timestamps = []
    positions = []
    velocities = []
    torques = []
    
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[topic]):
            try:
                timestamps.append(t.to_sec())
                positions.append(msg.position)
                velocities.append(msg.velocity)
                torques.append(msg.effort)
            except Exception as e:
                print(f"Error processing joint state message: {e}")
    
    return timestamps, positions, velocities, torques

def save_joint_states(timestamps, positions, velocities, torques, folder):
    os.makedirs(folder, exist_ok=True)
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "Position": positions,
        "Velocity": velocities,
        "Torque": torques
    })
    joint_states_file = os.path.join(folder, "joint_states.csv")
    df.to_csv(joint_states_file, index=False)
    print(f"Saved joint states data: {joint_states_file}")

def main():
    parser = argparse.ArgumentParser(description='Extract images and joint states from a ROS bag file.')
    parser.add_argument('--bag_file', required=True, help='Path to the ROS bag file')
    args = parser.parse_args()
    
    bag_name = os.path.splitext(os.path.basename(args.bag_file))[0]
    rgb_folder = f"rgb_data/{bag_name}_rgb"
    depth_folder = f"depth_data/{bag_name}"
    joint_folder = f"robot_data/{bag_name}"
    
    rgb_topic = "/camera/color/image_raw"
    depth_topic = "/camera/depth/image_rect_raw"
    joint_topic = "/ufactory/joint_states"
    
    rgb_images, rgb_timestamps = read_images_from_rosbag(args.bag_file, rgb_topic)
    depth_images, depth_timestamps = read_images_from_rosbag(args.bag_file, depth_topic)
    joint_timestamps, positions, velocities, torques = read_joint_states_from_rosbag(args.bag_file, joint_topic)
    
    save_images(rgb_images, rgb_timestamps, rgb_folder, "rgb")
    save_images(depth_images, depth_timestamps, depth_folder, "depth")
    save_joint_states(joint_timestamps, positions, velocities, torques, joint_folder)
    
if __name__ == "__main__":
    main()
