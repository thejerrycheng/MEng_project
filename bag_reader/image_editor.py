import rosbag
from sensor_msgs.msg import Image, CameraInfo, JointState
from cv_bridge import CvBridge
import cv2
import argparse
import os
import numpy as np
import pandas as pd
import json
import ast
import tkinter as tk
from tkinter import simpledialog
from scipy.interpolate import interp1d

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

def read_joint_states_from_rosbag(bag_file, topic):
    timestamps = []
    positions = []
    velocities = []
    torques = []
    
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[topic]):
            try:
                timestamps.append(t.to_sec())
                positions.append(list(msg.position))
                velocities.append(list(msg.velocity))
                torques.append(list(msg.effort))
            except Exception as e:
                print(f"Error processing joint state message: {e}")
    
    if len(timestamps) == 0:
        print("Warning: No joint state messages found in ROS bag.")
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    return np.array(timestamps), np.array(positions, dtype=float), np.array(velocities, dtype=float), np.array(torques, dtype=float)

def interpolate_joint_states(image_timestamps, joint_timestamps, positions, velocities, torques):
    if len(joint_timestamps) == 0:
        print("Warning: No joint states available. Using empty values instead of zeros.")
        return [{"position": [], "velocity": [], "effort": []} for _ in range(len(image_timestamps))]
    
    num_joints = positions.shape[1]
    interp_funcs = {
        "position": [interp1d(joint_timestamps, positions[:, i], kind='linear', fill_value='extrapolate') for i in range(num_joints)],
        "velocity": [interp1d(joint_timestamps, velocities[:, i], kind='linear', fill_value='extrapolate') for i in range(num_joints)],
        "effort": [interp1d(joint_timestamps, torques[:, i], kind='linear', fill_value='extrapolate') for i in range(num_joints)]
    }
    
    interpolated_joint_states = []
    for timestamp in image_timestamps:
        interp_joint_state = {
            "position": [float(func(timestamp)) for func in interp_funcs["position"]] if num_joints > 0 else [],
            "velocity": [float(func(timestamp)) for func in interp_funcs["velocity"]] if num_joints > 0 else [],
            "effort": [float(func(timestamp)) for func in interp_funcs["effort"]] if num_joints > 0 else []
        }
        interpolated_joint_states.append(interp_joint_state)
    
    return interpolated_joint_states

def normalize_depth_image(image):
    normalized_image = np.clip(image, 0, 10000)
    normalized_image = (normalized_image / 10000.0) * 255
    normalized_image = np.uint8(normalized_image)
    return cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2BGR)

def display_images(depth_images, color_images, index):    
    color_image = color_images[index].copy()
    target_size = (600, 400)
    color_image = cv2.resize(color_image, target_size, interpolation=cv2.INTER_AREA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    text_color = (255, 255, 255)
    thickness = 2
    position = (10, 30)
    cv2.putText(color_image, f"RGB Frame: {index}", position, font, font_scale, text_color, thickness, cv2.LINE_AA)
    return color_image

def save_clip(depth_images, color_images, joint_states, start_index, end_index, bag_file):
    # Extract the rosbag name (without extension) for the folder structure
    bag_name = os.path.splitext(os.path.basename(bag_file))[0]
    # Create the base folder for this rosbag's clips
    base_clip_folder = os.path.join("saved_clips", bag_name)
    os.makedirs(base_clip_folder, exist_ok=True)
    
    # Create a folder for this particular clip using start and end frame numbers
    clip_path = os.path.join(base_clip_folder, f"clip_{start_index}_to_{end_index}")
    os.makedirs(clip_path, exist_ok=True)
    
    depth_clip_folder = os.path.join(clip_path, "depth")
    color_clip_folder = os.path.join(clip_path, "rgb")
    joint_states_file = os.path.join(clip_path, "joint_states.json")
    os.makedirs(depth_clip_folder, exist_ok=True)
    os.makedirs(color_clip_folder, exist_ok=True)
    
    for i in range(start_index, end_index + 1):
        cv2.imwrite(os.path.join(depth_clip_folder, f"{i}.png"), normalize_depth_image(depth_images[i]))
        cv2.imwrite(os.path.join(color_clip_folder, f"{i}.png"), color_images[i])
        
    with open(joint_states_file, 'w') as f:
        json.dump(joint_states[start_index:end_index+1], f, indent=4)
    
    print(f"Saved clip from frame {start_index} to {end_index} at {clip_path}")

def main():
    parser = argparse.ArgumentParser(description='Read and display frames from a ROS bag file.')
    parser.add_argument('--bag', required=True, help='Path to the ROS bag file')
    args = parser.parse_args()
    bag_file = args.bag
    
    depth_topic = "/camera/depth/image_rect_raw"
    color_topic = "/camera/color/image_raw"
    joint_topic = "/ufactory/joint_states"
    
    depth_images, depth_timestamps = read_images_from_rosbag(bag_file, depth_topic)
    color_images, color_timestamps = read_images_from_rosbag(bag_file, color_topic)
    joint_timestamps, positions, velocities, torques = read_joint_states_from_rosbag(bag_file, joint_topic)
    interpolated_joint_states = interpolate_joint_states(depth_timestamps, joint_timestamps, positions, velocities, torques)
    
    index = 0
    total_frames = len(depth_images)
    clip_start = None
    
    while True:
        combined_image = display_images(depth_images, color_images, index)
        cv2.imshow('Frame Viewer', combined_image)
        key = cv2.waitKey(0) & 0xFF
        
        if key == 81:
            index = max(0, index - 1)
        elif key == 83:
            index = min(total_frames - 1, index + 1)
        elif key == ord('x'):
            if clip_start is None:
                clip_start = index
                print(f"Beginning of the clip selected at frame {clip_start}")
            else:
                print(f"Clip end selected at frame {index}")
        elif key == 13 and clip_start is not None:
            save_clip(depth_images, color_images, interpolated_joint_states, clip_start, index, bag_file)
            clip_start = None
        elif key == ord('c'):
            clip_start = None
            print("Clip selection cleared.")
        elif key == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
