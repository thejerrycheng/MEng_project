import rosbag
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import argparse
import os
import numpy as np
import pandas as pd
import json
import tkinter as tk
from tkinter import simpledialog


def read_timestamps(bag_file, topic):
    """Read timestamps from a ros topic."""
    
    timestamps = []
    
    with rosbag.Bag(bag_file, 'r') as bag:        
        for topic, _, t in bag.read_messages(topics=[topic]):
            timestamps.append(t.to_sec())
                
    return timestamps


def temporal_align(offset, a1, a2):
    if offset > 0:
        a1 = a1[offset:]
    else:
        a2 = a2[-offset:]
        
    length = min(len(a1), len(a2))
    
    return a1[:length], a2[:length]
        

def read_images_from_rosbag(bag_file, topic):
    bridge = CvBridge()
    images = []
    timestamps = []
    n = 0
    
    with rosbag.Bag(bag_file, 'r') as bag:        
        for topic, msg, t in bag.read_messages(topics=[topic]):
            try:
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                images.append(cv_image)
                timestamps.append(t.to_sec())
                n += 1
            except Exception as e:
                print(f"Error converting image: {e}")
                
        # print(n)
    return images, timestamps


def find_temporal_offset(t1, t2):
    """Temporally align two lists of timestamps and return the best
    index offset between t1 and t2
    """
        
    offset = -1
    best_diff = np.inf
            
    for i in range(min(20, len(t1))):
        for j in range(min(20, len(t2))):
            if best_diff > abs(t2[j] - t1[i]):
                best_diff = abs(t2[j] - t1[i])
                offset = i - j
                
    return offset


def read_camera_info_from_rosbag(bag_file, topic):
    camera_info = []
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[topic]):
            info = {
                'header': msg.header.stamp.to_sec(),
                'width': msg.width,
                'height': msg.height,
                'distortion_model': msg.distortion_model,
                'D': msg.D,
                'K': msg.K,
                'R': msg.R,
                'P': msg.P,
                'binning_x': msg.binning_x,
                'binning_y': msg.binning_y,
                'roi': [msg.roi.x_offset, msg.roi.y_offset, msg.roi.height, msg.roi.width, msg.roi.do_rectify]
            }
            camera_info.append(info)
    return camera_info

def normalize_depth_image(image):
    """*For display only*"""
    # Normalize the depth image to the range [0, 255] assuming min is 0 and max is 20000
    normalized_image = np.clip(image, 0, 10000)  # Clip values to the range [0, 20000]
    normalized_image = (normalized_image / 10000.0) * 255  # Scale to the range [0, 255]
    # Convert to uint8
    normalized_image = np.uint8(normalized_image)
    # Convert grayscale to RGB
    normalized_image_rgb = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)
    return normalized_image_rgb

def resize_images(images, target_size):
    return [cv2.resize(image, target_size, interpolation=cv2.INTER_AREA) for image in images]

def display_images(depth_images, color_images, start_index, end_index):    
    depth_start_image = normalize_depth_image(depth_images[start_index].copy())
    depth_end_image = normalize_depth_image(depth_images[end_index].copy())
    color_start_image = color_images[start_index].copy()
    color_end_image = color_images[end_index].copy()
        
    # Determine the minimum dimensions of all images
    target_size = (600, 400)

    # Resize all images to the minimum dimensions
    depth_start_image = cv2.resize(depth_start_image, target_size, interpolation=cv2.INTER_AREA)
    depth_end_image = cv2.resize(depth_end_image, target_size, interpolation=cv2.INTER_AREA)
    color_start_image = cv2.resize(color_start_image, target_size, interpolation=cv2.INTER_AREA)
    color_end_image = cv2.resize(color_end_image, target_size, interpolation=cv2.INTER_AREA)
    
    # Text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    text_color = (255, 255, 255)  # White text
    thickness = 2

    # Positions for the text
    position = (10, 30)

    # Add white text to the images
    cv2.putText(depth_start_image, f"Depth Start Frame: {start_index}", position, font, font_scale, text_color, thickness, cv2.LINE_AA)
    cv2.putText(depth_end_image, f"Depth End Frame: {end_index}", position, font, font_scale, text_color, thickness, cv2.LINE_AA)
    cv2.putText(color_start_image, f"RGB Start Frame: {start_index}", position, font, font_scale, text_color, thickness, cv2.LINE_AA)
    cv2.putText(color_end_image, f"RGB End Frame: {end_index}", position, font, font_scale, text_color, thickness, cv2.LINE_AA)

    # Concatenate images to create the final display
    top_row = cv2.hconcat([depth_start_image, depth_end_image])
    bottom_row = cv2.hconcat([color_start_image, color_end_image])
    combined_image = cv2.vconcat([top_row, bottom_row])
    
    return combined_image

def play_video(images, start_index, end_index):
    for i in range(start_index, end_index + 1):
        normalized_image = normalize_depth_image(images[i])
        cv2.imshow('Video Playback', normalized_image)
        if cv2.waitKey(int(1000 / 30)) & 0xFF == ord('x'):
            break

def save_images_and_info(depth_images, color_images, start_index, end_index, bag_file, out_folder, color_camera_info):
    root = tk.Tk()
    root.withdraw()
    suffix = simpledialog.askstring("Input", "Enter suffix for folder name:")
    
    if not suffix:
        print("Suffix is required to save images.")
        return

    base_folder = os.path.join(out_folder, f"{os.path.splitext(os.path.basename(bag_file))[0]}-{suffix}")
    depth_folder = os.path.join(base_folder, "depth")
    rgb_folder = os.path.join(base_folder, "rgb")
    os.makedirs(depth_folder, exist_ok=True)
    os.makedirs(rgb_folder, exist_ok=True)
    
    # Save depth and color images
    for i in range(start_index, end_index + 1, 6):
        depth_image = depth_images[i]
        color_image = color_images[i]

        depth_file_name = os.path.join(depth_folder, f"{i}.png")
        color_file_name = os.path.join(rgb_folder, f"{i}.png")

        cv2.imwrite(depth_file_name, depth_image)
        cv2.imwrite(color_file_name, color_image)

        print(f"Saved {depth_file_name}")
        print(f"Saved {color_file_name}")

    # Save camera info
    color_camera_info_file = os.path.join(base_folder, "color_camera_info.json")

    with open(color_camera_info_file, 'w') as f:
        json.dump(color_camera_info[0], f, indent=4)

    print(f"Saved {color_camera_info_file}")

def main():
    parser = argparse.ArgumentParser(description='Read and display frames from a ROS bag file.')
    parser.add_argument('--bag', required=True, help='Path to the ROS bag file')
    parser.add_argument('--out', required=True, help='Parent folder to save the output images')
    args = parser.parse_args()

    bag_file = args.bag
    out_folder = args.out
    # depth_topic = "/camera/depth/image_rect_raw"
    depth_topic = "/camera/aligned_depth_to_color/image_raw"
    color_topic = "/camera/color/image_raw"
    color_info_topic = "/camera/color/camera_info"
    
    depth_images, depth_timestamps = read_images_from_rosbag(bag_file, depth_topic)
    color_images, color_timestamps = read_images_from_rosbag(bag_file, color_topic)
    color_camera_info = read_camera_info_from_rosbag(bag_file, color_info_topic)
    
    print(f"Number of depth frames: {len(depth_images)}")
    print(f"Number of RGB frames: {len(color_images)}")

    offset = find_temporal_offset(depth_timestamps, color_timestamps)
    depth_images, color_images = temporal_align(offset, depth_images, color_images)
    
    print(f"Found best offset from depth to RGB is: {offset}")
    
    start_index = 0
    end_index = len(depth_images) - 1

    try:
        while True:
            combined_image = display_images(depth_images, color_images, start_index, end_index)
            cv2.imshow('Start and End Frames', combined_image)
            key = cv2.waitKey(0) & 0xFF

            if key == ord('t'):
                start_index = max(0, start_index - 10)
            elif key == ord('y'):
                start_index = min(len(depth_images) - 1, start_index + 10)
            elif key == ord('g'):
                start_index = max(0, start_index - 1)
            elif key == ord('h'):
                start_index = min(len(depth_images) - 1, start_index + 1)
            elif key == ord('u'):
                end_index = max(0, end_index - 10)
                end_index = min(end_index, min(len(depth_images), len(color_images)) - 1)
            elif key == ord('i'):
                end_index = min(len(depth_images) - 1, end_index + 10)
                end_index = min(end_index, min(len(depth_images), len(color_images)) - 1)
            elif key == ord('j'):
                end_index = max(0, end_index - 1)
            elif key == ord('k'):
                end_index = min(len(depth_images) - 1, end_index + 1)
                end_index = min(end_index, min(len(depth_images), len(color_images)) - 1)
            elif key == ord('p'):
                play_video(depth_images, start_index, end_index)
            elif key == ord('s'):
                save_images_and_info(depth_images, color_images, start_index, end_index, bag_file, out_folder, color_camera_info)
            elif key == ord('x'):  # Exit
                break

    except KeyboardInterrupt:
        print("Terminating gracefully...")

    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()