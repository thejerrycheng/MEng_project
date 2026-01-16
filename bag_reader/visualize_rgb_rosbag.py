#!/usr/bin/env python
"""
visualize_rgb_rosbag.py

Reads a ROS bag file and visualizes only the RGB images.

Controls:
- LEFT Arrow or 'A': Previous frame
- RIGHT Arrow or 'D': Next frame
- ESC or 'Q': Quit
"""

import os
import sys
import argparse
import rosbag_reader
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    """
    Parses command-line arguments for specifying the ROS bag file.
    Automatically looks for the bag file in the parent directory of `bag_reader`.
    """
    parser = argparse.ArgumentParser(description="Visualize RGB images from a ROS bag.")
    parser.add_argument(
        "--bag",
        type=str,
        required=True,
        help="Name of the ROS bag file (located in the same level folder as bag_reader)."
    )

    args = parser.parse_args()

    # Get the absolute path to the bag file
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Current script directory
    base_dir = os.path.abspath(os.path.join(script_dir, "../../"))  # Parent directory (same level as bag files)
    bag_file = os.path.join(base_dir, args.bag)

    if not os.path.exists(bag_file):
        print(f"[ERROR] Bag file not found: {bag_file}")
        sys.exit(1)

    return bag_file

def load_rgb_images_lazy(bag_file):
    """
    Generator function to read only RGB images from the ROS bag lazily.
    Prevents memory overflow by loading only one frame at a time.
    """
    bag = rosbag_reader.Bag(bag_file, 'r')
    topic = "/camera/color/image_raw"
    
    for _, msg, _ in bag.read_messages(topics=[topic]):
        yield msg

    bag.close()

def ros_image_to_numpy(msg):
    """
    Converts a ROS image message to a NumPy array safely.
    """
    try:
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return img
    except Exception as e:
        print(f"[ERROR] Failed to convert ROS image: {e}")
        return None

def visualize_rgb_images(bag_file):
    """
    Displays RGB images from the ROS bag using Matplotlib.
    Supports navigation with LEFT/RIGHT arrow keys.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    frame_gen = load_rgb_images_lazy(bag_file)
    frame_list = []
    
    try:
        frame = next(frame_gen)  # Load the first frame
    except StopIteration:
        print("[ERROR] No RGB images found in the bag file.")
        return

    def update_frame():
        """
        Updates the displayed frame when navigating.
        """
        nonlocal frame
        try:
            img = ros_image_to_numpy(frame)
            if img is None:
                img = np.zeros((480, 640, 3), dtype=np.uint8)  # Placeholder for errors
            
            ax.clear()
            ax.imshow(img)
            ax.set_title(f"RGB Image | Frame {len(frame_list) + 1}")
            ax.axis("off")
            plt.draw()
        except Exception as e:
            print(f"[ERROR] Failed to update frame: {e}")

    def on_key(event):
        """
        Handles keypress events for navigation.
        """
        nonlocal frame
        try:
            if event.key in ["right", "d"]:  # Next frame
                frame_list.append(frame)  # Store previous frame in history
                frame = next(frame_gen)  # Get next frame
                update_frame()
            elif event.key in ["left", "a"] and frame_list:  # Previous frame
                frame = frame_list.pop()  # Restore last frame
                update_frame()
            elif event.key in ["escape", "q"]:
                plt.close()
        except StopIteration:
            print("[INFO] No more frames.")

    update_frame()
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

def main():
    bag_file = parse_args()
    print("Opening bag file:", bag_file)
    visualize_rgb_images(bag_file)

if __name__ == '__main__':
    main()
