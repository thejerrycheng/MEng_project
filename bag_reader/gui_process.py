#!/usr/bin/env python3
import rosbag
from sensor_msgs.msg import Image, JointState
import cv2
import argparse
import os
import numpy as np
import pandas as pd
import json

try:
    from scipy.interpolate import interp1d
    HAS_SCIPY = True
except:
    HAS_SCIPY = False


# ------------------------------------------------------------
# MANUAL CONVERTER (Replaces CvBridge)
# ------------------------------------------------------------
def msg_to_numpy(msg):
    """
    Manually converts ROS Image message to numpy array to avoid
    conda/system GLIBCXX conflicts with cv_bridge.
    """
    try:
        if "8UC1" in msg.encoding or "mono8" in msg.encoding:
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        
        elif "16UC1" in msg.encoding or "mono16" in msg.encoding:
            return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        
        elif "bgr8" in msg.encoding:
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        
        elif "rgb8" in msg.encoding:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
        else:
            # Fallback for generic 8-bit, 3-channel
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            
    except Exception as e:
        print(f"Manual conversion error on {msg.encoding}: {e}")
        return None


# ------------------------------------------------------------
# ROSBAG READERS
# ------------------------------------------------------------

def read_images_from_rosbag(bag_file, topic):
    images, timestamps = [], []

    print(f"Reading {bag_file}...")
    
    with rosbag.Bag(bag_file, 'r') as bag:
        for _, msg, t in bag.read_messages(topics=[topic]):
            
            # Use manual numpy conversion instead of cv_bridge
            cv_image = msg_to_numpy(msg)
            
            if cv_image is not None:
                images.append(cv_image)
                timestamps.append(t.to_sec())

    return images, np.asarray(timestamps, dtype=float)


def read_joint_states_from_rosbag(bag_file, topic):
    ts, pos, vel, eff = [], [], [], []
    names = None

    with rosbag.Bag(bag_file, 'r') as bag:
        for _, msg, t in bag.read_messages(topics=[topic]):
            ts.append(t.to_sec())
            if names is None:
                names = list(msg.name) if msg.name else []
            pos.append(list(msg.position))
            vel.append(list(msg.velocity))
            eff.append(list(msg.effort))

    if len(ts) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), []

    ts = np.asarray(ts, dtype=float)
    pos = np.asarray(pos, dtype=float)
    vel = np.asarray(vel, dtype=float)
    eff = np.asarray(eff, dtype=float)

    if not names:
        names = [f"joint_{i+1}" for i in range(pos.shape[1])]

    return ts, pos, vel, eff, names


# ------------------------------------------------------------
# ALIGN RGB & DEPTH
# ------------------------------------------------------------

def find_temporal_offset(t1, t2):
    best, offset = 1e9, 0
    # Search start of arrays
    for i in range(min(50, len(t1))):
        for j in range(min(50, len(t2))):
            d = abs(t2[j] - t1[i])
            if d < best:
                best, offset = d, i - j
    return offset


def temporal_align(offset, a1, a2):
    if offset > 0:
        a1 = a1[offset:]
    elif offset < 0:
        a2 = a2[-offset:]
    L = min(len(a1), len(a2))
    return a1[:L], a2[:L]


# ------------------------------------------------------------
# ROBOT INTERPOLATION
# ------------------------------------------------------------

def interpolate_robot(query_ts, joint_ts, pos, vel, eff):
    if len(joint_ts) == 0 or pos.size == 0:
        return None

    t_min = joint_ts[0]
    t_max = joint_ts[-1]
    query_ts = np.clip(query_ts, t_min, t_max)

    order = np.argsort(joint_ts)
    jt = joint_ts[order]
    pos = pos[order]

    has_vel = vel.size > 0
    has_eff = eff.size > 0
    if has_vel: vel = vel[order]
    if has_eff: eff = eff[order]

    M = len(query_ts)
    J = pos.shape[1]

    pos_out = np.zeros((M, J))
    vel_out = np.zeros((M, J)) if has_vel else None
    eff_out = np.zeros((M, J)) if has_eff else None

    for j in range(J):
        pos_out[:, j] = np.interp(query_ts, jt, pos[:, j])
        if has_vel:
            vel_out[:, j] = np.interp(query_ts, jt, vel[:, j])
        if has_eff:
            eff_out[:, j] = np.interp(query_ts, jt, eff[:, j])

    return pos_out, vel_out, eff_out


# ------------------------------------------------------------
# UI DRAWING
# ------------------------------------------------------------

def normalize_depth(img):
    # Simple depth normalization for visualization
    img = np.clip(img, 0, 3000) # clip at 3 meters
    img = (img / 3000.0) * 255
    img = np.uint8(img)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def draw_joint_hud(image, joint_values, title):
    h, w, _ = image.shape
    panel_x, panel_y = 15, 50
    panel_w, panel_h = 260, 190
    alpha = 0.55

    overlay = image.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    cv2.putText(image, title, (panel_x + 10, panel_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for i, val in enumerate(joint_values):
        if np.isnan(val) or np.isinf(val):
            txt = f"J{i+1}:   --.- deg"
        else:
            txt = f"J{i+1}: {val:6.1f} deg"
        cv2.putText(image, txt, (panel_x + 10, panel_y + 55 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_preview(depth, rgb, s, e, rgb_ts, joint_ts, joint_pos, joint_vel, joint_eff, joint_names):
    # Ensure indices are safe
    s = min(max(0, s), len(rgb)-1)
    e = min(max(0, e), len(rgb)-1)

    d1 = normalize_depth(depth[s].copy())
    d2 = normalize_depth(depth[e].copy())
    c1 = rgb[s].copy()
    c2 = rgb[e].copy()

    target = (640, 420)
    d1 = cv2.resize(d1, target)
    d2 = cv2.resize(d2, target)
    c1 = cv2.resize(c1, target)
    c2 = cv2.resize(c2, target)

    ts_query = np.array([rgb_ts[s], rgb_ts[e]])
    interp = interpolate_robot(ts_query, joint_ts, joint_pos, joint_vel, joint_eff)

    if interp is not None:
        pos_interp = interp[0]
        start_j = np.degrees(pos_interp[0])
        end_j   = np.degrees(pos_interp[1])
    else:
        start_j = [0]*6
        end_j   = [0]*6

    cv2.putText(d1, f"Depth START {s}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),2)
    cv2.putText(d2, f"Depth END   {e}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),2)
    cv2.putText(c1, f"RGB START   {s}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),2)
    cv2.putText(c2, f"RGB END     {e}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),2)

    draw_joint_hud(c1, start_j, "JOINTS @ START")
    draw_joint_hud(c2, end_j,   "JOINTS @ END")

    top = cv2.hconcat([d1, d2])
    bot = cv2.hconcat([c1, c2])
    return cv2.vconcat([top, bot])


def play(rgb, s, e):
    for i in range(s, e + 1):
        cv2.imshow("Playback", rgb[i])
        if cv2.waitKey(30) & 0xFF == 27:
            break


# ------------------------------------------------------------
# SAVE EPISODE
# ------------------------------------------------------------

def save_episode(ep_idx, bag_name, out_root, rgb, depth, rgb_ts, joint_ts, joint_pos, joint_vel, joint_eff, joint_names, s, e):
    ep_name = f"{bag_name}_episode_{ep_idx:04d}"
    ep_dir = os.path.join(out_root, ep_name)

    rgb_dir = os.path.join(ep_dir, "rgb")
    depth_dir = os.path.join(ep_dir, "depth")
    robot_dir = os.path.join(ep_dir, "robot")

    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(robot_dir, exist_ok=True)

    saved_ts = []
    frame_id = 1
    for i in range(s, e + 1):
        cv2.imwrite(os.path.join(rgb_dir, f"frame_{frame_id:04d}.png"), rgb[i])
        # Save raw 16-bit depth
        cv2.imwrite(os.path.join(depth_dir, f"frame_{frame_id:04d}.png"), depth[i])
        saved_ts.append(rgb_ts[i])
        frame_id += 1

    saved_ts = np.asarray(saved_ts)
    interp = interpolate_robot(saved_ts, joint_ts, joint_pos, joint_vel, joint_eff)
    df = pd.DataFrame({"timestamp": saved_ts})

    if interp is not None:
        pos_i, vel_i, eff_i = interp
        for j, name in enumerate(joint_names):
            df[f"pos_{name}"] = pos_i[:, j]
            if vel_i is not None: df[f"vel_{name}"] = vel_i[:, j]
            if eff_i is not None: df[f"eff_{name}"] = eff_i[:, j]

    df.to_csv(os.path.join(robot_dir, "joint_states.csv"), index=False)

    meta = {
        "start_index": int(s),
        "end_index": int(e),
        "num_frames": int(e - s + 1),
        "t_start": float(saved_ts[0]),
        "t_end": float(saved_ts[-1])
    }

    with open(os.path.join(ep_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=4)

    print(f"✅ Saved {ep_name}")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    bag_file = args.bag
    out_root = args.out
    bag_name = os.path.splitext(os.path.basename(bag_file))[0]

    rgb_topic = "/camera/color/image_raw"
    depth_topic = "/camera/depth/image_rect_raw"
    joint_topic = "/joint_states_calibrated"

    print("Loading RGB...")
    rgb, rgb_ts = read_images_from_rosbag(bag_file, rgb_topic)
    print(f"RGB frames loaded: {len(rgb)}")

    print("Loading Depth...")
    depth, depth_ts = read_images_from_rosbag(bag_file, depth_topic)
    print(f"Depth frames loaded: {len(depth)}")

    print("Loading Joint States...")
    joint_ts, joint_pos, joint_vel, joint_eff, joint_names = read_joint_states_from_rosbag(bag_file, joint_topic)
    print(f"Joint state messages loaded: {len(joint_ts)}")

    # Safety check if loading failed
    if len(rgb) == 0 or len(depth) == 0:
        print("❌ Error: No images loaded. Please check bag file content or topic names.")
        return

    off = find_temporal_offset(depth_ts, rgb_ts)
    depth, rgb = temporal_align(off, depth, rgb)
    depth_ts, rgb_ts = temporal_align(off, depth_ts, rgb_ts)

    print(f"Frames loaded (aligned): {len(rgb)}")
    if len(rgb) < 2:
        return

    s, e = 0, 1
    ep_counter = 1

    print("\nControls:")
    print("← → : Move START frame")
    print("↑ ↓ : Move END frame")
    print("Enter : Save episode")
    print("p : Playback")
    print("ESC : Exit\n")

    while True:
        canvas = draw_preview(depth, rgb, s, e, rgb_ts, joint_ts, joint_pos, joint_vel, joint_eff, joint_names)
        cv2.imshow("IRIS Episode Cutter", canvas)
        key = cv2.waitKey(0)

        # Arrow keys
        if key == 81: # Left
            s = max(0, s - 1)
            e = max(s + 1, e)
        elif key == 83: # Right
            s = min(len(rgb) - 2, s + 1)
            e = max(s + 1, e)
        elif key == 82: # Up
            e = min(len(rgb) - 1, e + 1)
        elif key == 84: # Down
            e = max(s + 1, e - 1)
        elif key == ord('p'):
            play(rgb, s, e)
        elif key == 13: # Enter
            save_episode(ep_counter, bag_name, out_root, rgb, depth, rgb_ts, joint_ts, joint_pos, joint_vel, joint_eff, joint_names, s, e)
            ep_counter += 1
            s = min(e + 1, len(rgb) - 2)
            e = min(s + 1, len(rgb) - 1)
        elif key == 27: # Esc
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()