#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer
from tqdm import tqdm

# ==================================================
# Configuration
# ==================================================
MUJOCO_SIM_DIR = "/home/jerry/Desktop/MEng_project/mujoco_sim"
ASSETS_DIR = os.path.join(MUJOCO_SIM_DIR, "assets")
BASE_XML_PATH = os.path.join(ASSETS_DIR, "iris.xml")

NUM_JOINTS = 6

# ==================================================
# 1. Kinematics Helper
# ==================================================
class MujocoDirectKinematics:
    """Helper to compute FK without launching a window."""
    def __init__(self, xml_path):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML not found: {xml_path}")
        
        # Load from path so it handles meshes correctly
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Find EE
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.is_site = True
        if self.ee_id == -1:
            self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")
            self.is_site = False
        if self.ee_id == -1:
            self.ee_id = self.model.nbody - 1
            self.is_site = False

    def get_path(self, joints):
        """Returns (N, 3) XYZ path for a sequence of joints."""
        n = len(joints)
        path = np.zeros((n, 3))
        for i in range(n):
            self.data.qpos[:NUM_JOINTS] = joints[i]
            mujoco.mj_kinematics(self.model, self.data)
            if self.is_site:
                path[i] = self.data.site_xpos[self.ee_id]
            else:
                path[i] = self.data.xpos[self.ee_id]
        return path

# ==================================================
# 2. Scene Generator
# ==================================================
def generate_viz_xml(base_xml_path, episodes, fk_solver):
    """
    Reads the base XML and injects <geom> tags for every trajectory.
    ALSO disables gravity so the robot doesn't fall.
    """
    print("Generating Visualization XML...")
    
    with open(base_xml_path, "r") as f:
        xml_content = f.read()

    # --- Step A: Disable Gravity & Dynamics ---
    # We replace the existing <option> tag or insert a new one if missing
    # This ensures the robot acts like a statue.
    
    # 1. Remove existing option tag if present to avoid conflict
    if "<option" in xml_content:
        import re
        xml_content = re.sub(r'<option.*?>.*?</option>', '', xml_content, flags=re.DOTALL)
        # Also handle self-closing tags
        xml_content = re.sub(r'<option.*?/>', '', xml_content)

    # 2. Inject our "Frozen Physics" option
    # integrator="RK4" timestep="0.01" gravity="0 0 0"
    frozen_option = '<option gravity="0 0 0" integrator="RK4" timestep="0.01"/>'
    
    # Insert inside <mujoco> tag
    if "<mujoco" in xml_content:
        # Simple injection after the opening tag
        first_bracket = xml_content.find(">") + 1
        xml_content = xml_content[:first_bracket] + "\n    " + frozen_option + xml_content[first_bracket:]

    # --- Step B: Create Trajectory Geoms ---
    geoms = []
    stride = 2 
    
    for ep_idx, ep in enumerate(tqdm(episodes, desc="Computing Paths")):
        joints = ep['joints']
        path = fk_solver.get_path(joints)
        
        if len(path) < 2: continue

        # Start (Yellow)
        start = path[0]
        geoms.append(
            f'<geom name="ep{ep_idx}_start" type="sphere" pos="{start[0]} {start[1]} {start[2]}" '
            f'size="0.015" rgba="1 1 0 1" contype="0" conaffinity="0"/>'
        )

        # End (Green)
        end = path[-1]
        geoms.append(
            f'<geom name="ep{ep_idx}_end" type="sphere" pos="{end[0]} {end[1]} {end[2]}" '
            f'size="0.015" rgba="0 1 0 1" contype="0" conaffinity="0"/>'
        )

        # Path (Gray Lines)
        for i in range(0, len(path) - stride, stride):
            p1 = path[i]
            p2 = path[i+stride]
            dist = np.linalg.norm(p2 - p1)
            if dist > 0.001:
                geoms.append(
                    f'<geom type="capsule" fromto="{p1[0]} {p1[1]} {p1[2]} {p2[0]} {p2[1]} {p2[2]}" '
                    f'size="0.002" rgba="0.6 0.6 0.6 0.4" contype="0" conaffinity="0"/>'
                )

    geom_block = "\n".join(geoms)
    
    if "</worldbody>" in xml_content:
        new_xml = xml_content.replace("</worldbody>", f"{geom_block}\n</worldbody>")
    else:
        raise ValueError("Could not find </worldbody> tag in base XML.")
        
    return new_xml

# ==================================================
# 3. Data Loader
# ==================================================
def load_episodes(root_dir):
    print(f"Scanning {root_dir}...")
    episodes = []
    csv_paths = []
    for root, dirs, files in os.walk(root_dir):
        if "joint_states.csv" in files:
            csv_paths.append(os.path.join(root, "joint_states.csv"))
            
    if not csv_paths: return []

    for csv_file in csv_paths:
        try:
            df = pd.read_csv(csv_file)
            cols = [c for c in df.columns if c.startswith("pos_")]
            if len(cols) >= 6:
                episodes.append({
                    "joints": df[cols[:6]].to_numpy()
                })
        except: pass
    return episodes

# ==================================================
# Main
# ==================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw all human demo paths in MuJoCo.")
    parser.add_argument("--data_dir", type=str, required=True, help="Data folder")
    parser.add_argument("--xml_path", type=str, default=BASE_XML_PATH)
    args = parser.parse_args()

    # 1. Load Data
    episodes = load_episodes(args.data_dir)
    if not episodes:
        print("No data found.")
        exit()

    # 2. Init Solver
    fk_solver = MujocoDirectKinematics(args.xml_path)

    # 3. Generate Scene (Now with Gravity Disabled)
    viz_xml_string = generate_viz_xml(args.xml_path, episodes, fk_solver)
    
    print("\nScene Generated. Loading Viewer...")
    print(" - Physics: FROZEN (Gravity=0)")
    print(" - Robot: Positioned at Start of Episode 0")

    # 4. Load Augmented Model (Change dir to find assets)
    cwd = os.getcwd()
    assets_dir = os.path.dirname(args.xml_path)
    
    try:
        os.chdir(assets_dir)
        model = mujoco.MjModel.from_xml_string(viz_xml_string)
    except Exception as e:
        print(f"\nCRITICAL ERROR LOADING XML: {e}")
        exit()
    finally:
        os.chdir(cwd)

    data = mujoco.MjData(model)

    # 5. Position Robot at Start of First Episode
    if len(episodes) > 0:
        start_joints = episodes[0]['joints'][0]
        data.qpos[:NUM_JOINTS] = start_joints
        
        # Forward kinematics once to place everything
        mujoco.mj_forward(model, data)

    # 6. Launch with Physics Paused logic
    # We use launch_passive and manually sync to prevent physics integration drift
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Force the joints to stay at the start position every frame
            # This fights any residual drift
            if len(episodes) > 0:
                data.qpos[:NUM_JOINTS] = start_joints
                
            # We do NOT call mj_step here, only mj_forward (kinematics only)
            mujoco.mj_forward(model, data)
            viewer.sync()