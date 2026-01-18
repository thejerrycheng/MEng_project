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

# Default Files
ROBOT_XML = "iris.xml"
SCENE_XML = "scene2.xml" # Contains robot + obstacle

NUM_JOINTS = 6

# ==================================================
# 1. Kinematics Helper
# ==================================================
class MujocoDirectKinematics:
    """Helper to compute FK without launching a window."""
    def __init__(self, xml_path):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML not found: {xml_path}")
        
        # Load model. If xml_path uses <include>, MuJoCo handles it relative to cwd
        # We will handle cwd switching in main() to be safe.
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
    Reads the base XML (either iris.xml OR scene2.xml)
    and injects <geom> tags for every trajectory.
    ALSO disables gravity.
    """
    print(f"Generating Visualization from base: {os.path.basename(base_xml_path)}...")
    
    with open(base_xml_path, "r") as f:
        xml_content = f.read()

    # --- Step A: Disable Gravity & Dynamics ---
    # Remove existing option tag to override physics
    if "<option" in xml_content:
        import re
        xml_content = re.sub(r'<option.*?>.*?</option>', '', xml_content, flags=re.DOTALL)
        xml_content = re.sub(r'<option.*?/>', '', xml_content)

    # Inject Frozen Physics option
    frozen_option = '<option gravity="0 0 0" integrator="RK4" timestep="0.01"/>'
    
    # Insert after <mujoco> tag
    if "<mujoco" in xml_content:
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
    
    # Inject into Worldbody
    if "</worldbody>" in xml_content:
        # We append our geoms to the end of the worldbody
        # This preserves whatever obstacle was already there
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
    parser = argparse.ArgumentParser(description="Draw human demo paths with optional obstacle.")
    parser.add_argument("--data_dir", type=str, required=True, help="Data folder")
    parser.add_argument("--obstacle", action="store_true", help="If set, uses scene2.xml (with obstacle). Else iris.xml.")
    args = parser.parse_args()

    # 1. Select Base XML
    xml_filename = SCENE_XML if args.obstacle else ROBOT_XML
    selected_xml_path = os.path.join(ASSETS_DIR, xml_filename)

    if not os.path.exists(selected_xml_path):
        print(f"Error: Could not find {selected_xml_path}")
        exit()

    # 2. Load Data
    episodes = load_episodes(args.data_dir)
    if not episodes:
        print("No data found.")
        exit()

    # 3. Change Directory (Crucial for <include> tags to work)
    # We switch to ASSETS_DIR so "iris.xml" or "meshes/" can be found by MuJoCo
    cwd = os.getcwd()
    os.chdir(ASSETS_DIR)

    try:
        # 4. Init Solver (Computes FK using the selected XML)
        fk_solver = MujocoDirectKinematics(xml_filename)

        # 5. Generate Scene (Injects lines into XML)
        # Note: We pass the full path but since we are in ASSETS_DIR, filename works too.
        viz_xml_string = generate_viz_xml(xml_filename, episodes, fk_solver)
        
        print("\nScene Generated. Loading Viewer...")
        print(f" - Base Scene: {xml_filename}")
        print(" - Physics: FROZEN (Gravity=0)")

        # 6. Load Augmented Model
        model = mujoco.MjModel.from_xml_string(viz_xml_string)
        data = mujoco.MjData(model)

        # 7. Position Robot at Start of First Episode
        if len(episodes) > 0:
            start_joints = episodes[0]['joints'][0]
            data.qpos[:NUM_JOINTS] = start_joints
            mujoco.mj_forward(model, data)

        # 8. Launch Viewer with Hold Logic
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                # Constantly reset position to fight any drift
                if len(episodes) > 0:
                    data.qpos[:NUM_JOINTS] = start_joints
                
                mujoco.mj_forward(model, data)
                viewer.sync()

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
    finally:
        # Always return to original folder
        os.chdir(cwd)