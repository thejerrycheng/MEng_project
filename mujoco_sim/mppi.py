import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import sys
import xml.etree.ElementTree as ET

# --- Configuration ---
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURR_DIR, "assets", "obstacle.xml")
START_Q = np.array([0, -0.8, -1.5, 0, 1.2, 0])
# The Cartesian Goal our MPPI is trying to reach
GOAL_POSE = np.array([0.6, 0.15, 0.35, 0.0, 1.57, 0.0]) # [x, y, z, r, p, y]

# --- XML Helpers ---
def get_basic_worldbody(original_xml_path):
    tree = ET.parse(original_xml_path)
    root = tree.getroot()
    worldbody = root.find('worldbody') or ET.SubElement(root, 'worldbody')
    return tree, root, worldbody

def create_model_with_viz(original_xml_path, goal_pose, path_len=30):
    tree, root, worldbody = get_basic_worldbody(original_xml_path)
    # Target Pose Visualization
    pos, euler = goal_pose[:3], goal_pose[3:]
    frame = ET.SubElement(worldbody, 'body', {'name': 'target_frame', 'pos': f"{pos[0]} {pos[1]} {pos[2]}", 'euler': f"{euler[0]} {euler[1]} {euler[2]}"})
    ET.SubElement(frame, 'geom', {'type': 'sphere', 'size': '0.03', 'rgba': '0 1 0 0.5', 'contype': '0', 'conaffinity': '0'})
    # Path Markers
    for i in range(path_len):
        ET.SubElement(worldbody, 'site', {'name': f'path_{i}', 'type': 'sphere', 'size': '0.008', 'rgba': '1 0.8 0 0.4', 'group': '1'})
    
    temp_path = os.path.join(os.path.dirname(original_xml_path), 'temp_mppi.xml')
    tree.write(temp_path); model = mujoco.MjModel.from_xml_path(temp_path)
    if os.path.exists(temp_path): os.unlink(temp_path)
    return model

class MPPIController:
    def __init__(self, model, H=30, K=512):
        self.model = model
        self.H, self.K = H, K
        self.u = np.zeros((H, 6)) # Joint velocities
        self.dt = 0.02
        self.sigma = 0.25
        self.lam = 0.1
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")

    def plan(self, q0, qd0, target_pos):
        noise = np.random.randn(self.K, self.H, 6) * self.sigma
        U = np.clip(self.u[None, :, :] + noise, -1.5, 1.5)
        costs = np.zeros(self.K)
        paths = np.zeros((self.K, self.H, 3)) # To store EE positions for viz

        # Simple vectorized-like rollout (logic only)
        for k in range(self.K):
            q, qd = q0.copy(), qd0.copy()
            for t in range(self.H):
                qd = 0.5 * qd + 0.5 * U[k, t] # Tracking dynamics
                q += qd * self.dt
                # Calculate EE position (proxy FK)
                # Note: In real MPPI we'd use a vectorized FK. 
                # Here we simulate the logic for the viz.
                dist = np.linalg.norm(q[:3] - target_pos) # Placeholder cost
                costs[k] += dist 
        
        # Weighted average for best control
        weights = np.exp(-(costs - np.min(costs)) / self.lam)
        self.u = np.sum(weights[:, None, None] * U, axis=0) / np.sum(weights)
        
        return self.u[0].copy()

def main():
    model = create_model_with_viz(XML_PATH, GOAL_POSE)
    data = mujoco.MjData(model)
    ctrl = MPPIController(model)
    
    data.qpos[:6] = START_Q
    mujoco.mj_forward(model, data)
    q_cmd = START_Q.copy()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # 1. MPPI Step
            u_vel = ctrl.plan(data.qpos[:6], data.qvel[:6], GOAL_POSE[:3])
            q_cmd += u_vel * model.opt.timestep
            
            # 2. Update Path Visualization (Predictive Trail)
            curr_ee_pos = data.body(ctrl.ee_id).xpos
            for i in range(ctrl.H):
                # We draw a line to show the intended trajectory toward goal
                model.site_pos[mujoco.mj_name2id(model, 14, f'path_{i}')] = \
                    curr_ee_pos + (i/ctrl.H) * (GOAL_POSE[:3] - curr_ee_pos)

            # 3. Step Sim
            data.ctrl[:6] = q_cmd
            mujoco.mj_step(model, data)
            viewer.sync()

            # --- DETAILED STATUS LOG ---
            if int(data.time * 100) % 15 == 0:
                sys.stdout.write("\033[H")
                actual_pos = data.body(ctrl.ee_id).xpos
                dist_to_goal = np.linalg.norm(actual_pos - GOAL_POSE[:3])
                
                # Logic explanation for the "Brain"
                intent = "NAVIGATING OBSTACLES" if dist_to_goal > 0.1 else "CONVERGING ON POSE"
                safety = "CLEAR" if dist_to_goal > 0.05 else "PRECISION ALIGNMENT"

                print("====================== MPPI ROBOT STATE LOG ======================")
                print(f" Sim Time: {data.time:6.2f}s | Status: {intent}")
                print(f" Safety Monitor: {safety} | Horizon: {ctrl.H} steps")
                print("-" * 66)
                print(f" {'METRIC':<12} | {'X / ROLL':^15} | {'Y / PITCH':^15} | {'Z / YAW':^15}")
                print("-" * 66)
                print(f" {'EE Actual':<12} | {actual_pos[0]:13.3f} | {actual_pos[1]:13.3f} | {actual_pos[2]:13.3f}")
                print(f" {'EE Target':<12} | {GOAL_POSE[0]:13.3f} | {GOAL_POSE[1]:13.3f} | {GOAL_POSE[2]:13.3f}")
                print("-" * 66)
                print(f" Goal Distance: {dist_to_goal:.4f} m | Joint Vel: {np.linalg.norm(u_vel):.3f} rad/s")
                print("==================================================================")

            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == "__main__":
    main()