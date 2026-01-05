import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import sys
import argparse
import csv

# --- Configuration ---
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURR_DIR)
MUJOCO_SIM_DIR = os.path.join(PROJECT_ROOT, "mujoco_sim")
XML_PATH = os.path.join(MUJOCO_SIM_DIR, "assets", "obstacle.xml")
SAVE_DIR = os.path.join(CURR_DIR, "APF_path")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# High-stiffness Tracking Gains
KP, KD = 2800.0, 210.0

class CinemaValidator:
    def __init__(self, model, data):
        self.model, self.data = model, data
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")
        self.obs_names = ["table_obstacle", "pillar_obstacle", "small_block"]
        self.obs_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in self.obs_names]

    def randomize_realistic_scene(self, start_pt, end_pt):
        """Randomizes obstacles with real-world scales and ensures points aren't blocked."""
        obs_meta = {}
        for name, g_id in zip(self.obs_names, self.obs_ids):
            if g_id == -1: continue

            # Realistic scaling: Pillars (7cm wide), Blocks (10cm), Tables (20cm)
            if "pillar" in name:
                # Ensure it is a numpy array
                size = np.array([np.random.uniform(0.03, 0.05), np.random.uniform(0.15, 0.3), 0.0])
            else:
                size = np.random.uniform(0.04, 0.1, size=3)
            
            # Buffer check: Keep obstacles away from the immediate start/end zones
            valid_pos = False
            while not valid_pos:
                pos = np.array([
                    np.random.uniform(0.35, 0.65), 
                    np.random.uniform(-0.4, 0.4), 
                    np.random.uniform(0.1, 0.4)
                ])
                dist_start = np.linalg.norm(pos - start_pt)
                dist_end = np.linalg.norm(pos - end_pt)
                # Ensure at least 18cm distance from start/end points
                if dist_start > 0.18 and dist_end > 0.18:
                    valid_pos = True
            
            self.model.geom_pos[g_id] = pos
            self.model.geom_size[g_id] = size
            obs_meta[name] = {'pos': pos.tolist(), 'size': size.tolist()}
        return obs_meta

    def plan_offline(self, start_pos, end_pos, max_steps=6000):
        """APF Planner with safety distance and convergence check."""
        curr_p = np.array(start_pos)
        path = [curr_p.copy()]
        
        for _ in range(max_steps):
            diff_goal = end_pos - curr_p
            dist_goal = np.linalg.norm(diff_goal)
            f_att = diff_goal / (dist_goal + 1e-6)
            
            f_rep = np.zeros(3)
            for g_id in self.obs_ids:
                diff_obs = curr_p - self.model.geom_pos[g_id]
                dist_obs = np.linalg.norm(diff_obs) - np.max(self.model.geom_size[g_id])
                if dist_obs < 0.2: # Influence radius
                    mag = 0.08 * (1.0/max(dist_obs, 0.01) - 1.0/0.2) * (1.0/max(dist_obs, 0.01)**2)
                    f_rep += (diff_obs / (np.linalg.norm(diff_obs) + 1e-6)) * mag
            
            step = (f_att * 1.0 + f_rep) * 0.004
            curr_p += step
            path.append(curr_p.copy())
            if dist_goal < 0.01: return np.array(path)
        
        return None 

    def get_id_control(self, target_pos, target_obj):
        """Inverse Dynamics for precision tracking + Subject tracking."""
        curr_pos = self.data.body(self.ee_id).xpos
        curr_mat = self.data.body(self.ee_id).xmat.reshape(3, 3)
        
        # Look-At Matrix calculation
        z_axis = (target_obj - curr_pos)
        z_axis /= (np.linalg.norm(z_axis) + 1e-6)
        up = np.array([0, 0, 1])
        if abs(np.dot(z_axis, up)) > 0.99: up = np.array([0, 1, 0])
        x_axis = np.cross(up, z_axis)
        x_axis /= (np.linalg.norm(x_axis) + 1e-6)
        y_axis = np.cross(z_axis, x_axis)
        target_mat = np.stack([x_axis, y_axis, z_axis], axis=1)

        pos_err = target_pos - curr_pos
        rot_err_mat = target_mat @ curr_mat.T
        rot_err_quat = np.zeros(4)
        mujoco.mju_mat2Quat(rot_err_quat, rot_err_mat.flatten())
        rot_err_vec = rot_err_quat[1:] * np.sign(rot_err_quat[0])
        
        cart_err = np.concatenate([pos_err, rot_err_vec])
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_id)
        J = np.vstack([jacp, jacr])[:, :6]
        
        J_inv = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.eye(6))
        q_accel = KP * (J_inv @ cart_err) - KD * self.data.qvel[:6]
        
        self.data.qacc[:6] = q_accel
        mujoco.mj_inverse(self.model, self.data)
        return self.data.qfrc_inverse[:6].copy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument("--mode", choices=['pan', 'dolly', 'crane'], default='dolly')
    parser.add_argument("--duration", type=float, default=7.0)
    args = parser.parse_args()

    presets = {
        'pan': ([0.4, -0.3, 0.45], [0.4, 0.3, 0.45], [0.75, 0.0, 0.2]),
        'dolly': ([0.3, 0.0, 0.4], [0.65, 0.0, 0.4], [0.85, 0.0, 0.4]),
        'crane': ([0.45, 0.0, 0.2], [0.45, 0.0, 0.6], [0.7, 0.0, 0.15])
    }

    iteration = 0
    while iteration < args.num:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
        data = mujoco.MjData(model)
        validator = CinemaValidator(model, data)
        
        start_pt, end_pt, subject = presets[args.mode]
        obs_meta = validator.randomize_realistic_scene(start_pt, end_pt)
        
        # Phase 1: Planning
        full_path = validator.plan_offline(start_pt, end_pt)
        
        if full_path is None:
            print(f"⚠️  Run {iteration}: APF failed (likely blocked). Retrying...")
            continue

        print(f"✅ Run {iteration+1}/{args.num} Planned: {len(full_path)} waypoints.")
        csv_rows = []

        # Phase 2: Execution & Visualization
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while data.time < args.duration and viewer.is_running():
                t_norm = data.time / args.duration
                idx = min(int(t_norm * len(full_path)), len(full_path)-1)
                target_p = full_path[idx]
                
                data.ctrl[:6] = validator.get_id_control(target_p, np.array(subject))
                mujoco.mj_step(model, data)

                if int(data.time/model.opt.timestep) % 8 == 0:
                    viewer.user_scn.ngeom = 0
                    # Draw Subject (Yellow Cube)
                    mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], 
                                        mujoco.mjtGeom.mjGEOM_BOX, [0.03,0.03,0.03], 
                                        subject, np.eye(3).flatten(), [1, 0.8, 0, 1])
                    viewer.user_scn.ngeom += 1
                    # Draw Planned Path (Grey Breadcrumbs)
                    for p in full_path[::25]:
                        mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], 
                                            2, [0.004,0,0], p, np.eye(3).flatten(), [1, 1, 1, 0.1])
                        viewer.user_scn.ngeom += 1
                    # Draw Target (Green Sphere)
                    mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], 
                                        2, [0.015,0,0], target_p, np.eye(3).flatten(), [0, 1, 0, 1])
                    viewer.user_scn.ngeom += 1
                    viewer.sync()

                csv_rows.append({
                    'time': data.time,
                    'target_x': target_p[0], 'target_y': target_p[1], 'target_z': target_p[2],
                    'actual_x': data.body(validator.ee_id).xpos[0],
                    'actual_y': data.body(validator.ee_id).xpos[1],
                    'actual_z': data.body(validator.ee_id).xpos[2],
                    'obs_meta': str(obs_meta)
                })

        # Save telemetry
        csv_fn = os.path.join(SAVE_DIR, f"run_{iteration}_{args.mode}.csv")
        with open(csv_fn, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        
        iteration += 1
        print(f"📊 Run {iteration} saved to {csv_fn}")

if __name__ == "__main__":
    main()