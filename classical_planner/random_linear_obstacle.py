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

class RealisticInterceptPlanner:
    def __init__(self, model, data):
        self.model, self.data = model, data
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")
        self.obs_name = "pillar_obstacle"
        self.obs_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, self.obs_name)
        
        # Clear secondary obstacles
        for other in ["table_obstacle", "small_block"]:
            other_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, other)
            if other_id != -1:
                self.model.geom_pos[other_id] = [0, 0, -10]

    def randomize_obstacle_appearance(self, start_pt, end_pt):
        t = np.random.uniform(0.35, 0.65)
        intercept_pos = (1 - t) * start_pt + t * end_pt
        intercept_pos += np.random.uniform(-0.03, 0.03, size=3)
        self.model.geom_pos[self.obs_id] = intercept_pos

        geom_types = [mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_SPHERE]
        chosen_type = np.random.choice(geom_types)
        self.model.geom_type[self.obs_id] = chosen_type
        
        # Real-world sizing (cm)
        if chosen_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            size = np.array([np.random.uniform(0.03, 0.05), np.random.uniform(0.12, 0.2), 0.0])
        elif chosen_type == mujoco.mjtGeom.mjGEOM_BOX:
            size = np.random.uniform(0.04, 0.08, size=3)
        else:
            size = np.array([np.random.uniform(0.05, 0.08), 0, 0])
            
        self.model.geom_size[self.obs_id] = size
        self.model.geom_rgba[self.obs_id] = [np.random.uniform(0.1, 0.7), 0.4, 0.7, 1.0]

        return {'pos': intercept_pos.tolist(), 'size': size.tolist(), 'type': int(chosen_type)}

    def plan_offline(self, start_pos, end_pos, max_steps=12000):
        curr_p = np.array(start_pos)
        path = [curr_p.copy()]
        last_p = curr_p.copy()
        stuck_count = 0
        
        for i in range(max_steps):
            diff_goal = end_pos - curr_p
            dist_goal = np.linalg.norm(diff_goal)
            f_att = (diff_goal / (dist_goal + 1e-6)) * 1.8
            
            diff_obs = curr_p - self.model.geom_pos[self.obs_id]
            dist_obs = np.linalg.norm(diff_obs) - np.max(self.model.geom_size[self.obs_id])
            
            f_rep = np.zeros(3)
            influence_rad = 0.30
            if dist_obs < influence_rad:
                mag = 0.35 * (1.0/max(dist_obs, 0.01) - 1.0/influence_rad) * (1.0/max(dist_obs, 0.01)**2)
                dir_obs = diff_obs / (np.linalg.norm(diff_obs) + 1e-6)
                # Symmetry breaker: push 'side' and 'up'
                side_vec = np.array([-dir_obs[1], dir_obs[0], 0.15])
                f_rep = (dir_obs * mag) + (side_vec * mag * 0.45)

            # Progress check
            if np.linalg.norm(curr_p - last_p) < 1e-5:
                stuck_count += 1
            else:
                stuck_count = 0
            
            if stuck_count > 15: # Escape noise
                curr_p += np.random.uniform(-0.06, 0.06, size=3)
                stuck_count = 0
            else:
                total_f = f_att + f_rep
                curr_p += (total_f / (np.linalg.norm(total_f) + 1e-6)) * 0.0035
            
            path.append(curr_p.copy())
            last_p = curr_p.copy()
            if dist_goal < 0.012: return np.array(path)
            
        return None

    def get_id_control(self, target_pos, target_obj):
        curr_pos = self.data.body(self.ee_id).xpos
        curr_mat = self.data.body(self.ee_id).xmat.reshape(3, 3)
        
        z_axis = (target_obj - curr_pos); z_axis /= (np.linalg.norm(z_axis) + 1e-6)
        up = np.array([0, 0, 1])
        x_axis = np.cross(up, z_axis); x_axis /= (np.linalg.norm(x_axis) + 1e-6)
        y_axis = np.cross(z_axis, x_axis)
        target_mat = np.stack([x_axis, y_axis, z_axis], axis=1)

        pos_err = target_pos - curr_pos
        rot_err_mat = target_mat @ curr_mat.T
        rot_err_quat = np.zeros(4)
        mujoco.mju_mat2Quat(rot_err_quat, rot_err_mat.flatten())
        rot_err_vec = rot_err_quat[1:] * np.sign(rot_err_quat[0])
        
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_id)
        J = np.vstack([jacp, jacr])[:, :6]
        
        J_inv = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.eye(6))
        q_accel = KP * (J_inv @ np.concatenate([pos_err, rot_err_vec])) - KD * self.data.qvel[:6]
        
        self.data.qacc[:6] = q_accel
        mujoco.mj_inverse(self.model, self.data)
        return self.data.qfrc_inverse[:6].copy(), pos_err

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument("--mode", choices=['pan', 'dolly', 'crane'], default='pan')
    parser.add_argument("--duration", type=float, default=6.0)
    args = parser.parse_args()

    presets = {
        'pan': ([0.4, 0.1, 0.4], [0.4, 0.45, 0.4], [0.75, 0.3, 0.2]),
        'dolly': ([0.3, 0.3, 0.4], [0.65, 0.3, 0.4], [0.8, 0.3, 0.3]),
        'crane': ([0.4, 0.3, 0.2], [0.4, 0.3, 0.55], [0.7, 0.3, 0.15])
    }

    iteration = 0
    while iteration < args.num:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
        data = mujoco.MjData(model)
        planner = RealisticInterceptPlanner(model, data)
        
        start_pt, end_pt, subject = presets[args.mode]
        obs_meta = planner.randomize_obstacle_appearance(np.array(start_pt), np.array(end_pt))
        
        sys.stdout.write(f"\r[OFFLINE] Planning Shot {iteration+1}...")
        sys.stdout.flush()
        full_path = planner.plan_offline(start_pt, end_pt)
        
        if full_path is None: continue

        csv_rows = []
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while data.time < args.duration and viewer.is_running():
                t_loop = time.time()
                t_norm = data.time / args.duration
                idx = min(int(t_norm * len(full_path)), len(full_path)-1)
                target_p = full_path[idx]
                
                tau, p_err = planner.get_id_control(target_p, np.array(subject))
                data.ctrl[:6] = tau
                mujoco.mj_step(model, data)

                # --- LIVE TERMINAL MONITOR ---
                if int(data.time/model.opt.timestep) % 40 == 0:
                    sys.stdout.write("\033[H\033[J")
                    print("================ CINEMA ROBOT LIVE MONITOR ================")
                    print(f" Shot: {iteration+1}/{args.num} | Mode: {args.mode.upper()}")
                    print(f" Progress: {t_norm*100:.1f}% | Time: {data.time:.2f}s")
                    print(f" Tracking Error: {np.linalg.norm(p_err)*1000:.2f} mm")
                    print("-" * 55)
                    print(f" Obstacle Type: {obs_meta['type']} (2=Cyl, 3=Box, 2=Sph)")
                    print(f" Obstacle Pos:  {np.array(obs_meta['pos']).round(3)}")
                    print("===========================================================")

                if int(data.time/model.opt.timestep) % 15 == 0:
                    viewer.user_scn.ngeom = 0
                    for p in full_path[::30]:
                        mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], 2, [0.003,0,0], p, np.eye(3).flatten(), [1,1,1,0.1])
                        viewer.user_scn.ngeom += 1
                    viewer.sync()

                csv_rows.append({'time': data.time, 'target_x': target_p[0], 'actual_x': data.body(planner.ee_id).xpos[0], 'obs_meta': str(obs_meta)})
                while (time.time() - t_loop) < model.opt.timestep: pass

        csv_fn = os.path.join(SAVE_DIR, f"dynamic_run_{iteration}.csv")
        with open(csv_fn, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        iteration += 1

if __name__ == "__main__":
    main()