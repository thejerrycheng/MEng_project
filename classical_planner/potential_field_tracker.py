import mujoco
import mujoco.viewer
import numpy as np
import os
import sys
import time

# --- Configuration ---
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURR_DIR)
MUJOCO_SIM_DIR = os.path.join(PROJECT_ROOT, "mujoco_sim")
XML_PATH = os.path.join(MUJOCO_SIM_DIR, "assets", "obstacle.xml")
SAVE_DIR = os.path.join(MUJOCO_SIM_DIR, "optimal_path_rrt")

# Robot Parameters
START_Q = np.array([0.0, -0.8, -1.5, 0.0, 1.2, 0.0])
GOAL_Q  = np.array([0.8, 0.6, -1.0, 0.5, -0.5, 1.5])

# Controller Gains
KP = 1500.0
KD = 120.0

class CinemaPlannerTracker:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")
        self.obstacle_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) 
                            for n in ["table_obstacle", "pillar_obstacle", "small_block", "overhead_bar"]]

    def generate_trajectory(self, start_q, goal_q, max_steps=10000, target_spacing=0.005):
        print("Planning collision-free trajectory...")
        q_curr = start_q.copy()
        raw_path = [q_curr.copy()]
        
        for _ in range(max_steps):
            f_att = -0.5 * (q_curr - goal_q)
            self.data.qpos[:6] = q_curr
            mujoco.mj_forward(self.model, self.data)
            f_rep = np.zeros(6)
            for i in range(self.data.ncon):
                con = self.data.contact[i]
                if con.geom1 in self.obstacle_ids or con.geom2 in self.obstacle_ids:
                    rho = max(con.dist, 0.01)
                    if rho < 0.4:
                        jacp = np.zeros((3, self.model.nv))
                        mujoco.mj_jacBody(self.model, self.data, jacp, None, self.ee_id)
                        mag = 0.1 * (1.0/rho - 1.0/0.4) * (1.0/rho**2)
                        f_rep += jacp[:, :6].T @ (mag * con.frame[:3])
            
            step = f_att + f_rep
            if np.linalg.norm(step) > 0.005: 
                step = (step / np.linalg.norm(step)) * 0.005
            
            q_curr += step
            raw_path.append(q_curr.copy())
            if np.linalg.norm(q_curr - goal_q) < 0.005: break

        raw_path = np.array(raw_path)
        diffs = np.diff(raw_path, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        cumulative_dist = np.insert(np.cumsum(dists), 0, 0)
        total_length = cumulative_dist[-1]
        num_samples = int(total_length / target_spacing)
        even_dist_markers = np.linspace(0, total_length, num_samples)
        
        evenly_spaced_path = []
        for j in range(6):
            joint_coords = np.interp(even_dist_markers, cumulative_dist, raw_path[:, j])
            evenly_spaced_path.append(joint_coords)
            
        return np.array(evenly_spaced_path).T

    def get_ee_pos(self, q):
        """Helper to find the EE position of a specific joint configuration"""
        temp_data = mujoco.MjData(self.model)
        temp_data.qpos[:6] = q
        mujoco.mj_forward(self.model, temp_data)
        return temp_data.body(self.ee_id).xpos.copy()

    def get_id_control(self, target_q):
        q_err = target_q - self.data.qpos[:6]
        v_err = 0 - self.data.qvel[:6]
        q_accel_des = KP * q_err + KD * v_err
        self.data.qacc[:6] = q_accel_des
        mujoco.mj_inverse(self.model, self.data)
        return self.data.qfrc_inverse[:6].copy()

def main():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    tracker = CinemaPlannerTracker(model, data)

    trajectory = tracker.generate_trajectory(START_Q, GOAL_Q)
    num_waypoints = len(trajectory)
    
    # Pre-calculate visual breadcrumbs
    visual_path = [tracker.get_ee_pos(q) for q in trajectory[::25]]

    data.time = 0
    data.qpos[:6] = START_Q
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_idx = 0
        while viewer.is_running():
            step_start = time.time()

            # 1. Update Target and Control
            if step_idx < num_waypoints:
                target_q = trajectory[step_idx]
                tau = tracker.get_id_control(target_q)
                data.ctrl[:6] = tau
                
                # Get the 3D position of the CURRENT target joint config
                current_target_ee = tracker.get_ee_pos(target_q)
                
                mujoco.mj_step(model, data)
                step_idx += 1
            else:
                tau = tracker.get_id_control(GOAL_Q)
                data.ctrl[:6] = tau
                current_target_ee = tracker.get_ee_pos(GOAL_Q)
                mujoco.mj_step(model, data)

            # --- 2. Visualization ---
            viewer.user_scn.ngeom = 0
            
            # Static Grey Breadcrumbs (Desired Path)
            for pos in visual_path:
                mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], 
                                    mujoco.mjtGeom.mjGEOM_SPHERE, [0.005, 0, 0], 
                                    pos, np.eye(3).flatten(), [0.5, 0.5, 0.5, 0.2])
                viewer.user_scn.ngeom += 1
            
            # Active Moving Target Position (Green Sphere)
            mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], 
                                mujoco.mjtGeom.mjGEOM_SPHERE, [0.015, 0, 0], 
                                current_target_ee, np.eye(3).flatten(), [0, 1, 0, 1])
            viewer.user_scn.ngeom += 1
            
            viewer.sync()

            # --- 3. Monitoring ---
            if step_idx % 50 == 0:
                sys.stdout.write("\033[H\033[J")
                progress = (step_idx / num_waypoints) * 100
                print("================ ID TRACKING MONITOR ================")
                print(f" Progress: {progress:.1f}% ({step_idx}/{num_waypoints})")
                print(f" Status:   {'[ EXECUTING ]' if step_idx < num_waypoints else '[ COMPLETED ]'}")
                print(f" Target EE: {current_target_ee}")
                print("======================================================")

            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

    np.save(os.path.join(SAVE_DIR, "waypoints.npy"), np.array(trajectory))
    print(f"\n[!] Done. Saved {num_waypoints} waypoints.")

if __name__ == "__main__":
    main()