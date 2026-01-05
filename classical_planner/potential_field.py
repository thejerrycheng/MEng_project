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

# APF Hyperparameters
K_ATT = 0.5          
K_REP = 0.08         
RHO_0 = 0.4          
MAX_STEP = 0.02      
GOAL_THRESHOLD = 0.05
STUCK_THRESHOLD = 1e-5

class PotentialFieldPlanner:
    def __init__(self, model, start_q, goal_q):
        self.model = model
        self.data = mujoco.MjData(model)
        self.q_curr = start_q.copy()
        self.goal_q = goal_q
        self.path = [start_q.copy()]
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")
        
        # Pre-calculate Goal EE position for visualization
        self.data.qpos[:6] = goal_q
        mujoco.mj_forward(model, self.data)
        self.goal_ee_pos = self.data.body(self.ee_id).xpos.copy()

        self.obstacle_geom_ids = []
        for name in ["table_obstacle", "pillar_obstacle", "small_block", "overhead_bar"]:
            try:
                self.obstacle_geom_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
            except: continue

    def get_ee_pos(self, q):
        self.data.qpos[:6] = q
        mujoco.mj_forward(self.model, self.data)
        return self.data.body(self.ee_id).xpos.copy()

    def get_repulsive_force(self, q):
        self.data.qpos[:6] = q
        mujoco.mj_forward(self.model, self.data)
        repulsion = np.zeros(6)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 in self.obstacle_geom_ids or contact.geom2 in self.obstacle_geom_ids:
                rho = max(contact.dist, 0.01) 
                if rho < RHO_0:
                    jacp = np.zeros((3, self.model.nv))
                    mujoco.mj_jacBody(self.model, self.data, jacp, None, self.ee_id)
                    mag = K_REP * (1.0/rho - 1.0/RHO_0) * (1.0/(rho**2))
                    normal = contact.frame[:3] 
                    repulsion += jacp[:, :6].T @ (mag * normal)
        return repulsion

    def step(self):
        f_att = -K_ATT * (self.q_curr - self.goal_q)
        f_rep = self.get_repulsive_force(self.q_curr)
        total_force = f_att + f_rep
        
        if np.linalg.norm(total_force) < STUCK_THRESHOLD:
            total_force += np.random.normal(0, 0.02, 6)
        
        mag = np.linalg.norm(total_force)
        if mag > MAX_STEP:
            total_force = (total_force / mag) * MAX_STEP
            
        self.q_curr += total_force
        self.path.append(self.q_curr.copy())
        return np.linalg.norm(self.q_curr - self.goal_q) < GOAL_THRESHOLD

def main():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    planner = PotentialFieldPlanner(model, START_Q, GOAL_Q)

    # Visualization constants
    id_mat = np.eye(3).flatten()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_count = 0
        while viewer.is_running() and step_count < 5000:
            reached = planner.step()
            
            # --- Rendering Path and Goal ---
            viewer.user_scn.ngeom = 0
            
            # 1. Draw Target Position (Green Sphere)
            mujoco.mjv_initGeom(viewer.user_scn.geoms[0], mujoco.mjtGeom.mjGEOM_SPHERE, 
                                [0.04, 0, 0], planner.goal_ee_pos, id_mat, [0, 1, 0, 0.5])
            viewer.user_scn.ngeom += 1

            # 2. Draw Path (Grey Dots) - only draw every 5th step to save memory
            path_visual_limit = 500 
            for i in range(0, len(planner.path), 5):
                if viewer.user_scn.ngeom >= path_visual_limit: break
                pos = planner.get_ee_pos(planner.path[i])
                mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], 
                                    mujoco.mjtGeom.mjGEOM_SPHERE, 
                                    [0.01, 0, 0], pos, id_mat, [0.5, 0.5, 0.5, 0.3])
                viewer.user_scn.ngeom += 1

            # Update Robot Pose
            data.qpos[:6] = planner.q_curr
            mujoco.mj_forward(model, data)
            viewer.sync()

            # Logging
            sys.stdout.write("\033[H\033[J")
            status_str = "[\033[92m REACHED \033[0m]" if reached else "[\033[94m DESCENDING \033[0m]"
            print("================ APF LIVE MONITORING ================")
            print(f" Step: {step_count} | Status: {status_str}")
            print(f" Dist to Goal: {np.linalg.norm(planner.q_curr - GOAL_Q):.4f}")
            print("======================================================")
            
            if reached: break
            step_count += 1
            time.sleep(0.005)

        if reached:
            np.save(os.path.join(SAVE_DIR, "waypoints.npy"), np.array(planner.path))
            print(f"\n[!] SUCCESS: {len(planner.path)} waypoints saved.")
        viewer.close()

if __name__ == "__main__":
    main()