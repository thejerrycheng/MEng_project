import mujoco
import mujoco.viewer
import numpy as np
import os
import sys
import time
import argparse

# --- Configuration ---
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURR_DIR)
XML_PATH = os.path.join(PROJECT_ROOT, "mujoco_sim", "assets", "scene2.xml")

# Hyperparameters
K_ATT = 1.0          
K_REP = 0.5         
RHO_0 = 0.3          
SPACING = 0.2        
MAX_STEP = 0.02      

# --- IK Helper for Start/End ---
def solve_ik(model, data, target_xyz, body_name="ee_mount"):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    best_q = data.qpos[:6].copy()
    min_dist = float('inf')
    
    for attempt in range(15):
        data.qpos[:6] = np.random.uniform(-2, 2, 6) if attempt > 0 else np.zeros(6)
        mujoco.mj_forward(model, data)
        for _ in range(200):
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, None, body_id)
            err = target_xyz - data.body(body_id).xpos
            dist = np.linalg.norm(err)
            if dist < 1e-4: return data.qpos[:6].copy()
            if dist < min_dist:
                min_dist = dist
                best_q = data.qpos[:6].copy()
            J = jacp[:, :6]
            dq = J.T @ np.linalg.solve(J @ J.T + 0.01 * np.eye(3), err)
            data.qpos[:6] += dq * 0.5
            mujoco.mj_forward(model, data)
    return best_q if min_dist < 0.01 else None

class PotentialFieldPlanner:
    def __init__(self, model, start_q, goal_q):
        self.model = model
        self.data = mujoco.MjData(model)
        self.q_curr = start_q.copy()
        self.goal_q = goal_q
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")
        
        # Trace storage
        self.trace_points = []
        
        self.data.qpos[:6] = goal_q
        mujoco.mj_forward(model, self.data)
        self.goal_pos = self.data.body(self.ee_id).xpos.copy()
        self.obs_ids = [i for i in range(model.ngeom) if "obstacle" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").lower()]

    def get_vector_at_point(self, point):
        vec_att = self.goal_pos - point
        force_att = K_ATT * (vec_att / (np.linalg.norm(vec_att) + 1e-5))
        force_rep = np.zeros(3)
        for gid in self.obs_ids:
            obs_pos = self.model.geom_pos[gid]
            obs_size = np.max(self.model.geom_size[gid])
            diff = point - obs_pos
            dist = np.linalg.norm(diff) - obs_size
            if 0 < dist < RHO_0:
                mag = K_REP * (1.0/dist - 1.0/RHO_0) * (1.0/dist**2)
                force_rep += mag * (diff / np.linalg.norm(diff))
        total_force = force_att + force_rep
        mag = np.linalg.norm(total_force)
        return (total_force / (mag + 1e-5)), mag

    def step(self):
        self.data.qpos[:6] = self.q_curr
        mujoco.mj_forward(self.model, self.data)
        
        # Save current EE position to trace
        self.trace_points.append(self.data.body(self.ee_id).xpos.copy())
        
        f_att = -0.5 * (self.q_curr - self.goal_q)
        f_rep = np.zeros(6)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.dist < RHO_0 and contact.dist > 0:
                jacp = np.zeros((3, self.model.nv))
                mujoco.mj_jacBody(self.model, self.data, jacp, None, self.ee_id)
                mag = K_REP * (1.0/max(contact.dist, 0.01) - 1.0/RHO_0)
                f_rep += jacp[:, :6].T @ (mag * contact.frame[:3])
        force = f_att + f_rep
        mag = np.linalg.norm(force)
        if mag > MAX_STEP: force = (force / mag) * MAX_STEP
        self.q_curr += force
        return np.linalg.norm(self.q_curr - self.goal_q) < 0.05

def get_rotation_matrix(direction):
    z = direction / np.linalg.norm(direction)
    ref = np.array([0, 1, 0]) if abs(z[0]) > 0.9 else np.array([1, 0, 0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack((x, y, z)).flatten()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", nargs=3, type=float, default=[0, 0.1, 0.3])
    parser.add_argument("--end", nargs=3, type=float, default=[0, 0.5, 0.3])
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    start_q = solve_ik(model, data, np.array(args.start))
    goal_q  = solve_ik(model, data, np.array(args.end))
    if start_q is None or goal_q is None: return

    planner = PotentialFieldPlanner(model, start_q, goal_q)
    id_mat = np.eye(3).flatten()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            reached = planner.step()
            viewer.user_scn.ngeom = 0
            
            # 1. Draw 3D Vector Field (Arrows)
            x_range = np.arange(-0.6, 0.61, SPACING)
            y_range = np.arange(-0.6, 0.61, SPACING)
            z_range = np.arange(0.1, 0.71, SPACING)
            for x in x_range:
                for y in y_range:
                    for z in z_range:
                        pos = np.array([x, y, z])
                        direction, mag = planner.get_vector_at_point(pos)
                        rot_mat = get_rotation_matrix(direction)
                        color = [0.2, 0.5, 1.0, 0.2] if mag < 1.5 else [1.0, 0.2, 0.2, 0.3]
                        mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], 
                                            mujoco.mjtGeom.mjGEOM_ARROW, [0.005, 0.005, 0.07], 
                                            pos, rot_mat, color)
                        viewer.user_scn.ngeom += 1

            # 2. Draw Ghost Trace (Grey Spheres)
            # Only draw every 3rd point to avoid hitting geom limits
            for i in range(0, len(planner.trace_points), 3):
                if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom: break
                mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], 
                                    mujoco.mjtGeom.mjGEOM_SPHERE, [0.008, 0, 0], 
                                    planner.trace_points[i], id_mat, [0.7, 0.7, 0.7, 0.2])
                viewer.user_scn.ngeom += 1

            # 3. Draw Goal
            mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, 
                                [0.05, 0, 0], planner.goal_pos, id_mat, [0, 1, 0, 0.5])
            viewer.user_scn.ngeom += 1

            data.qpos[:6] = planner.q_curr
            mujoco.mj_forward(model, data)
            viewer.sync()
            
            if reached:
                time.sleep(2.0)
                planner.q_curr = start_q.copy()
                planner.trace_points = [] # Clear trace for next loop

if __name__ == "__main__":
    main()