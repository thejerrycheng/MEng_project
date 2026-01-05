import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import sys

# --- Configuration ---
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURR_DIR)
MUJOCO_SIM_DIR = os.path.join(PROJECT_ROOT, "mujoco_sim")
XML_PATH = os.path.join(MUJOCO_SIM_DIR, "assets", "obstacle.xml")

# Tracking Gains (High stiffness for precise tracking)
KP_ID, KD_ID = 2500.0, 150.0

# RRT* Hyperparameters
MAX_RRT_SAMPLES = 20000
RRT_STEP_SIZE = 0.2
RRT_SEARCH_RADIUS = 0.8

# ==========================================
# 1. Scene & Workspace Manager
# ==========================================
class SceneManager:
    def __init__(self, model, data):
        self.model, self.data = model, data
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")
        self.obs_names = ["pillar_obstacle", "table_obstacle", "small_block"]
        self.obs_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in self.obs_names]
        
        # Workspace definitions (Task Space XYZ)
        # Keep points in front of and to the side of the robot base
        self.ws_min = np.array([0.3, -0.4, 0.2])
        self.ws_max = np.array([0.7,  0.4, 0.6])

    def randomize_scene(self):
        """Randomizes obstacles and generates start/end points in workspace."""
        # 1. Randomize Obstacles (realistic sizes)
        for g_id in self.obs_ids:
            if g_id == -1: continue
            # Random pos within workspace
            pos = np.random.uniform(self.ws_min - 0.1, self.ws_max + 0.1)
            # Keep z positive and not too high
            pos[2] = np.random.uniform(0.05, 0.4) 
            self.model.geom_pos[g_id] = pos
            
            # Random realistic size (e.g., 5cm to 15cm dimensions)
            size = np.random.uniform(0.05, 0.15, size=3)
            # If it's a pillar/cylinder, Z-size is height/2
            if self.model.geom_type[g_id] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                 size[0] = size[1] # radius
                 size[2] = np.random.uniform(0.1, 0.3) # height
            self.model.geom_size[g_id] = size
            
            # Random color
            self.model.geom_rgba[g_id] = [np.random.rand(), np.random.rand(), np.random.rand(), 1.0]

        # 2. Generate valid start/end points
        start_pos = np.random.uniform(self.ws_min, self.ws_max)
        # Ensure end point is sufficiently far away
        while True:
            end_pos = np.random.uniform(self.ws_min, self.ws_max)
            if np.linalg.norm(start_pos - end_pos) > 0.3: break
            
        subject_pos = np.array([0.8, 0.0, 0.2]) # Fixed subject to look at
        return start_pos, end_pos, subject_pos

    def solve_ik(self, target_pos, initial_q):
        """Simple Numerical IK to convert Task Space -> Joint Space."""
        temp_data = mujoco.MjData(self.model)
        temp_data.qpos[:6] = initial_q
        mujoco.mj_forward(self.model, temp_data)
        
        for _ in range(500): # Max iterations
            curr_pos = temp_data.body(self.ee_id).xpos
            err = target_pos - curr_pos
            if np.linalg.norm(err) < 1e-3: return temp_data.qpos[:6].copy()
            
            jacp = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, temp_data, jacp, None, self.ee_id)
            J = jacp[:, :6]
            # Damped Least Squares
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err)
            temp_data.qpos[:6] += dq * 0.5 # step size
            mujoco.mj_forward(self.model, temp_data)
            
        print("Warning: IK did not converge.")
        return temp_data.qpos[:6].copy() # Return best effort

# ==========================================
# 2. Offline RRT* Planner (Joint Space)
# ==========================================
class GlobalRRTStar:
    def __init__(self, model, start_q, goal_q):
        self.model = model
        # Use a separate data structure for collision checking to not mess up main sim
        self.check_data = mujoco.MjData(model) 
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")
        
        self.nodes = [start_q]
        self.parents = {0: None}
        self.costs = {0: 0.0}
        self.goal_q = goal_q
        self.samples_count = 0
        self.goal_found = False
        self.best_goal_node = None
        self.best_cost = float('inf')

    def is_collision_free(self, q_check):
        self.check_data.qpos[:6] = q_check
        mujoco.mj_forward(self.model, self.check_data)
        return self.check_data.ncon == 0

    def step(self):
        if self.samples_count >= MAX_RRT_SAMPLES: return False
        self.samples_count += 1
        
        # Informed Sampling Strategy
        if self.goal_found and np.random.random() < 0.6:
            path_indices = self.get_path_indices()
            q_base = self.nodes[np.random.choice(path_indices)]
            q_rand = q_base + np.random.normal(0, 0.25, 6)
        elif np.random.random() < 0.1:
            q_rand = self.goal_q
        else:
            q_rand = np.random.uniform(-np.pi, np.pi, 6)

        node_arr = np.array(self.nodes)
        nearest_idx = np.argmin(np.linalg.norm(node_arr - q_rand, axis=1))
        q_near = self.nodes[nearest_idx]
        
        diff = q_rand - q_near
        d = np.linalg.norm(diff)
        if d < 1e-6: return True
        q_new = q_near + (diff / d) * min(d, RRT_STEP_SIZE)
        
        if self.is_collision_free(q_new):
            dists_to_new = np.linalg.norm(node_arr - q_new, axis=1)
            neighbors = np.where(dists_to_new < RRT_SEARCH_RADIUS)[0]
            
            best_p, min_c = nearest_idx, self.costs[nearest_idx] + np.linalg.norm(q_new - q_near)
            # RRT* Rewiring (Find best parent)
            for n_idx in neighbors:
                c_via_n = self.costs[n_idx] + np.linalg.norm(q_new - self.nodes[n_idx])
                if c_via_n < min_c: best_p, min_c = n_idx, c_via_n
            
            new_idx = len(self.nodes)
            self.nodes.append(q_new)
            self.parents[new_idx] = best_p; self.costs[new_idx] = min_c
            
            # RRT* Rewiring (Update neighbors)
            for n_idx in neighbors:
                c_via_new = self.costs[new_idx] + np.linalg.norm(self.nodes[n_idx] - q_new)
                if c_via_new < self.costs[n_idx]:
                    self.parents[n_idx] = new_idx; self.costs[n_idx] = c_via_new
            
            if np.linalg.norm(q_new - self.goal_q) < RRT_STEP_SIZE:
                self.goal_found = True
                if min_c < self.best_cost:
                    self.best_cost = min_c; self.best_goal_node = new_idx
        return True

    def get_path_indices(self):
        if self.best_goal_node is None: return []
        path, curr = [], self.best_goal_node
        while curr is not None:
            path.append(curr); curr = self.parents[curr]
        return path[::-1]

# ==========================================
# 3. Online Inverse Dynamics Tracker
# ==========================================
class IDController:
    def __init__(self, model, data):
        self.model, self.data = model, data
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")

    def get_torque(self, target_pos, subject_pos):
        curr_pos = self.data.body(self.ee_id).xpos
        
        # Orientation (Look-at subject)
        z = (subject_pos - curr_pos); z /= (np.linalg.norm(z) + 1e-6)
        up = np.array([0,0,1]) if abs(z[2]) < 0.9 else np.array([0,1,0])
        x = np.cross(up, z); x /= (np.linalg.norm(x) + 1e-6)
        y = np.cross(z, x)
        target_mat = np.stack([x,y,z], axis=1)
        
        # Errors
        pos_err = target_pos - curr_pos
        curr_mat = self.data.body(self.ee_id).xmat.reshape(3,3)
        rot_err_mat = target_mat @ curr_mat.T
        quat = np.zeros(4); mujoco.mju_mat2Quat(quat, rot_err_mat.flatten())
        rot_err = quat[1:] * np.sign(quat[0])
        
        # Jacobian & Inverse Dynamics
        jacp, jacr = np.zeros((3, 6)), np.zeros((3, 6))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_id)
        J = np.vstack([jacp, jacr])
        
        # J_pseudo_inv * (Kp*err - Kd*vel)
        cart_err = np.concatenate([pos_err, rot_err])
        J_pinv = J.T @ np.linalg.solve(J @ J.T + 1e-4*np.eye(6), np.eye(6))
        q_acc = KP_ID * (J_pinv @ cart_err) - KD_ID * self.data.qvel[:6]
        
        self.data.qacc[:6] = q_acc
        mujoco.mj_inverse(self.model, self.data)
        return self.data.qfrc_inverse[:6].copy()


# ==========================================
# Main Execution Loop
# ==========================================
def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    
    scene_mgr = SceneManager(model, data)
    id_ctrl = IDController(model, data)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")

    # Run multiple automated iterations
    for iteration in range(1, 100):
        # --- 1. Randomize Scene & Get Task Points ---
        start_pos, end_pos, subject_pos = scene_mgr.randomize_scene()
        
        # Use IK to find joint configurations for start/end
        # Use a neutral guess for IK stability
        neutral_q = np.array([0.0, -0.5, -1.0, 0.0, 1.0, 0.0]) 
        start_q = scene_mgr.solve_ik(start_pos, neutral_q)
        goal_q = scene_mgr.solve_ik(end_pos, start_q)

        # --- 2. Offline Planning ---
        print(f"\n[Run {iteration}] Starting Offline RRT* Planning...")
        rrt = GlobalRRTStar(model, start_q, goal_q)
        start_plan_time = time.time()
        
        # Planning loop (no visualization here)
        while time.time() - start_plan_time < 10.0: # 10s timeout
            rrt.step()
            if rrt.goal_found and rrt.samples_count % 500 == 0:
                 sys.stdout.write(f"\r  Goal found. Optimizing... Samples: {rrt.samples_count}, Cost: {rrt.best_cost:.3f}")
                 sys.stdout.flush()
            if rrt.samples_count >= MAX_RRT_SAMPLES: break
            
        if not rrt.goal_found:
            print("\n  Planning timed out or failed. Skipping run.")
            continue
            
        # Extract best path
        path_indices = rrt.get_path_indices()
        joint_path = [rrt.nodes[i] for i in path_indices]
        
        # Convert joint path to task-space path for visualization
        temp_data = mujoco.MjData(model)
        task_path_viz = []
        for q in joint_path:
            temp_data.qpos[:6] = q
            mujoco.mj_forward(model, temp_data)
            task_path_viz.append(temp_data.body(ee_id).xpos.copy())

        print(f"\n  Path finalized with {len(joint_path)} waypoints. Starting execution.")

        # --- 3. Execution & Visualization ---
        # Reset sim to start condition
        data.qpos[:6] = start_q
        data.qvel[:6] = 0
        data.time = 0
        mujoco.mj_forward(model, data)
        
        DURATION = 6.0

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while data.time < DURATION and viewer.is_running():
                step_start = time.time()
                
                # Interpolate target along joint path
                t_norm = data.time / DURATION
                idx_float = t_norm * (len(joint_path) - 1)
                idx = int(idx_float)
                alpha = idx_float - idx
                if idx < len(joint_path) - 1:
                    target_q = (1-alpha)*joint_path[idx] + alpha*joint_path[idx+1]
                else:
                    target_q = joint_path[-1]

                # Calculate Task-space target position for current joint target
                # (Used for Green Dot visualization and ID controller)
                temp_data.qpos[:6] = target_q
                mujoco.mj_forward(model, temp_data)
                target_pos_current = temp_data.body(ee_id).xpos.copy()

                # Get tracking torque
                tau = id_ctrl.get_torque(target_pos_current, subject_pos)
                data.ctrl[:6] = tau
                mujoco.mj_step(model, data)

                # Visualization
                if int(data.time/model.opt.timestep) % 20 == 0:
                    viewer.user_scn.ngeom = 0
                    # Draw Grey Dots (Desired Path)
                    for p in task_path_viz:
                        mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], 
                                            mujoco.mjtGeom.mjGEOM_SPHERE, [0.008,0,0], 
                                            p, np.eye(3).flatten(), [0.5, 0.5, 0.5, 0.3])
                        viewer.user_scn.ngeom += 1
                    
                    # Draw Green Dot (Current Desired Target)
                    mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], 
                                        mujoco.mjtGeom.mjGEOM_SPHERE, [0.02,0,0], 
                                        target_pos_current, np.eye(3).flatten(), [0, 1, 0, 0.8])
                    viewer.user_scn.ngeom += 1
                    viewer.sync()

                # Sync to real-time
                elapsed = time.time() - step_start
                if elapsed < model.opt.timestep:
                    time.sleep(model.opt.timestep - elapsed)

if __name__ == "__main__":
    main()