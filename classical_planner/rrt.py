import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import sys

# --- Configuration ---
# This gets: /Users/jerrycheng/Desktop/MEng_project/classical_planner
CURR_DIR = os.path.dirname(os.path.abspath(__file__))

# Navigate up to: /Users/jerrycheng/Desktop/MEng_project
PROJECT_ROOT = os.path.dirname(CURR_DIR)

# Define the path to the mujoco_sim directory
MUJOCO_SIM_DIR = os.path.join(PROJECT_ROOT, "mujoco_sim")

# Assets are in mujoco_sim/assets/obstacle.xml
XML_PATH = os.path.join(MUJOCO_SIM_DIR, "assets", "obstacle.xml")

# Save path is in mujoco_sim/optimal_path_rrt
SAVE_DIR = os.path.join(MUJOCO_SIM_DIR, "optimal_path_rrt")

# Robot Parameters
START_Q = np.array([0.0, -0.8, -1.5, 0.0, 1.2, 0.0])
GOAL_Q  = np.array([0.8, 0.6, -1.0, 0.5, -0.5, 1.5])

# RRT* Budget and Hyperparameters
MAX_SAMPLES = 50000    
STEP_SIZE = 0.15      
SEARCH_RADIUS = 0.6   
MAX_VIZ_GEOMS = 4800  

class GlobalRRTStar:
    def __init__(self, model, start_q, goal_q):
        self.model = model
        self.check_data = mujoco.MjData(model)
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")
        
        self.nodes = [start_q]
        self.parents = {0: None}
        self.costs = {0: 0.0}
        self.node_ee_positions = [self.get_ee_pos(start_q)]
        
        self.goal_q = goal_q
        self.goal_ee_pos = self.get_ee_pos(goal_q)
        
        self.samples_count = 0
        self.goal_found = False
        self.best_goal_node = None
        self.best_cost = float('inf')
        self.all_samples = [] 
        self.discovery_log = []

    def get_ee_pos(self, q):
        self.check_data.qpos[:6] = q
        mujoco.mj_forward(self.model, self.check_data)
        return self.check_data.body(self.ee_id).xpos.copy()

    def is_collision_free(self, q1, q2=None):
        def check(q):
            self.check_data.qpos[:6] = q
            mujoco.mj_forward(self.model, self.check_data)
            if self.check_data.ncon == 0: return True
            for i in range(self.check_data.ncon):
                con = self.check_data.contact[i]
                g1 = mujoco.mj_id2name(self.model, 14, con.geom1)
                g2 = mujoco.mj_id2name(self.model, 14, con.geom2)
                if g1 is None or g2 is None: continue
                if "floor" in [g1, g2] and any(x in [g1, g2] for x in ["base", "link1"]): continue
                return False
            return True
        if q2 is None: return check(q1)
        for t in np.linspace(0, 1, 6):
            if not check(q1 + t*(q2-q1)): return False
        return True

    def step(self):
        if self.samples_count >= MAX_SAMPLES: return False
        self.samples_count += 1
        
        # --- REVISED SAMPLING STRATEGY ---
        p = np.random.random()
        
        # If goal is found, use Informed Sampling 50% of the time
        if self.goal_found and p < 0.50:
            # 1. Get the current optimal path waypoints
            path_indices = self.get_path_indices()
            # 2. Pick a random node index from that path
            target_node_idx = np.random.choice(path_indices)
            q_base = self.nodes[target_node_idx]
            # 3. Add Gaussian noise (std dev of 0.2 rad) to search "around" the path
            q_rand = q_base + np.random.normal(0, 0.2, 6)
            q_rand = np.clip(q_rand, -3.14, 3.14)
            
        elif p < 0.15: 
            # Standard Goal Bias
            q_rand = self.goal_q
        else:
            # Global Uniform Exploration
            q_rand = np.array([np.random.uniform(-3.14, 3.14) for _ in range(6)])
        
        # --- END OF REVISED SAMPLING ---

        node_arr = np.array(self.nodes)
        nearest_idx = np.argmin(np.linalg.norm(node_arr - q_rand, axis=1))
        q_near = self.nodes[nearest_idx]
        
        diff = q_rand - q_near
        d = np.linalg.norm(diff)
        if d < 1e-6: return True
        q_new = q_near + (diff / d) * min(d, STEP_SIZE)
        
        is_free = self.is_collision_free(q_near, q_new)
        ee_new = self.get_ee_pos(q_new)
        self.all_samples.append((ee_new, is_free))
        
        if is_free:
            dists_to_new = np.linalg.norm(node_arr - q_new, axis=1)
            neighbors = np.where(dists_to_new < SEARCH_RADIUS)[0]
            
            best_p, min_c = nearest_idx, self.costs[nearest_idx] + np.linalg.norm(q_new - q_near)
            for n_idx in neighbors:
                c_via_n = self.costs[n_idx] + np.linalg.norm(q_new - self.nodes[n_idx])
                if c_via_n < min_c and self.is_collision_free(self.nodes[n_idx], q_new):
                    best_p, min_c = n_idx, c_via_n
            
            new_idx = len(self.nodes)
            self.nodes.append(q_new); self.node_ee_positions.append(ee_new)
            self.parents[new_idx] = best_p; self.costs[new_idx] = min_c
            
            for n_idx in neighbors:
                c_via_new = self.costs[new_idx] + np.linalg.norm(self.nodes[n_idx] - q_new)
                if c_via_new < self.costs[n_idx] and self.is_collision_free(q_new, self.nodes[n_idx]):
                    self.parents[n_idx] = new_idx; self.costs[n_idx] = c_via_new
            
            if np.linalg.norm(q_new - self.goal_q) < STEP_SIZE:
                self.goal_found = True
                if min_c < self.best_cost:
                    delta = self.best_cost - min_c
                    self.best_cost = min_c; self.best_goal_node = new_idx
                    self.discovery_log.append(f"Better Path! Cost: {min_c:.4f} (-{delta:.4f})")
        return True

    def get_path_indices(self):
        if self.best_goal_node is None: return []
        path, curr = [], self.best_goal_node
        while curr is not None:
            path.append(curr); curr = self.parents[curr]
        return path

def main():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    rrt = GlobalRRTStar(model, START_Q, GOAL_Q)
    
    id_mat = np.eye(3).flatten().astype(np.float64)
    zero_v = np.zeros(3).astype(np.float64)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while rrt.samples_count < MAX_SAMPLES:
            if not viewer.is_running(): break
            
            for _ in range(40): rrt.step()
            
            # --- Rendering ---
            viewer.user_scn.ngeom = 0
            opt_indices = rrt.get_path_indices()
            opt_set = set(opt_indices)

            # Draw samples
            for pos, free in rrt.all_samples[-200:]:
                if viewer.user_scn.ngeom >= 500: break
                color = [0,1,1,0.2] if free else [1,0,0,0.1]
                mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], 2, [0.005,0,0], pos, id_mat, color)
                viewer.user_scn.ngeom += 1

            # Draw Tree
            for child, parent in rrt.parents.items():
                if parent is None or viewer.user_scn.ngeom >= MAX_VIZ_GEOMS: continue
                is_opt = (child in opt_set and parent in opt_set and abs(opt_indices.index(child) - opt_indices.index(parent)) == 1)
                color, rad = ([1,0,0,1], 0.007) if is_opt else ([0,1,0,0.2], 0.002)
                idx = viewer.user_scn.ngeom
                mujoco.mjv_initGeom(viewer.user_scn.geoms[idx], 3, zero_v, zero_v, id_mat, color)
                mujoco.mjv_connector(viewer.user_scn.geoms[idx], 3, rad, rrt.node_ee_positions[parent], rrt.node_ee_positions[child])
                viewer.user_scn.ngeom += 1
            viewer.sync()

        # --- Revised Terminal Logging ---
            sys.stdout.write("\033[H\033[J") 
            
            # Define status strings outside the f-string to avoid backslash error
            status_str = "[\033[92m GOAL REACHED \033[0m]" if rrt.goal_found else "[\033[94m EXPLORING \033[0m]"
            best_cost_str = f"{rrt.best_cost:.4f}" if rrt.goal_found else "N/A"

            log_output = [
                "================ RRT* LIVE MONITORING ================",
                f" Samples: {rrt.samples_count}/{MAX_SAMPLES} | Nodes: {len(rrt.nodes)}",
                f" Status:  {status_str}",
                f" Best Cost: {best_cost_str}",
                "-" * 54
            ]

            if rrt.discovery_log:
                log_output.append(" Recent Optimization Discoveries:")
                for log in rrt.discovery_log[-5:]:
                    # The [+] prefix uses a color code; we can use a variable or simple concat
                    yellow_plus = "\033[93m[+]\033[0m"
                    log_output.append(f"  {yellow_plus} {log}")
            else:
                log_output.append(" Waiting for initial path discovery...")

            log_output.append("======================================================")
            
            sys.stdout.write("\n".join(log_output) + "\n")
            sys.stdout.flush()
            
        # --- Final Save & Exit ---
        if rrt.best_goal_node is not None:
            path_q = []
            curr = rrt.best_goal_node
            while curr is not None:
                path_q.append(rrt.nodes[curr]); curr = rrt.parents[curr]
            path_q = np.array(path_q[::-1])
            np.save(os.path.join(SAVE_DIR, "waypoints.npy"), path_q)
            print(f"\n[!] SUCCESS: {len(path_q)} waypoints saved.")
        else:
            print("\n[!] FAILURE: Sampling budget reached with no valid path.")

        print("[!] Cleaning up and exiting...")
        viewer.close()

    sys.exit()

if __name__ == "__main__":
    main()