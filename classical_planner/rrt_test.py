import mujoco
import mujoco.viewer
import numpy as np
import os
import csv
import time
import argparse

# ============================================================
# Configuration
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", nargs=3, type=float, default=[0.0, 0.25, 0.3])
    parser.add_argument("--end", nargs=3, type=float, default=[0.5, 0.25, 0.3])
    parser.add_argument("--num", type=int, default=1)
    parser.add_argument("--viz", action="store_true", default=True)
    parser.add_argument("--max_samples", type=int, default=5000)
    return parser.parse_args()

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURR_DIR)
XML_PATH = os.path.join(PROJECT_ROOT, "mujoco_sim", "assets", "scene2.xml")
SAVE_ROOT = os.path.join(os.getcwd(), "rrt_generated_dataset")

STEP_SIZE = 0.2          
SEARCH_RADIUS = 0.5       
MAX_NEIGHBORS = 10       
COLLISION_RES = 0.05     
WAYPOINT_RESOLUTION = 0.02 

# ============================================================
# Robust IK
# ============================================================
def solve_ik(model, data, target_xyz, body_id):
    target_y_positive = np.array([0, 1, 0])
    for attempt in range(50): 
        data.qpos[:6] = np.random.uniform(-3, 3, 6)
        mujoco.mj_forward(model, data)
        for _ in range(500):
            jacp = np.zeros((3, model.nv)); mujoco.mj_jacBody(model, data, jacp, None, body_id)
            jacr = np.zeros((3, model.nv)); mujoco.mj_jacBody(model, data, None, jacr, body_id)
            curr_xyz = data.body(body_id).xpos
            curr_mat = data.body(body_id).xmat.reshape(3,3)
            pos_err = target_xyz - curr_xyz
            rot_err = np.cross(curr_mat[:, 2], target_y_positive)
            J = np.vstack([jacp[:, :6], jacr[:, :6]])
            dq = np.linalg.solve(J.T @ J + 0.05 * np.eye(6), J.T @ np.concatenate([pos_err, rot_err * 0.8]))
            data.qpos[:6] += dq * 0.4
            data.qpos[:6] = np.clip(data.qpos[:6], -3.1, 3.1)
            mujoco.mj_forward(model, data)
            if np.linalg.norm(pos_err) < 1e-4 and np.linalg.norm(rot_err) < 1e-3:
                return data.qpos[:6].copy()
    return None

# ============================================================
# Optimized RRT* (Explicit Contact Filtering)
# ============================================================
class RRTStarOpt:
    def __init__(self, model, start_q, goal_q):
        self.model = model
        self.data = mujoco.MjData(model)
        self.start_q = start_q
        self.goal_q = goal_q
        self.nodes = [start_q.copy()]
        self.parents = {0: None}
        self.costs = {0: 0.0}
        self.goal_found = False
        self.best_goal_node = None
        self.best_cost = float("inf")
        
        # 1. Identify OBSTACLE Geoms
        self.obstacle_ids = set([
            i for i in range(model.ngeom) 
            if "obstacle" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").lower()
        ])
        
        # 2. Identify ROBOT Geoms (Traverse specific bodies)
        # We need this to ensure we only flag collisions between ROBOT and OBSTACLE
        robot_body_names = ["shoulder", "arm_link1", "elbow", "arm_link2", "wrist1", "wrist2", "ee_mount", "gripper_base", "gripper_l", "gripper_r"]
        self.robot_geom_ids = set()
        
        for name in robot_body_names:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid != -1:
                # Iterate over all geoms attached to this body
                geom_start = model.body_geomadr[bid]
                geom_num = model.body_geomnum[bid]
                for i in range(geom_start, geom_start + geom_num):
                    self.robot_geom_ids.add(i)

        if not self.obstacle_ids:
            print("Warning: No 'obstacle' geoms found.")

    def is_collision_free(self, q1, q2=None):
        def check(q):
            self.data.qpos[:6] = q
            # mj_forward computes positions AND contact pairs
            mujoco.mj_forward(self.model, self.data)
            
            # Iterate through contacts explicitly
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                g1, g2 = c.geom1, c.geom2
                
                # Logic: Collision is strictly (Robot AND Obstacle)
                g1_is_robot = g1 in self.robot_geom_ids
                g2_is_robot = g2 in self.robot_geom_ids
                g1_is_obs = g1 in self.obstacle_ids
                g2_is_obs = g2 in self.obstacle_ids
                
                # Check for Crossing Pairs (One is Robot, One is Obstacle)
                if (g1_is_robot and g2_is_obs) or (g1_is_obs and g2_is_robot):
                    return False # Collision Detected
                    
            return True # No relevant collision found

        if q2 is None: return check(q1)
        
        dist = np.linalg.norm(q1 - q2)
        steps = int(np.ceil(dist / COLLISION_RES))
        
        if steps <= 1: return check(q2) 
        
        for t in np.linspace(0, 1, steps + 1): 
            if not check(q1 + t * (q2 - q1)): return False
        return True

    def step(self):
        # 1. Sample
        if np.random.rand() < 0.2:
            q_rand = self.goal_q
        else:
            q_rand = np.random.uniform(-np.pi, np.pi, 6)

        # 2. Nearest Neighbor
        node_arr = np.array(self.nodes)
        dists = np.linalg.norm(node_arr - q_rand, axis=1)
        nearest_idx = np.argmin(dists)
        q_near = self.nodes[nearest_idx]

        # 3. Steer
        diff = q_rand - q_near
        dist = np.linalg.norm(diff)
        if dist < 1e-6: return

        q_new = q_near + (diff / dist) * min(dist, STEP_SIZE)
        
        # 4. Check Collision (Filtered)
        if not self.is_collision_free(q_near, q_new): return

        # 5. Find Neighbors
        dists_to_new = np.linalg.norm(node_arr - q_new, axis=1)
        mask = dists_to_new < SEARCH_RADIUS
        neighbor_indices = np.where(mask)[0]
        
        if len(neighbor_indices) > MAX_NEIGHBORS:
            closest_k_rel = np.argsort(dists_to_new[neighbor_indices])[:MAX_NEIGHBORS]
            neighbor_indices = neighbor_indices[closest_k_rel]

        # 6. Choose Best Parent
        best_p = nearest_idx
        min_c = self.costs[nearest_idx] + np.linalg.norm(q_new - q_near)

        for n_idx in neighbor_indices:
            cost_via_n = self.costs[n_idx] + dists_to_new[n_idx]
            if cost_via_n < min_c:
                if self.is_collision_free(self.nodes[n_idx], q_new):
                    best_p = n_idx
                    min_c = cost_via_n
        
        # Add Node
        new_idx = len(self.nodes)
        self.nodes.append(q_new)
        self.parents[new_idx] = best_p
        self.costs[new_idx] = min_c

        # 7. Rewire
        for n_idx in neighbor_indices:
            new_cost_for_neighbor = min_c + dists_to_new[n_idx]
            if new_cost_for_neighbor < self.costs[n_idx]:
                if self.is_collision_free(q_new, self.nodes[n_idx]):
                    self.parents[n_idx] = new_idx
                    self.costs[n_idx] = new_cost_for_neighbor

        # 8. Check Goal
        if np.linalg.norm(q_new - self.goal_q) < STEP_SIZE:
            self.goal_found = True
            if min_c < self.best_cost:
                self.best_cost = min_c
                self.best_goal_node = new_idx

    def get_path(self):
        if self.best_goal_node is None: return None
        idx = self.best_goal_node
        path_indices = []
        while idx is not None:
            path_indices.append(idx)
            idx = self.parents[idx]
        return [self.nodes[i] for i in path_indices[::-1]]

# ============================================================
# Main Loop
# ============================================================
def main():
    args = parse_args()
    os.makedirs(SAVE_ROOT, exist_ok=True)
    
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
    except Exception as e:
        print(f"Error loading XML: {e}"); return

    data = mujoco.MjData(model)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")

    print(f"Solving IK {args.start} -> {args.end}...")
    start_q = solve_ik(model, data, np.array(args.start), ee_id)
    goal_q  = solve_ik(model, data, np.array(args.end), ee_id)

    if start_q is None or goal_q is None:
        print("❌ IK Failed."); return

    viewer = mujoco.viewer.launch_passive(model, data) if args.viz else None
    
    successful_episodes = 0
    while successful_episodes < args.num:
        print(f"--- Episode {successful_episodes+1} ---")
        
        # Initialize Planner
        rrt = RRTStarOpt(model, start_q, goal_q)
        
        start_time = time.time()
        print(f"Planning (Max {args.max_samples} samples)...")
        
        for i in range(args.max_samples):
            rrt.step()
            
            if i % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"  Iter {i} | Nodes: {len(rrt.nodes)} | Best Cost: {rrt.best_cost:.2f} | Time: {elapsed:.1f}s")
                if viewer: viewer.sync()

        print(f"Planning Done in {time.time() - start_time:.2f}s")
        path = rrt.get_path()

        if path:
            print("Path found! Visualizing...")
            
            ep_dir = os.path.join(SAVE_ROOT, f"episode_{successful_episodes:03d}")
            os.makedirs(ep_dir, exist_ok=True)
            
            dense_path = [path[0]]
            for j in range(len(path)-1):
                steps = max(2, int(np.linalg.norm(path[j+1]-path[j])/WAYPOINT_RESOLUTION))
                for a in np.linspace(0, 1, steps, endpoint=False)[1:]: 
                    dense_path.append((1-a)*path[j]+a*path[j+1])
            dense_path.append(path[-1])

            solution_xyz = []
            for q in dense_path:
                data.qpos[:6]=q; mujoco.mj_forward(model, data)
                solution_xyz.append(data.body(ee_id).xpos.copy())

            with open(os.path.join(ep_dir, "path.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["idx","q1","q2","q3","q4","q5","q6","x","y","z"])
                for i, q in enumerate(dense_path): writer.writerow([i, *q, *solution_xyz[i]])

            if viewer:
                data.qpos[:6] = start_q
                mujoco.mj_forward(model, data)
                
                print("Visualizing ideal path (Static). Close window to exit.")
                while viewer.is_running():
                    viewer.user_scn.ngeom = 0
                    for point in solution_xyz:
                        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom: break
                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[viewer.user_scn.ngeom], 
                            mujoco.mjtGeom.mjGEOM_SPHERE, 
                            np.array([0.008, 0, 0]), point, np.eye(3).flatten(), np.array([1, 0, 0, 0.8])
                        )
                        viewer.user_scn.ngeom+=1
                    viewer.sync()
                    time.sleep(0.02)
            
            successful_episodes += 1
        else: 
            print("❌ Planning failed to find a path.")

    if viewer: viewer.close()

if __name__ == "__main__":
    main()