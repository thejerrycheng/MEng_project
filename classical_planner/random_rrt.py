import mujoco
import mujoco.viewer
import numpy as np
import os
import csv
import time
import argparse

# ============================================================
# Configuration & Arguments
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", nargs=3, type=float, default=[0.2, 0.0, 0.5])
    parser.add_argument("--end", nargs=3, type=float, default=[0.7, 0.0, 0.5])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--viz", action="store_true", help="Visualize the IK/Planning process")
    return parser.parse_args()

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURR_DIR)
XML_PATH = os.path.join(PROJECT_ROOT, "mujoco_sim", "assets", "scene2.xml")

SAVE_ROOT = os.path.join(os.getcwd(), "random_rrt_dataset2")
os.makedirs(SAVE_ROOT, exist_ok=True)

MAX_RRT_SAMPLES = 20000
WAYPOINT_RESOLUTION = 0.02 
WS_X, WS_Y = (0.3, 0.9), (0.1, 0.9)

# ============================================================
# Robust IK Utility
# ============================================================
def solve_ik(model, data, target_xyz, body_id, max_retries=5):
    """Robust IK with multiple random restarts and damped least squares."""
    # Try different initial joint positions if the default fails
    seeds = [
        np.array([0.0, -0.8, -1.5, 0.0, 1.2, 0.0]), # Default "bent" pose
        np.zeros(6),                                # Zero pose
        np.array([0.0, -0.2, -0.2, 0.0, 0.5, 0.0]), # Reaching up
    ]
    
    for attempt in range(max_retries):
        if attempt < len(seeds):
            data.qpos[:6] = seeds[attempt]
        else:
            data.qpos[:6] = np.random.uniform(-1.5, 1.5, 6)
            
        mujoco.mj_forward(model, data)
        
        for i in range(500):
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, None, body_id)
            curr_xyz = data.body(body_id).xpos
            error = target_xyz - curr_xyz
            
            if np.linalg.norm(error) < 1e-4:
                return data.qpos[:6].copy()
            
            # Damped Least Squares (Levenberg-Marquardt)
            J = jacp[:, :6]
            diag = 0.01 * np.eye(3)
            step = J.T @ np.linalg.solve(J @ J.T + diag, error)
            
            data.qpos[:6] += step * 0.5
            # Clamp to prevent joint limit issues during IK
            data.qpos[:6] = np.clip(data.qpos[:6], -3.1, 3.1)
            mujoco.mj_forward(model, data)
            
    return None

# ============================================================
# Obstacle Manager & RRT*
# ============================================================
class ObstacleManager:
    def __init__(self, model):
        self.model = model
        self.obstacles = {
            "cylinder_obstacle": {"size": np.array([0.1, 0.125])},
            "cube_obstacle": {"size": np.array([0.05, 0.125, 0.125])}
        }
        self.geom_ids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in self.obstacles}

    def randomize(self):
        placements = []
        for name, cfg in self.obstacles.items():
            gid = self.geom_ids[name]
            if gid < 0: continue
            x, y = np.random.uniform(*WS_X), np.random.uniform(*WS_Y)
            z = cfg["size"][-1]
            yaw = np.random.uniform(-np.pi, np.pi)
            self.model.geom_pos[gid] = [x, y, z]
            quat = np.zeros(4)
            mujoco.mju_axisAngle2Quat(quat, np.array([0, 0, 1]), yaw)
            self.model.geom_quat[gid] = quat
            placements.append({"name": name, "x": x, "y": y, "z": z, "yaw": yaw})
        return placements

class GlobalRRTStar:
    def __init__(self, model, start_q, goal_q):
        self.model, self.check_data = model, mujoco.MjData(model)
        self.nodes, self.parents, self.costs = [start_q.copy()], {0: None}, {0: 0.0}
        self.goal_q, self.samples_count = goal_q.copy(), 0
        self.goal_found, self.best_goal_node, self.best_cost = False, None, float("inf")

    def is_collision_free(self, q):
        self.check_data.qpos[:6] = q
        mujoco.mj_forward(self.model, self.check_data)
        return self.check_data.ncon == 0

    def step(self):
        if self.samples_count >= MAX_RRT_SAMPLES: return False
        self.samples_count += 1
        
        if self.goal_found and np.random.rand() < 0.6:
            q_base = self.nodes[np.random.choice(self.get_path_indices())]
            q_rand = q_base + np.random.normal(0, 0.2, 6)
        elif np.random.rand() < 0.1: q_rand = self.goal_q
        else: q_rand = np.random.uniform(-np.pi, np.pi, 6)

        node_arr = np.array(self.nodes)
        nearest_idx = np.argmin(np.linalg.norm(node_arr - q_rand, axis=1))
        q_near = self.nodes[nearest_idx]
        diff = q_rand - q_near
        dist = np.linalg.norm(diff)
        if dist < 1e-6: return True

        q_new = q_near + 0.2 * (diff / dist)
        if not self.is_collision_free(q_new): return True

        dists = np.linalg.norm(node_arr - q_new, axis=1)
        neighbors = np.where(dists < 0.8)[0]
        best_parent, min_cost = nearest_idx, self.costs[nearest_idx] + np.linalg.norm(q_new - q_near)

        for n in neighbors:
            c = self.costs[n] + np.linalg.norm(q_new - self.nodes[n])
            if c < min_cost: best_parent, min_cost = n, c

        new_idx = len(self.nodes)
        self.nodes.append(q_new); self.parents[new_idx] = best_parent; self.costs[new_idx] = min_cost

        if np.linalg.norm(q_new - self.goal_q) < 0.25:
            self.goal_found = True
            if min_cost < self.best_cost:
                self.best_cost, self.best_goal_node = min_cost, new_idx
        return True

    def get_path_indices(self):
        if self.best_goal_node is None: return []
        path, curr = [], self.best_goal_node
        while curr is not None:
            path.append(curr); curr = self.parents[curr]
        return path[::-1]

# ============================================================
# Main Logic
# ============================================================
def main():
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")
    obs_mgr = ObstacleManager(model)

    # 1. Solve IK
    print(f"Solving IK for Start {args.start} and End {args.end}...")
    start_q = solve_ik(model, data, np.array(args.start), ee_id)
    goal_q  = solve_ik(model, data, np.array(args.end), ee_id)

    if start_q is None or goal_q is None:
        print("❌ IK Failed. Trying a safer Z height might help, or check robot reachable workspace.")
        return

    # 2. Episode Loop
    viewer = None
    if args.viz:
        viewer = mujoco.viewer.launch_passive(model, data)

    for ep in range(args.episodes):
        ep_dir = os.path.join(SAVE_ROOT, f"episode_{ep:03d}")
        os.makedirs(ep_dir, exist_ok=True)
        obstacles = obs_mgr.randomize()

        rrt = GlobalRRTStar(model, start_q, goal_q)
        t_start = time.time()
        
        while time.time() - t_start < 10.0:
            rrt.step()
            if rrt.goal_found and rrt.samples_count > 1000: break
            if viewer and viewer.is_running(): viewer.sync()

        if not rrt.goal_found:
            print(f"[Ep {ep}] ❌ Failed to find path.")
            continue

        # Extract and Densify
        sparse_indices = rrt.get_path_indices()
        sparse_path = [rrt.nodes[i] for i in sparse_indices]
        dense_path = [sparse_path[0]]
        for q0, q1 in zip(sparse_path[:-1], sparse_path[1:]):
            steps = max(2, int(np.linalg.norm(q1 - q0) / WAYPOINT_RESOLUTION))
            for a in np.linspace(0, 1, steps, endpoint=False)[1:]:
                dense_path.append((1 - a) * q0 + a * q1)
        dense_path.append(sparse_path[-1])

        # CSV Logging
        rows = []
        for i, q in enumerate(dense_path):
            data.qpos[:6] = q
            mujoco.mj_forward(model, data)
            rows.append([i, *q, *data.body(ee_id).xpos])
            if viewer and viewer.is_running(): viewer.sync()

        with open(os.path.join(ep_dir, "path.csv"), "w", newline="") as f:
            csv.writer(f).writerow(["idx","q1","q2","q3","q4","q5","q6","x","y","z"])
            csv.writer(f).writerows(rows)
            
        print(f"[Ep {ep}] ✅ Success: {len(dense_path)} steps.")

    if viewer: viewer.close()

if __name__ == "__main__":
    main()