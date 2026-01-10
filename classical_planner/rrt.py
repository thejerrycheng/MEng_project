import mujoco
import mujoco.viewer
import numpy as np
import os
import sys
import argparse
import time

# ---------- Configuration ----------
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURR_DIR)
MUJOCO_SIM_DIR = os.path.join(PROJECT_ROOT, "mujoco_sim")
XML_PATH = os.path.join(MUJOCO_SIM_DIR, "assets", "scene2.xml")
SAVE_ROOT = os.path.join(MUJOCO_SIM_DIR, "optimal_path_rrt")

MAX_SAMPLES = 10000 
STEP_SIZE = 0.15
SEARCH_RADIUS = 0.6
BUBBLE_RADIUS = 0.075 # 15cm diameter

# ---------- IK Solver for Parsing ----------
def solve_ik(model, data, target_xyz, body_name="ee_mount"):
    """Finds joint angles for the requested --start and --end positions."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    # Random restarts to find a reachable solution
    for attempt in range(15):
        data.qpos[:6] = np.random.uniform(-2, 2, 6) if attempt > 0 else np.array([0, -0.8, -1.5, 0, 1.2, 0])
        mujoco.mj_forward(model, data)
        for _ in range(200):
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, None, body_id)
            err = target_xyz - data.body(body_id).xpos
            if np.linalg.norm(err) < 1e-4:
                return data.qpos[:6].copy()
            # Damped Least Squares
            J = jacp[:, :6]
            dq = J.T @ np.linalg.solve(J @ J.T + 0.01 * np.eye(3), err)
            data.qpos[:6] += dq * 0.5
            mujoco.mj_forward(model, data)
    return None

# ---------- RRT* ----------
class GlobalRRTStar:
    def __init__(self, model, start_q, goal_q):
        self.model = model
        self.check_data = mujoco.MjData(model)
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")

        self.nodes = [start_q.copy()]
        self.parents = {0: None}
        self.costs = {0: 0.0}
        self.node_ee_positions = [self.get_ee_pos(start_q)]

        self.goal_q = goal_q.copy()
        self.samples_count = 0
        self.goal_found = False
        self.best_goal_node = None
        self.best_cost = float("inf")
        
        self.obstacle_ids = [i for i in range(model.ngeom) 
                             if "obstacle" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").lower()]

    def get_ee_pos(self, q):
        self.check_data.qpos[:6] = q
        mujoco.mj_forward(self.model, self.check_data)
        return self.check_data.body(self.ee_id).xpos.copy()

    def is_collision_free(self, q1, q2=None):
        def check_capsule_collision(q):
            self.check_data.qpos[:6] = q
            mujoco.mj_forward(self.model, self.check_data)
            body_chain = ["shoulder", "arm_link1", "elbow", "arm_link2", "wrist1", "wrist2", "ee_mount"]
            for i in range(len(body_chain) - 1):
                p1 = self.check_data.body(body_chain[i]).xpos
                p2 = self.check_data.body(body_chain[i+1]).xpos
                for pt in [p1, p2, (p1 + p2) / 2.0]:
                    for gid in self.obstacle_ids:
                        dist = np.linalg.norm(pt - self.check_data.geom_xpos[gid]) - np.max(self.model.geom_size[gid])
                        if dist < BUBBLE_RADIUS: return False
            return True

        if q2 is None: return check_capsule_collision(q1)
        for t in np.linspace(0, 1, 8):
            if not check_capsule_collision(q1 + t * (q2 - q1)): return False
        return True

    def get_path_indices(self):
        if self.best_goal_node is None: return []
        path, curr = [], self.best_goal_node
        while curr is not None:
            path.append(curr); curr = self.parents[curr]
        return path

    def step(self):
        if self.samples_count >= MAX_SAMPLES: return False
        self.samples_count += 1
        p = np.random.random()
        # Exploitation vs Exploration
        if self.goal_found and p < 0.50:
            best_indices = self.get_path_indices()
            q_base = self.nodes[np.random.choice(best_indices)]
            q_rand = np.clip(q_base + np.random.normal(0, 0.2, 6), -3.14, 3.14)
        elif p < 0.15:
            q_rand = self.goal_q
        else:
            q_rand = np.random.uniform(-3.14, 3.14, 6)

        node_arr = np.array(self.nodes)
        nearest_idx = int(np.argmin(np.linalg.norm(node_arr - q_rand, axis=1)))
        q_near = self.nodes[nearest_idx]
        diff = q_rand - q_near
        d = float(np.linalg.norm(diff))
        if d < 1e-9: return True

        q_new = q_near + (diff / d) * min(d, STEP_SIZE)
        if not self.is_collision_free(q_near, q_new): return True

        dists_to_new = np.linalg.norm(node_arr - q_new, axis=1)
        neighbors = np.where(dists_to_new < SEARCH_RADIUS)[0]
        best_p, min_c = nearest_idx, self.costs[nearest_idx] + np.linalg.norm(q_new - q_near)
        for n_idx in neighbors:
            c_via_n = self.costs[int(n_idx)] + np.linalg.norm(q_new - self.nodes[int(n_idx)])
            if c_via_n < min_c and self.is_collision_free(self.nodes[int(n_idx)], q_new):
                best_p, min_c = int(n_idx), float(c_via_n)

        new_idx = len(self.nodes)
        self.nodes.append(q_new.copy()); self.parents[new_idx], self.costs[new_idx] = best_p, min_c
        self.node_ee_positions.append(self.get_ee_pos(q_new))

        for n_idx in neighbors:
            n_idx = int(n_idx)
            c_via_new = min_c + np.linalg.norm(self.nodes[n_idx] - q_new)
            if c_via_new < self.costs[n_idx] and self.is_collision_free(q_new, self.nodes[n_idx]):
                self.parents[n_idx], self.costs[n_idx] = new_idx, float(c_via_new)

        if np.linalg.norm(q_new - self.goal_q) < STEP_SIZE:
            self.goal_found = True
            if min_c < self.best_cost:
                self.best_cost, self.best_goal_node = float(min_c), new_idx
        return True

# ---------- Visualization Helpers ----------
def draw_capsule_bubbles(viewer, data):
    body_chain = ["shoulder", "arm_link1", "elbow", "arm_link2", "wrist1", "wrist2", "ee_mount"]
    for i in range(len(body_chain) - 1):
        p1, p2 = data.body(body_chain[i]).xpos, data.body(body_chain[i+1]).xpos
        mujoco.mjv_connector(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                             mujoco.mjtGeom.mjGEOM_CAPSULE, BUBBLE_RADIUS, p1, p2)
        viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba = [0, 1, 1, 0.2]
        viewer.user_scn.ngeom += 1

# ---------- Main Execution ----------
def main():
    parser = argparse.ArgumentParser(description="RRT* with Cartesian Parsers")
    parser.add_argument("--start", nargs=3, type=float, default=[0, 0.1, 0.3])
    parser.add_argument("--end", nargs=3, type=float, default=[0, 0.5, 0.3])
    args = parser.parse_args()

    os.makedirs(SAVE_ROOT, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    print(f"Solving IK for Start {args.start} and End {args.end}...")
    start_q = solve_ik(model, data, np.array(args.start))
    goal_q  = solve_ik(model, data, np.array(args.end))

    if start_q is None or goal_q is None:
        print("❌ IK Failed. Try points closer to the robot base."); return

    rrt = GlobalRRTStar(model, start_q, goal_q)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while rrt.samples_count < MAX_SAMPLES and viewer.is_running():
            for _ in range(40): rrt.step()
            viewer.user_scn.ngeom = 0
            opt_indices = rrt.get_path_indices()
            for child, parent in rrt.parents.items():
                if parent is None: continue
                color = [1, 0, 0, 1] if child in opt_indices and parent in opt_indices else [0, 1, 0, 0.1]
                mujoco.mjv_connector(viewer.user_scn.geoms[viewer.user_scn.ngeom], mujoco.mjtGeom.mjGEOM_LINE, 0.002, rrt.node_ee_positions[parent], rrt.node_ee_positions[child])
                viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba = color
                viewer.user_scn.ngeom += 1
            viewer.sync()

        indices = rrt.get_path_indices()
        if indices:
            path_q = [rrt.nodes[i] for i in indices[::-1]]
            print("Path Found! Replaying...")
            while viewer.is_running():
                for q in path_q:
                    if not viewer.is_running(): break
                    data.qpos[:6] = q; mujoco.mj_forward(model, data)
                    viewer.user_scn.ngeom = 0
                    draw_capsule_bubbles(viewer, data)
                    viewer.sync(); time.sleep(0.05)
                time.sleep(1.0)
        else: print("No path found.")

if __name__ == "__main__":
    main()