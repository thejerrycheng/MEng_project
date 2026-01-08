import mujoco
import mujoco.viewer
import numpy as np
import os
import sys
import argparse

# ---------- Configuration ----------
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURR_DIR)
MUJOCO_SIM_DIR = os.path.join(PROJECT_ROOT, "mujoco_sim")
XML_PATH = os.path.join(MUJOCO_SIM_DIR, "assets", "obstacle.xml")
SAVE_DIR = os.path.join(MUJOCO_SIM_DIR, "optimal_path_rrt")

START_Q = np.array([0.0, -0.8, -1.5, 0.0, 1.2, 0.0])
GOAL_Q  = np.array([0.8, 0.6, -1.0, 0.5, -0.5, 1.5])

MAX_SAMPLES = 1000
STEP_SIZE = 0.15
SEARCH_RADIUS = 0.6
MAX_VIZ_GEOMS = 4800

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
        self.all_samples = []

    def get_ee_pos(self, q):
        self.check_data.qpos[:6] = q
        mujoco.mj_forward(self.model, self.check_data)
        return self.check_data.body(self.ee_id).xpos.copy()

    def is_collision_free(self, q1, q2=None):
        def check(q):
            self.check_data.qpos[:6] = q
            mujoco.mj_forward(self.model, self.check_data)
            return self.check_data.ncon == 0

        if q2 is None:
            return check(q)

        for t in np.linspace(0, 1, 6):
            if not check(q1 + t * (q2 - q1)):
                return False
        return True

    def step(self):
        if self.samples_count >= MAX_SAMPLES:
            return False

        self.samples_count += 1
        p = np.random.rand()

        if self.goal_found and p < 0.5:
            q_rand = self.nodes[self.best_goal_node] + np.random.normal(0, 0.2, 6)
        elif p < 0.15:
            q_rand = self.goal_q
        else:
            q_rand = np.random.uniform(-np.pi, np.pi, 6)

        node_arr = np.array(self.nodes)
        nearest = int(np.argmin(np.linalg.norm(node_arr - q_rand, axis=1)))
        q_near = self.nodes[nearest]

        d = np.linalg.norm(q_rand - q_near)
        if d < 1e-9:
            return True

        q_new = q_near + (q_rand - q_near) / d * min(d, STEP_SIZE)
        if not self.is_collision_free(q_near, q_new):
            return True

        dists = np.linalg.norm(node_arr - q_new, axis=1)
        neighbors = np.where(dists < SEARCH_RADIUS)[0]

        best_p = nearest
        best_c = self.costs[nearest] + np.linalg.norm(q_new - q_near)

        for i in neighbors:
            c = self.costs[i] + np.linalg.norm(q_new - self.nodes[i])
            if c < best_c:
                best_p, best_c = i, c

        idx = len(self.nodes)
        self.nodes.append(q_new.copy())
        self.parents[idx] = best_p
        self.costs[idx] = best_c
        self.node_ee_positions.append(self.get_ee_pos(q_new))

        if np.linalg.norm(q_new - self.goal_q) < STEP_SIZE:
            self.goal_found = True
            if best_c < self.best_cost:
                self.best_cost = best_c
                self.best_goal_node = idx

        return True

    def extract_best_path_q(self):
        if self.best_goal_node is None:
            return None
        path = []
        cur = self.best_goal_node
        while cur is not None:
            path.append(self.nodes[cur])
            cur = self.parents[cur]
        return np.array(path[::-1])

# ---------- Final Path Viewer (STAYS OPEN) ----------
def visualize_final_path_in_viewer(model, path_q):
    data = mujoco.MjData(model)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")

    id_mat = np.eye(3).flatten()
    zero = np.zeros(3)

    # Precompute EE points
    tmp = mujoco.MjData(model)
    ee_pts = []
    for q in path_q:
        tmp.qpos[:6] = q
        mujoco.mj_forward(model, tmp)
        ee_pts.append(tmp.body(ee_id).xpos.copy())

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # --- play once ---
        for q in path_q:
            if not viewer.is_running():
                return
            data.qpos[:6] = q
            mujoco.mj_forward(model, data)

            viewer.user_scn.ngeom = 0
            for i in range(len(ee_pts) - 1):
                idx = viewer.user_scn.ngeom
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[idx],
                    mujoco.mjtGeom.mjGEOM_LINE,
                    zero, zero, id_mat,
                    np.array([1, 0, 0, 1], dtype=np.float32)
                )
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[idx],
                    mujoco.mjtGeom.mjGEOM_LINE,
                    0.01,
                    ee_pts[i],
                    ee_pts[i + 1]
                )
                viewer.user_scn.ngeom += 1

            viewer.sync()

        # --- hold final pose indefinitely ---
        while viewer.is_running():
            viewer.sync()

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    rrt = GlobalRRTStar(model, START_Q, GOAL_Q)

    if args.headless:
        while rrt.samples_count < MAX_SAMPLES:
            rrt.step()
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and rrt.samples_count < MAX_SAMPLES:
                for _ in range(40):
                    rrt.step()
                viewer.sync()

    path_q = rrt.extract_best_path_q()
    if path_q is None:
        print("[FAIL] No path found.")
        return

    np.save(os.path.join(SAVE_DIR, "waypoints.npy"), path_q)
    print(f"[OK] Saved {len(path_q)} waypoints.")

    visualize_final_path_in_viewer(model, path_q)
    print("[DONE] Viewer will remain open until you close it.")

if __name__ == "__main__":
    main()
