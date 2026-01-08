import mujoco
import numpy as np
import os
import csv
import time

# ============================================================
# Configuration
# ============================================================
# ------------------------------------------------------------
# Robust project root resolution
# ------------------------------------------------------------
CURR_DIR = os.path.dirname(os.path.abspath(__file__))          # classical_planner/
PROJECT_ROOT = os.path.dirname(CURR_DIR)                       # MEng_project/

XML_PATH = os.path.join(
    PROJECT_ROOT,
    "mujoco_sim",
    "assets",
    "obstacle.xml"
)

# Save under CURRENT FOLDER
SAVE_ROOT = os.path.join(os.getcwd(), "random_rrt_dataset")
os.makedirs(SAVE_ROOT, exist_ok=True)

MAX_EPISODES = 100
MAX_RRT_SAMPLES = 20000

# Densification (rad per step in joint space)
WAYPOINT_RESOLUTION = 0.02   # smaller = denser path

# Workspace (positive X/Y only)
WS_X = (0.3, 0.9)
WS_Y = (0.1, 0.9)

# ============================================================
# CSV Utility
# ============================================================
def save_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

# ============================================================
# Obstacle Manager (FIXED geometry, random pose)
# ============================================================
class ObstacleManager:
    """
    Matches MJCF:

    <geom name="cylinder_obstacle" type="cylinder" size="0.1 0.125"/>
    <geom name="cube_obstacle"     type="box"      size="0.05 0.125 0.125"/>
    """

    def __init__(self, model):
        self.model = model

        self.obstacles = {
            "cylinder_obstacle": {
                "type": "cylinder",
                "size": np.array([0.1, 0.125]),  # radius, half-height
            },
            "cube_obstacle": {
                "type": "box",
                "size": np.array([0.05, 0.125, 0.125]),
            }
        }

        self.geom_ids = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in self.obstacles
        }

    def randomize(self):
        placements = []

        for name, cfg in self.obstacles.items():
            gid = self.geom_ids[name]
            if gid < 0:
                continue

            x = np.random.uniform(*WS_X)
            y = np.random.uniform(*WS_Y)

            # Place on floor
            z = cfg["size"][-1]

            yaw = np.random.uniform(-np.pi, np.pi)

            self.model.geom_pos[gid] = [x, y, z]

            quat = np.zeros(4)
            mujoco.mju_axisAngle2Quat(quat, np.array([0, 0, 1]), yaw)
            self.model.geom_quat[gid] = quat

            placements.append({
                "name": name,
                "type": cfg["type"],
                "x": x,
                "y": y,
                "z": z,
                "yaw": yaw,
                "size": cfg["size"].tolist()
            })

        return placements

# ============================================================
# Global RRT* (Joint Space, Offline)
# ============================================================
class GlobalRRTStar:
    def __init__(self, model, start_q, goal_q):
        self.model = model
        self.check_data = mujoco.MjData(model)

        self.nodes = [start_q.copy()]
        self.parents = {0: None}
        self.costs = {0: 0.0}

        self.goal_q = goal_q.copy()
        self.samples_count = 0
        self.goal_found = False
        self.best_goal_node = None
        self.best_cost = float("inf")

    def is_collision_free(self, q):
        self.check_data.qpos[:6] = q
        mujoco.mj_forward(self.model, self.check_data)
        return self.check_data.ncon == 0

    def step(self):
        if self.samples_count >= MAX_RRT_SAMPLES:
            return False

        self.samples_count += 1

        if self.goal_found and np.random.rand() < 0.6:
            path = self.get_path_indices()
            q_base = self.nodes[np.random.choice(path)]
            q_rand = q_base + np.random.normal(0, 0.25, 6)
        elif np.random.rand() < 0.1:
            q_rand = self.goal_q
        else:
            q_rand = np.random.uniform(-np.pi, np.pi, 6)

        node_arr = np.array(self.nodes)
        nearest_idx = np.argmin(np.linalg.norm(node_arr - q_rand, axis=1))
        q_near = self.nodes[nearest_idx]

        diff = q_rand - q_near
        dist = np.linalg.norm(diff)
        if dist < 1e-6:
            return True

        q_new = q_near + 0.2 * diff / dist

        if not self.is_collision_free(q_new):
            return True

        dists = np.linalg.norm(node_arr - q_new, axis=1)
        neighbors = np.where(dists < 0.8)[0]

        best_parent = nearest_idx
        min_cost = self.costs[nearest_idx] + np.linalg.norm(q_new - q_near)

        for n in neighbors:
            c = self.costs[n] + np.linalg.norm(q_new - self.nodes[n])
            if c < min_cost:
                best_parent, min_cost = n, c

        new_idx = len(self.nodes)
        self.nodes.append(q_new)
        self.parents[new_idx] = best_parent
        self.costs[new_idx] = min_cost

        for n in neighbors:
            c = self.costs[new_idx] + np.linalg.norm(self.nodes[n] - q_new)
            if c < self.costs[n]:
                self.parents[n] = new_idx
                self.costs[n] = c

        if np.linalg.norm(q_new - self.goal_q) < 0.2:
            self.goal_found = True
            if min_cost < self.best_cost:
                self.best_cost = min_cost
                self.best_goal_node = new_idx

        return True

    def get_path_indices(self):
        if self.best_goal_node is None:
            return []

        path = []
        curr = self.best_goal_node
        while curr is not None:
            path.append(curr)
            curr = self.parents[curr]
        return path[::-1]

# ============================================================
# Path Densification (CRITICAL)
# ============================================================
def densify_joint_path(joint_path, resolution=0.02):
    dense = [joint_path[0]]

    for q0, q1 in zip(joint_path[:-1], joint_path[1:]):
        dist = np.linalg.norm(q1 - q0)
        steps = max(2, int(dist / resolution))
        for a in np.linspace(0, 1, steps, endpoint=False)[1:]:
            dense.append((1 - a) * q0 + a * q1)

    dense.append(joint_path[-1])
    return dense

# ============================================================
# Main Dataset Generation
# ============================================================
def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")

    obs_mgr = ObstacleManager(model)

    for ep in range(MAX_EPISODES):
        ep_dir = os.path.join(SAVE_ROOT, f"episode_{ep:03d}")
        os.makedirs(ep_dir, exist_ok=True)

        # 1. Randomize obstacles
        obstacles = obs_mgr.randomize()

        # 2. Start / Goal
        start_q = np.array([0.0, -0.6, -1.2, 0.0, 1.0, 0.0])
        goal_q  = np.array([0.6,  0.4, -0.8, 0.3, 0.6, 0.0])

        # 3. RRT*
        rrt = GlobalRRTStar(model, start_q, goal_q)
        t0 = time.time()

        while time.time() - t0 < 10.0:
            rrt.step()
            if rrt.goal_found:
                break

        if not rrt.goal_found:
            print(f"[Episode {ep}] ❌ planning failed")
            continue

        sparse_path = [rrt.nodes[i] for i in rrt.get_path_indices()]
        joint_path = densify_joint_path(sparse_path, WAYPOINT_RESOLUTION)

        # 4. EE path
        tmp = mujoco.MjData(model)
        ee_path = []
        for q in joint_path:
            tmp.qpos[:6] = q
            mujoco.mj_forward(model, tmp)
            ee_path.append(tmp.body(ee_id).xpos.copy())

        # 5. Save CSVs
        save_csv(
            os.path.join(ep_dir, "path.csv"),
            ["idx","q1","q2","q3","q4","q5","q6","x","y","z"],
            [[i, *joint_path[i], *ee_path[i]] for i in range(len(joint_path))]
        )

        save_csv(
            os.path.join(ep_dir, "obstacles.csv"),
            ["name","type","x","y","z","yaw","size"],
            [[o["name"], o["type"], o["x"], o["y"], o["z"], o["yaw"], o["size"]] for o in obstacles]
        )

        save_csv(
            os.path.join(ep_dir, "meta.csv"),
            ["success","samples","cost","num_waypoints"],
            [[1, rrt.samples_count, rrt.best_cost, len(joint_path)]]
        )

        print(f"[Episode {ep}] ✅ saved {len(joint_path)} waypoints")

if __name__ == "__main__":
    main()
