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
    parser.add_argument("--start", nargs=3, type=float, default=[0, 0.1, 0.3])
    parser.add_argument("--end", nargs=3, type=float, default=[0, 0.5, 0.3])
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--bubble_radius", type=float, default=0.075) 
    return parser.parse_args()

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURR_DIR)
XML_PATH = os.path.join(PROJECT_ROOT, "mujoco_sim", "assets", "scene2.xml")
SAVE_ROOT = os.path.join(os.getcwd(), "rrt_generated_dataset")

STEP_SIZE = 0.2          
SEARCH_RADIUS = 0.5       
WAYPOINT_RESOLUTION = 0.02 

# ============================================================
# Robust IK: Z-axis Aligned with Y-positive
# ============================================================
def solve_ik(model, data, target_xyz, body_id):
    """
    Solves IK where the local Z-axis of the EE body faces the World Y+ axis.
    """
    best_q = None
    min_error = float('inf')
    
    # Target orientation vector: Y+ in world space
    target_y_positive = np.array([0, 1, 0])

    for attempt in range(25): 
        # Randomized seeds to escape local minima
        data.qpos[:6] = np.random.uniform(-3, 3, 6)
        mujoco.mj_forward(model, data)
        
        for _ in range(400):
            # 1. Position Jacobian
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, None, body_id)
            
            # 2. Rotation Jacobian (Angular Velocity)
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, None, jacr, body_id)
            
            # Errors
            curr_xyz = data.body(body_id).xpos
            curr_mat = data.body(body_id).xmat.reshape(3,3)
            
            pos_err = target_xyz - curr_xyz
            
            # ROTATION LOGIC:
            # Align local Z-axis (curr_mat[:, 2]) with World Y+ ([0, 1, 0])
            ee_z_axis = curr_mat[:, 2] 
            rot_err = np.cross(ee_z_axis, target_y_positive)

            # Combine errors (Full 6-DOF)
            J = np.vstack([jacp[:, :6], jacr[:, :6]])
            full_err = np.concatenate([pos_err, rot_err * 0.8]) # Stronger rotation bias
            
            # Damped Least Squares Solve
            dq = np.linalg.solve(J.T @ J + 0.01 * np.eye(6), J.T @ full_err)
            data.qpos[:6] += dq * 0.4
            data.qpos[:6] = np.clip(data.qpos[:6], -3.1, 3.1)
            mujoco.mj_forward(model, data)
            
            if np.linalg.norm(pos_err) < 1e-4 and np.linalg.norm(rot_err) < 1e-3:
                # Success check: also ensure no collision for starting poses
                if data.ncon == 0:
                    return data.qpos[:6].copy()
                
    return best_q if min_error < 0.01 else None

# ============================================================
# RRT* Implementation (Proximity-Based)
# ============================================================
class RRTStar:
    def __init__(self, model, start_q, goal_q, bubble_radius):
        self.model, self.data = model, mujoco.MjData(model)
        self.start_q, self.goal_q = start_q, goal_q
        self.nodes, self.parents, self.costs = [start_q.copy()], {0: None}, {0: 0.0}
        self.samples_count = 0
        self.goal_found, self.best_goal_node, self.best_cost = False, None, float("inf")
        self.bubble_radius = bubble_radius
        self.obstacle_ids = [i for i in range(model.ngeom) if "obstacle" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").lower()]

    def is_collision_free(self, q1, q2=None):
        def check(q):
            self.data.qpos[:6] = q
            mujoco.mj_forward(self.model, self.data)
            # Body chain for 15cm diameter capsules
            body_chain = ["shoulder", "arm_link1", "elbow", "arm_link2", "wrist1", "wrist2", "ee_mount"]
            for i in range(len(body_chain) - 1):
                p1, p2 = self.data.body(body_chain[i]).xpos, self.data.body(body_chain[i+1]).xpos
                mid = (p1 + p2) / 2
                for gid in self.obstacle_ids:
                    # Sphere-proxy distance check at link points
                    for pt in [p1, mid, p2]:
                        dist = np.linalg.norm(pt - self.data.geom_xpos[gid]) - np.max(self.model.geom_size[gid])
                        if dist < self.bubble_radius: return False
            return True
        if q2 is None: return check(q1)
        for t in np.linspace(0, 1, 8):
            if not check(q1 + t * (q2 - q1)): return False
        return True

    def step(self):
        self.samples_count += 1
        q_rand = self.goal_q if np.random.rand() < 0.2 else np.random.uniform(-np.pi, np.pi, 6)
        node_arr = np.array(self.nodes)
        nearest_idx = np.argmin(np.linalg.norm(node_arr - q_rand, axis=1))
        q_near = self.nodes[nearest_idx]
        diff = q_rand - q_near
        q_new = q_near + (diff / np.linalg.norm(diff)) * min(np.linalg.norm(diff), STEP_SIZE)
        if not self.is_collision_free(q_near, q_new): return
        
        dists = np.linalg.norm(node_arr - q_new, axis=1)
        neighbors = np.where(dists < SEARCH_RADIUS)[0]
        best_p, min_c = nearest_idx, self.costs[nearest_idx] + np.linalg.norm(q_new - q_near)
        for n_idx in neighbors:
            if self.costs[n_idx] + np.linalg.norm(q_new - self.nodes[n_idx]) < min_c and self.is_collision_free(self.nodes[n_idx], q_new):
                best_p, min_c = n_idx, self.costs[n_idx] + np.linalg.norm(q_new - self.nodes[n_idx])
        
        new_idx = len(self.nodes); self.nodes.append(q_new); self.parents[new_idx], self.costs[new_idx] = best_p, min_c
        for n_idx in neighbors:
            if min_c + np.linalg.norm(self.nodes[n_idx] - q_new) < self.costs[n_idx] and self.is_collision_free(q_new, self.nodes[n_idx]):
                self.parents[n_idx], self.costs[n_idx] = new_idx, min_c + np.linalg.norm(self.nodes[n_idx] - q_new)
        
        if np.linalg.norm(q_new - self.goal_q) < STEP_SIZE:
            self.goal_found = True
            if min_c < self.best_cost: self.best_cost, self.best_goal_node = min_c, new_idx

    def get_path(self):
        if self.best_goal_node is None: return None
        idx, curr = [], self.best_goal_node
        while curr is not None: idx.append(curr); curr = self.parents[curr]
        return [self.nodes[i] for i in idx[::-1]]

# ============================================================
# Main Loop
# ============================================================
def draw_capsule_bubbles(viewer, data, radius):
    body_chain = ["shoulder", "arm_link1", "elbow", "arm_link2", "wrist1", "wrist2", "ee_mount"]
    for i in range(len(body_chain) - 1):
        mujoco.mjv_connector(viewer.user_scn.geoms[viewer.user_scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, radius, data.body(body_chain[i]).xpos, data.body(body_chain[i+1]).xpos)
        viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba = [0, 1, 1, 0.2]
        viewer.user_scn.ngeom += 1

def main():
    args = parse_args()
    os.makedirs(SAVE_ROOT, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")

    print(f"Solving IK (Z facing Y+) for Start {args.start}...")
    start_q = solve_ik(model, data, np.array(args.start), ee_id)
    goal_q  = solve_ik(model, data, np.array(args.end), ee_id)

    if start_q is None or goal_q is None:
        print("❌ IK Failed."); return

    viewer = mujoco.viewer.launch_passive(model, data) if args.viz else None
    successful_episodes = 0
    while successful_episodes < args.num:
        print(f"--- Episode {successful_episodes+1} ---")
        rrt = RRTStar(model, start_q, goal_q, args.bubble_radius)
        for i in range(args.max_samples):
            rrt.step()
            if rrt.goal_found and i > 1500: break
            if viewer and i % 100 == 0: viewer.sync()

        path = rrt.get_path()
        if path:
            ep_dir = os.path.join(SAVE_ROOT, f"episode_{successful_episodes:03d}"); os.makedirs(ep_dir, exist_ok=True)
            dense_path = [path[0]]
            for j in range(len(path)-1):
                steps = max(2, int(np.linalg.norm(path[j+1]-path[j])/WAYPOINT_RESOLUTION))
                for a in np.linspace(0, 1, steps, endpoint=False)[1:]: dense_path.append((1-a)*path[j]+a*path[j+1])
            dense_path.append(path[-1])

            # Pre-calculate trail for replay
            trail_xyz = []
            for q in dense_path:
                data.qpos[:6]=q; mujoco.mj_forward(model, data)
                trail_xyz.append(data.body(ee_id).xpos.copy())

            if viewer:
                for idx, q in enumerate(dense_path):
                    data.qpos[:6]=q; mujoco.mj_forward(model, data); viewer.user_scn.ngeom=0
                    draw_capsule_bubbles(viewer, data, args.bubble_radius)
                    for j in range(idx+1):
                        mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.005, 0, 0]), trail_xyz[j], np.eye(3).flatten(), np.array([0, 1, 0, 0.4]))
                        viewer.user_scn.ngeom+=1
                    viewer.sync(); time.sleep(0.01)

            # Write data to CSV
            with open(os.path.join(ep_dir, "path.csv"), "w", newline="") as f:
                writer = csv.writer(f); writer.writerow(["idx","q1","q2","q3","q4","q5","q6","x","y","z"])
                for i, q in enumerate(dense_path): writer.writerow([i, *q, *trail_xyz[i]])
            successful_episodes += 1
        else: print("❌ Planning failed.")

    if viewer: viewer.close()

if __name__ == "__main__":
    main()