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

MAX_SAMPLES = 10000 
STEP_SIZE = 0.15
EE_SPHERE_RADIUS = 0.04 # 15cm diameter

# ---------- IK Solver ----------
def solve_ik(model, data, target_xyz):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")
    for attempt in range(20):
        data.qpos[:6] = np.random.uniform(-3, 3, 6) if attempt > 0 else np.zeros(6)
        mujoco.mj_forward(model, data)
        for _ in range(200):
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, None, body_id)
            err = target_xyz - data.body(body_id).xpos
            if np.linalg.norm(err) < 1e-4: return data.qpos[:6].copy()
            J = jacp[:, :6]
            dq = J.T @ np.linalg.solve(J @ J.T + 0.01 * np.eye(3), err)
            data.qpos[:6] += dq * 0.5
            mujoco.mj_forward(model, data)
    return None

# ---------- Tree Class ----------
class Tree:
    def __init__(self, root, model, ee_id):
        self.nodes = [root]
        self.parents = {0: None}
        self.ee_positions = [self._get_ee(root, model, ee_id)]
    
    def _get_ee(self, q, model, ee_id):
        d = mujoco.MjData(model)
        d.qpos[:6] = q
        mujoco.mj_forward(model, d)
        return d.body(ee_id).xpos.copy()

    def add_node(self, q, parent_idx, model, ee_id):
        idx = len(self.nodes)
        self.nodes.append(q)
        self.parents[idx] = parent_idx
        self.ee_positions.append(self._get_ee(q, model, ee_id))
        return idx

# ---------- RRT-Connect Planner ----------
class RRTConnect:
    def __init__(self, model, start_q, goal_q):
        self.model = model
        self.check_data = mujoco.MjData(model)
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")
        self.treeA = Tree(start_q, model, self.ee_id)
        self.treeB = Tree(goal_q, model, self.ee_id)
        # Pre-identify obstacles for distance checking
        self.obs_ids = [i for i in range(model.ngeom) if "obstacle" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").lower()]

    def is_collision_free(self, q):
        self.check_data.qpos[:6] = q
        mujoco.mj_forward(self.model, self.check_data)
        
        # 1. Custom End-Effector Sphere Check
        ee_pos = self.check_data.body(self.ee_id).xpos
        for gid in self.obs_ids:
            obs_pos = self.check_data.geom_xpos[gid]
            # Max geom size used as obstacle radius approximation
            obs_radius = np.max(self.model.geom_size[gid])
            dist = np.linalg.norm(ee_pos - obs_pos) - obs_radius
            if dist < EE_SPHERE_RADIUS:
                return False

        # 2. Standard Mesh Collision for the rest of the arm (ignoring EE mesh)
        # We check ncon but filter out contacts involving the end-effector geoms if necessary.
        # For simplicity, if Sphere check passes, we ensure no other part hits.
        if self.check_data.ncon > 0:
            # You could add logic here to filter contacts by geom ID if the 15cm sphere 
            # overlaps with the arm's own wrist geoms.
            return False
            
        return True

    def extend(self, tree, q_target):
        node_arr = np.array(tree.nodes)
        nearest_idx = np.argmin(np.linalg.norm(node_arr - q_target, axis=1))
        q_near = tree.nodes[nearest_idx]
        diff = q_target - q_near
        dist = np.linalg.norm(diff)
        q_new = q_near + (diff / dist) * min(dist, STEP_SIZE)
        if self.is_collision_free(q_new):
            new_idx = tree.add_node(q_new, nearest_idx, self.model, self.ee_id)
            if np.linalg.norm(q_new - q_target) < 1e-3: return "Reached", new_idx
            return "Advanced", new_idx
        return "Trapped", None

    def connect(self, tree, q_target):
        status, last_idx = "Advanced", None
        while status == "Advanced":
            status, last_idx = self.extend(tree, q_target)
        return status, last_idx

def update_visuals(viewer, treeA, treeB, current_ee_pos=None, final_path_ee=None):
    viewer.user_scn.ngeom = 0
    id_mat = np.eye(3).flatten()

    # Draw trees
    for t_idx, tree in enumerate([treeA, treeB]):
        color_node = [0, 1, 0, 0.4] if t_idx == 0 else [0.2, 0.4, 1, 0.4]
        for i, pos in enumerate(tree.ee_positions):
            if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom: break
            mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], 
                                mujoco.mjtGeom.mjGEOM_SPHERE, [0.005, 0, 0], pos, id_mat, color_node)
            viewer.user_scn.ngeom += 1
            parent = tree.parents[i]
            if parent is not None:
                mujoco.mjv_connector(viewer.user_scn.geoms[viewer.user_scn.ngeom], 
                                     mujoco.mjtGeom.mjGEOM_LINE, 0.001, pos, tree.ee_positions[parent])
                viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba = [0.5, 0.5, 0.5, 0.1]
                viewer.user_scn.ngeom += 1

    # Draw the 15cm Safety Sphere at the current planning/execution point
    if current_ee_pos is not None:
        mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], 
                            mujoco.mjtGeom.mjGEOM_SPHERE, [EE_SPHERE_RADIUS, 0, 0], 
                            current_ee_pos, id_mat, [1, 1, 0, 0.2]) # Yellow transparent
        viewer.user_scn.ngeom += 1

    # Draw Final Path
    if final_path_ee is not None:
        for i in range(len(final_path_ee) - 1):
            mujoco.mjv_connector(viewer.user_scn.geoms[viewer.user_scn.ngeom], 
                                 mujoco.mjtGeom.mjGEOM_LINE, 0.004, final_path_ee[i], final_path_ee[i+1])
            viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba = [1, 0, 0, 1]
            viewer.user_scn.ngeom += 1

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", nargs=3, type=float, default=[0, 0.15, 0.2])
    parser.add_argument("--end", nargs=3, type=float, default=[0, 0.5, 0.1])
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    start_q = solve_ik(model, data, np.array(args.start))
    goal_q  = solve_ik(model, data, np.array(args.end))
    if start_q is None or goal_q is None: return

    planner = RRTConnect(model, start_q, goal_q)
    trees = [planner.treeA, planner.treeB]
    swapped = False
    
    # NEW: Track the best path and its cost
    best_path = None
    best_path_ee = None
    min_cost = float('inf')

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for step in range(MAX_SAMPLES):
            if not viewer.is_running(): break
            q_rand = np.random.uniform(-3.14, 3.14, 6)
            
            # Standard RRT-Connect Expand
            status, idx_active = planner.extend(trees[0], q_rand)
            
            if status != "Trapped":
                # Try to connect the trees
                status_conn, idx_other = planner.connect(trees[1], trees[0].nodes[idx_active])
                
                if status_conn == "Reached":
                    # Reconstruction logic
                    p_act, p_oth = [], []
                    curr = idx_active
                    while curr is not None:
                        p_act.append(trees[0].nodes[curr])
                        curr = trees[0].parents[curr]
                    curr = idx_other
                    while curr is not None:
                        p_oth.append(trees[1].nodes[curr])
                        curr = trees[1].parents[curr]
                    
                    # Form the full candidate path
                    candidate_path = (p_act[::-1] + p_oth) if not swapped else (p_oth[::-1] + p_act)
                    
                    # Calculate cost (Joint-space distance)
                    current_cost = sum(np.linalg.norm(candidate_path[i+1] - candidate_path[i]) 
                                       for i in range(len(candidate_path)-1))
                    
                    # If this path is better, update the "Final Path"
                    if current_cost < min_cost:
                        min_cost = current_cost
                        best_path = candidate_path
                        
                        # Generate EE coordinates for the Red Line visualization
                        best_path_ee = []
                        temp_data = mujoco.MjData(model)
                        for q in best_path:
                            temp_data.qpos[:6] = q
                            mujoco.mj_forward(model, temp_data)
                            best_path_ee.append(temp_data.body(planner.ee_id).xpos.copy())
                        
                        print(f"Step {step}: New optimal path found! Cost: {min_cost:.4f}")
            
            # Swap trees for bi-directional growth
            trees[0], trees[1] = trees[1], trees[0]
            swapped = not swapped
            
            # Visualization: Only update visuals every 10 steps to keep the UI smooth
            if step % 10 == 0:
                # current_ee_pos points to the tip of the growing tree
                tip_pos = trees[0].ee_positions[-1]
                update_visuals(viewer, planner.treeA, planner.treeB, 
                               current_ee_pos=tip_pos, final_path_ee=best_path_ee)
                viewer.sync()

        # Final playback loop using the BEST found path
        print(f"Refinement complete. Replaying best path with cost: {min_cost:.4f}")
        while viewer.is_running() and best_path:
            for q in best_path:
                if not viewer.is_running(): break
                data.qpos[:6] = q
                mujoco.mj_forward(model, data)
                ee_pos = data.body(planner.ee_id).xpos
                update_visuals(viewer, planner.treeA, planner.treeB, 
                               current_ee_pos=ee_pos, final_path_ee=best_path_ee)
                viewer.sync()
                time.sleep(0.04)

if __name__ == "__main__":
    main()