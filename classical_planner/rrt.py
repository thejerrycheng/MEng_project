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
        self.goal_ee_pos = self.get_ee_pos(goal_q)

        self.samples_count = 0
        self.goal_found = False
        self.best_goal_node = None
        self.best_cost = float("inf")
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

            if self.check_data.ncon == 0:
                return True

            # Filter/ignore a couple of allowed contacts if you want:
            for i in range(self.check_data.ncon):
                con = self.check_data.contact[i]
                g1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
                g2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)
                if g1 is None or g2 is None:
                    continue
                if "floor" in (g1, g2) and any(x in (g1, g2) for x in ("base", "link1")):
                    continue
                return False
            return True

        if q2 is None:
            return check(q1)

        # Edge checking
        for t in np.linspace(0, 1, 6):
            if not check(q1 + t * (q2 - q1)):
                return False
        return True

    def get_path_indices(self):
        if self.best_goal_node is None:
            return []
        path = []
        curr = self.best_goal_node
        while curr is not None:
            path.append(curr)
            curr = self.parents[curr]
        return path  # goal->start order

    def step(self):
        if self.samples_count >= MAX_SAMPLES:
            return False
        self.samples_count += 1

        # --- Sampling strategy ---
        p = np.random.random()

        if self.goal_found and p < 0.50:
            path_indices = self.get_path_indices()
            if len(path_indices) > 0:
                target_node_idx = np.random.choice(path_indices)
                q_base = self.nodes[target_node_idx]
                q_rand = q_base + np.random.normal(0, 0.2, 6)
                q_rand = np.clip(q_rand, -3.14, 3.14)
            else:
                q_rand = np.array([np.random.uniform(-3.14, 3.14) for _ in range(6)])

        elif p < 0.15:
            q_rand = self.goal_q
        else:
            q_rand = np.array([np.random.uniform(-3.14, 3.14) for _ in range(6)])

        node_arr = np.array(self.nodes)
        nearest_idx = int(np.argmin(np.linalg.norm(node_arr - q_rand, axis=1)))
        q_near = self.nodes[nearest_idx]

        diff = q_rand - q_near
        d = float(np.linalg.norm(diff))
        if d < 1e-9:
            return True

        q_new = q_near + (diff / d) * min(d, STEP_SIZE)

        is_free = self.is_collision_free(q_near, q_new)
        ee_new = self.get_ee_pos(q_new)
        self.all_samples.append((ee_new, is_free))

        if not is_free:
            return True

        # Choose parent among neighbors
        dists_to_new = np.linalg.norm(node_arr - q_new, axis=1)
        neighbors = np.where(dists_to_new < SEARCH_RADIUS)[0]

        best_p = nearest_idx
        min_c = self.costs[nearest_idx] + float(np.linalg.norm(q_new - q_near))

        for n_idx in neighbors:
            c_via_n = self.costs[int(n_idx)] + float(np.linalg.norm(q_new - self.nodes[int(n_idx)]))
            if c_via_n < min_c and self.is_collision_free(self.nodes[int(n_idx)], q_new):
                best_p, min_c = int(n_idx), c_via_n

        new_idx = len(self.nodes)
        self.nodes.append(q_new.copy())
        self.node_ee_positions.append(ee_new.copy())
        self.parents[new_idx] = best_p
        self.costs[new_idx] = float(min_c)

        # Rewire
        for n_idx in neighbors:
            n_idx = int(n_idx)
            c_via_new = self.costs[new_idx] + float(np.linalg.norm(self.nodes[n_idx] - q_new))
            if c_via_new < self.costs[n_idx] and self.is_collision_free(q_new, self.nodes[n_idx]):
                self.parents[n_idx] = new_idx
                self.costs[n_idx] = float(c_via_new)

        # Goal check
        if float(np.linalg.norm(q_new - self.goal_q)) < STEP_SIZE:
            self.goal_found = True
            if min_c < self.best_cost:
                delta = self.best_cost - min_c
                self.best_cost = float(min_c)
                self.best_goal_node = new_idx
                self.discovery_log.append(f"Better Path! Cost: {min_c:.4f} (-{delta:.4f})")

        return True

    def extract_best_path_q(self):
        if self.best_goal_node is None:
            return None
        path_q = []
        curr = self.best_goal_node
        while curr is not None:
            path_q.append(self.nodes[curr])
            curr = self.parents[curr]
        path_q = np.array(path_q[::-1])  # start->goal
        return path_q


# ---------- Visualization helpers ----------
def _set_nice_camera(cam, model):
    # Simple “good enough” camera defaults
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    # Look at model center
    cam.lookat[:] = model.stat.center
    cam.distance = 2.5 * model.stat.extent
    cam.azimuth = 135
    cam.elevation = -20


def render_final_path_offscreen(model, path_q, out_png, out_mp4,
                                width=1280, height=720, fps=30,
                                hold_seconds=2.0, animate_robot=True):
    """
    Headless/offscreen render:
      - A still PNG showing the final path (as red segments)
      - An MP4 that holds, then (optionally) animates robot along path
    """
    try:
        import imageio.v2 as imageio
    except Exception as e:
        raise RuntimeError(
            "imageio is required for MP4/PNG export. Install with: pip install imageio"
        ) from e

    data = mujoco.MjData(model)

    # Offscreen renderer
        # --------------------------------------------------
    # Safe renderer size (prevents MuJoCo crash)
    # --------------------------------------------------
    safe_w, safe_h = _safe_render_size(model, width, height)

    renderer = mujoco.Renderer(
        model,
        width=safe_w,
        height=safe_h
    )


    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    _set_nice_camera(cam, model)

    id_mat = np.eye(3).flatten().astype(np.float64)
    zero_v = np.zeros(3, dtype=np.float64)

    # Precompute EE points for drawing connectors
    check_data = mujoco.MjData(model)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")
    ee_pts = []
    for q in path_q:
        check_data.qpos[:6] = q
        mujoco.mj_forward(model, check_data)
        ee_pts.append(check_data.body(ee_id).xpos.copy())
    ee_pts = np.array(ee_pts)

    def overlay_path_geoms():
        scn = renderer.scene
        # Add our custom geoms after the scene is updated.
        for i in range(len(ee_pts) - 1):
            if scn.ngeom >= scn.maxgeom:
                break
            idx = scn.ngeom
            mujoco.mjv_initGeom(
                scn.geoms[idx],
                mujoco.mjtGeom.mjGEOM_LINE,
                zero_v, zero_v, id_mat,
                np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)  # red
            )
            mujoco.mjv_connector(
                scn.geoms[idx],
                mujoco.mjtGeom.mjGEOM_LINE,
                0.008,  # thickness
                ee_pts[i],
                ee_pts[i + 1]
            )
            scn.ngeom += 1

    # --- Make a “hero” still image ---
    data.qpos[:6] = path_q[0]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam, scene_option=opt)
    overlay_path_geoms()
    img = renderer.render()
    imageio.imwrite(out_png, img)

    # --- Make MP4: hold + animate (optional) ---
    writer = imageio.get_writer(out_mp4, fps=fps)

    hold_frames = int(max(1, hold_seconds * fps))

    # hold at start pose with path overlay
    for _ in range(hold_frames):
        data.qpos[:6] = path_q[0]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam, scene_option=opt)
        overlay_path_geoms()
        writer.append_data(renderer.render())

    if animate_robot:
        # simple per-waypoint animation (repeat each waypoint a few frames)
        frames_per_wp = max(1, int(0.05 * fps))  # ~50ms per waypoint step
        for q in path_q:
            for _ in range(frames_per_wp):
                data.qpos[:6] = q
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=cam, scene_option=opt)
                overlay_path_geoms()
                writer.append_data(renderer.render())

        # hold at end
        for _ in range(hold_frames):
            data.qpos[:6] = path_q[-1]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam, scene_option=opt)
            overlay_path_geoms()
            writer.append_data(renderer.render())

    writer.close()

def _safe_render_size(model, width, height):
    """
    Clamp requested render size to MuJoCo framebuffer limits
    to avoid Renderer crashes.
    """
    fb_w = int(model.vis.global_.offwidth)
    fb_h = int(model.vis.global_.offheight)

    # MuJoCo defaults to 640x480 if not specified in XML
    if fb_w <= 0:
        fb_w = 640
    if fb_h <= 0:
        fb_h = 480

    safe_w = min(width, fb_w)
    safe_h = min(height, fb_h)

    if safe_w != width or safe_h != height:
        print(
            f"[WARN] Requested render size {width}x{height} exceeds "
            f"framebuffer {fb_w}x{fb_h}. "
            f"Clamping to {safe_w}x{safe_h}."
        )

    return safe_w, safe_h


def visualize_final_path_in_viewer(model, path_q, speed=1.0):
    """
    Interactive viewer visualization: animates robot along the final path and draws the path as red lines.
    """
    data = mujoco.MjData(model)

    id_mat = np.eye(3).flatten().astype(np.float64)
    zero_v = np.zeros(3, dtype=np.float64)

    # Precompute EE points
    check_data = mujoco.MjData(model)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")
    ee_pts = []
    for q in path_q:
        check_data.qpos[:6] = q
        mujoco.mj_forward(model, check_data)
        ee_pts.append(check_data.body(ee_id).xpos.copy())
    ee_pts = np.array(ee_pts)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Draw path once per frame (simple + robust)
        for q in path_q:
            if not viewer.is_running():
                break
            data.qpos[:6] = q
            mujoco.mj_forward(model, data)

            # reset custom geoms
            viewer.user_scn.ngeom = 0

            # draw final path (only)
            for i in range(len(ee_pts) - 1):
                if viewer.user_scn.ngeom >= MAX_VIZ_GEOMS:
                    break
                idx = viewer.user_scn.ngeom
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[idx],
                    mujoco.mjtGeom.mjGEOM_LINE,
                    zero_v, zero_v, id_mat,
                    np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
                )
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[idx],
                    mujoco.mjtGeom.mjGEOM_LINE,
                    0.010,
                    ee_pts[i],
                    ee_pts[i + 1]
                )
                viewer.user_scn.ngeom += 1

            viewer.sync()

            # crude pacing
            mujoco.mj_sleep(0.01 / max(1e-6, speed))


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true",
                        help="Run planning without live viewer; export final path PNG/MP4 instead.")
    parser.add_argument("--no-final-viewer", action="store_true",
                        help="Do not open the interactive viewer for the final path (useful on servers).")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    rrt = GlobalRRTStar(model, START_Q, GOAL_Q)

    # ---------------------------
    # 1) PLANNING
    # ---------------------------
    if args.headless:
        # No viewer: just run steps
        while rrt.samples_count < MAX_SAMPLES:
            ok = rrt.step()
            if not ok:
                break

            # light terminal log every so often
            if rrt.samples_count % 1000 == 0:
                status = "GOAL" if rrt.goal_found else "EXPLORE"
                best_cost_str = f"{rrt.best_cost:.4f}" if rrt.goal_found else "N/A"
                print(f"[headless] samples={rrt.samples_count:6d} nodes={len(rrt.nodes):6d} "
                      f"status={status:7s} best_cost={best_cost_str}")

    else:
        # Live viewer planning (your original behavior)
        id_mat = np.eye(3).flatten().astype(np.float64)
        zero_v = np.zeros(3, dtype=np.float64)

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while rrt.samples_count < MAX_SAMPLES and viewer.is_running():
                for _ in range(40):
                    rrt.step()

                # Render (samples + tree)
                viewer.user_scn.ngeom = 0
                opt_indices = rrt.get_path_indices()
                opt_set = set(opt_indices)

                # samples
                for pos, free in rrt.all_samples[-200:]:
                    if viewer.user_scn.ngeom >= 500:
                        break
                    color = [0, 1, 1, 0.2] if free else [1, 0, 0, 0.1]
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([0.005, 0, 0], dtype=np.float64),
                        pos,
                        id_mat,
                        np.array(color, dtype=np.float32),
                    )
                    viewer.user_scn.ngeom += 1

                # tree
                for child, parent in rrt.parents.items():
                    if parent is None or viewer.user_scn.ngeom >= MAX_VIZ_GEOMS:
                        continue
                    is_opt = (
                        child in opt_set and parent in opt_set
                        and abs(opt_indices.index(child) - opt_indices.index(parent)) == 1
                    )
                    color, rad = ([1, 0, 0, 1], 0.007) if is_opt else ([0, 1, 0, 0.2], 0.002)
                    idx = viewer.user_scn.ngeom
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[idx],
                        mujoco.mjtGeom.mjGEOM_LINE,
                        zero_v, zero_v, id_mat,
                        np.array(color, dtype=np.float32)
                    )
                    mujoco.mjv_connector(
                        viewer.user_scn.geoms[idx],
                        mujoco.mjtGeom.mjGEOM_LINE,
                        rad,
                        rrt.node_ee_positions[parent],
                        rrt.node_ee_positions[child]
                    )
                    viewer.user_scn.ngeom += 1

                viewer.sync()

                # terminal log
                sys.stdout.write("\033[H\033[J")
                status_str = "[\033[92m GOAL REACHED \033[0m]" if rrt.goal_found else "[\033[94m EXPLORING \033[0m]"
                best_cost_str = f"{rrt.best_cost:.4f}" if rrt.goal_found else "N/A"
                lines = [
                    "================ RRT* LIVE MONITORING ================",
                    f" Samples: {rrt.samples_count}/{MAX_SAMPLES} | Nodes: {len(rrt.nodes)}",
                    f" Status:  {status_str}",
                    f" Best Cost: {best_cost_str}",
                    "-" * 54
                ]
                if rrt.discovery_log:
                    lines.append(" Recent Optimization Discoveries:")
                    yellow_plus = "\033[93m[+]\033[0m"
                    for log in rrt.discovery_log[-5:]:
                        lines.append(f"  {yellow_plus} {log}")
                else:
                    lines.append(" Waiting for initial path discovery...")
                lines.append("======================================================")
                sys.stdout.write("\n".join(lines) + "\n")
                sys.stdout.flush()

    # ---------------------------
    # 2) SAVE PATH
    # ---------------------------
    path_q = rrt.extract_best_path_q()
    if path_q is None:
        print("\n[!] FAILURE: Sampling budget reached with no valid path.")
        sys.exit(1)

    np.save(os.path.join(SAVE_DIR, "waypoints.npy"), path_q)
    print(f"\n[!] SUCCESS: {len(path_q)} waypoints saved to {os.path.join(SAVE_DIR, 'waypoints.npy')}")

    # ---------------------------
    # 3) FINAL PATH VISUALIZATION (works in headless!)
    # # ---------------------------
    # out_png = os.path.join(SAVE_DIR, "final_path.png")
    # out_mp4 = os.path.join(SAVE_DIR, "final_path.mp4")
    # render_final_path_offscreen(
    #     model,
    #     path_q,
    #     out_png=out_png,
    #     out_mp4=out_mp4,
    #     width=args.width,
    #     height=args.height,
    #     fps=args.fps,
    #     hold_seconds=2.0,
    #     animate_robot=True
    # )
    # print(f"[!] Exported: {out_png}")
    # print(f"[!] Exported: {out_mp4}")

    # Optional: open viewer to show only the FINAL path (no tree)
    if (not args.headless) or (not args.no_final_viewer):
        try:
            visualize_final_path_in_viewer(model, path_q, speed=1.0)
        except Exception as e:
            print(f"[!] Viewer skipped (likely no display): {e}")

    print("[!] Done.")


if __name__ == "__main__":
    main()
