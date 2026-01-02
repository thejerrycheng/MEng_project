import mujoco
import mujoco.viewer
import numpy as np
import time
import sys
import os

# --- Configuration ---
# Ensure this points to the XML with the <site name="target_ball"...> added
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURR_DIR, "assets", "iris_with_axis.xml")
START_Q = np.array([0, -45, -90, 0, 0, 0], dtype=np.float64) 

# Trajectory Settings
LINEAR_VELOCITY = 0.15   # m/s constant speed between points
ARRIVAL_TOLERANCE = 0.02 # 2cm tolerance to consider reached

# --- Waypoints: [X, Y, Z, Roll, Pitch, Yaw] ---
WAYPOINTS = [
    [0.3, -0.2, 0.4, 0, 0, 0],
    [0.4,  0.2, 0.5, 0, 0, 0],
    [0.5,  0.0, 0.3, 0, 1.57, 0],
    [0.3,  0.3, 0.6, 1.57, 0, 1.57],
    [0.2,  0.0, 0.4, 0, 0, 0],
]

class TrajectoryFollower:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")
        
        # Look for the visual site in the XML
        self.target_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_ball")
        if self.target_site_id == -1:
            print("WARNING: <site name='target_ball'> not found in XML. Visualization disabled.")

        # Initialize Robot Pose
        self.data.qpos[:6] = np.deg2rad(START_Q)
        mujoco.mj_forward(model, data)
        
        # Initialize IK targets to current EE position
        self.current_ik_target = self.data.body(self.ee_id).xpos.copy()
        
        self.waypoint_idx = 0
        self.damping = 1e-3

        # --- VISUALIZATION INIT ---
        # Set the green ball to the FIRST waypoint immediately
        if self.target_site_id != -1 and len(WAYPOINTS) > 0:
            self.model.site_pos[self.target_site_id] = WAYPOINTS[0][:3]

    def solve_ik_step(self, target_p, target_e):
        """Standard DLS IK Solve for one step"""
        # Convert Euler target to Matrix
        target_quat = np.zeros(4)
        mujoco.mju_euler2Quat(target_quat, target_e, 'xyz')
        target_mat = np.zeros(9)
        mujoco.mju_quat2Mat(target_mat, target_quat)
        target_mat = target_mat.reshape(3, 3)

        # Get Current State
        curr_pos = self.data.body(self.ee_id).xpos
        curr_mat = self.data.body(self.ee_id).xmat.reshape(3, 3)
        
        # Calculate Errors
        pos_err = target_p - curr_pos
        rot_err_mat = target_mat @ curr_mat.T
        rot_err_quat = np.zeros(4)
        mujoco.mju_mat2Quat(rot_err_quat, rot_err_mat.flatten())
        rot_err_vec = rot_err_quat[1:] * np.sign(rot_err_quat[0])

        # Jacobian & DLS
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_id)
        J = np.vstack([jacp, jacr])[:, :6]
        
        error = np.concatenate([pos_err, rot_err_vec])
        dq = J.T @ np.linalg.solve(J @ J.T + self.damping * np.eye(6), error)
        
        # Integration with speed limit safety
        dq = np.clip(dq, -2.0, 2.0) 
        self.data.qpos[:6] += dq * 0.2
        self.data.qpos[:6] = np.clip(self.data.qpos[:6], 
                                     self.model.jnt_range[:6, 0], 
                                     self.model.jnt_range[:6, 1])

    def update(self, dt):
        """
        Returns True if all waypoints are completed.
        Handles interpolation and updating visual marker.
        """
        if self.waypoint_idx >= len(WAYPOINTS):
            return True # Sequence Finished

        # Get current destination coordinates
        dest_coords = WAYPOINTS[self.waypoint_idx]
        dest_pos = np.array(dest_coords[:3])
        dest_euler = np.array(dest_coords[3:])

        # --- 1. Workspace Check ---
        # Simple check: if target is too far from origin, skip it.
        if np.linalg.norm(dest_pos) > 0.85: 
            print(f"--> Waypoint {self.waypoint_idx} out of reach. Skipping.")
            self.next_waypoint()
            return False

        # --- 2. Position Interpolation (Constant Speed) ---
        # Vector from current interpolated IK target to ultimate destination
        pos_diff = dest_pos - self.current_ik_target
        dist_to_go = np.linalg.norm(pos_diff)
        
        if dist_to_go > ARRIVAL_TOLERANCE:
            # Move the IK target closer by exactly (Velocity * dt)
            move_direction = pos_diff / dist_to_go
            move_step = move_direction * LINEAR_VELOCITY * dt
            
            # Ensure we don't overshoot if close
            if np.linalg.norm(move_step) > dist_to_go:
                 self.current_ik_target = dest_pos
            else:
                 self.current_ik_target += move_step
        else:
            # --- Reached Destination ---
            print(f"--> Reached Waypoint {self.waypoint_idx}")
            self.next_waypoint()

        # --- 3. Run IK on the interpolated point ---
        # The IK follows the smoothly moving 'current_ik_target', 
        # while the visual ball stays at the ultimate destination.
        self.solve_ik_step(self.current_ik_target, dest_euler)
        return False

    def next_waypoint(self):
        """Helper to increment index and update visualization"""
        self.waypoint_idx += 1
        # Update visual marker to the NEW target if available
        if self.waypoint_idx < len(WAYPOINTS) and self.target_site_id != -1:
            self.model.site_pos[self.target_site_id] = WAYPOINTS[self.waypoint_idx][:3]

def main():
    if not os.path.exists(XML_PATH):
        print(f"XML not found: {XML_PATH}")
        return

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    follower = TrajectoryFollower(model, data)

    sys.stdout.write("\033[2J\033[H") # Clear terminal

    # Use passive viewer so we control the stepping loop
    with mujoco.viewer.launch_passive(model, data) as viewer:
        time.sleep(0.5) # Give viewer a moment to initialize

        while viewer.is_running():
            step_start = time.time()
            
            # Update trajectory logic
            finished = follower.update(model.opt.timestep)
            if finished:
                print("\n=== TRAJECTORY COMPLETE ===")
                # Keep running viewer but stop updating robot
                while viewer.is_running():
                     viewer.sync()
                     time.sleep(0.1)
                break

            # Step Physics
            mujoco.mj_step(model, data)
            viewer.sync() 

            # Simple Dashboard
            if int(data.time * 100) % 20 == 0:
                sys.stdout.write("\033[H")
                curr_ee = data.body(follower.ee_id).xpos
                target_wp = WAYPOINTS[min(follower.waypoint_idx, len(WAYPOINTS)-1)][:3]
                print("========== CONSTANT VELOCITY FOLLOWER ==========")
                print(f" Sim Time: {data.time:6.2f}s | Current WP Index: {follower.waypoint_idx}/{len(WAYPOINTS)}")
                print(f" Target WP Pos: [{target_wp[0]:.2f}, {target_wp[1]:.2f}, {target_wp[2]:.2f}]")
                print(f" Current EE Pos:   [{curr_ee[0]:.2f}, {curr_ee[1]:.2f}, {curr_ee[2]:.2f}]")
                print(f" Distance to WP: {np.linalg.norm(curr_ee - target_wp):.4f} m")
                print("================================================")
                sys.stdout.flush()

            # Real-time sync
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == "__main__":
    main()