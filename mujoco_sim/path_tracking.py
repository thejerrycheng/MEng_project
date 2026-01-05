import mujoco
import mujoco.viewer
import numpy as np
import time
import sys
import os

# --- Configuration ---
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURR_DIR, "assets", "iris.xml")

# Circle Parameters
CIRCLE_CENTER = np.array([0.4, 0.0, 0.4])
CIRCLE_RADIUS = 0.15
NUM_WAYPOINTS = 100  # Evenly spaced points

# Controller Parameters
KP = 600.0
KD = 50.0
LINEAR_VELOCITY = 0.1  # m/s
ARRIVAL_TOLERANCE = 0.005

def generate_circular_path():
    """Generates evenly spaced waypoints on a circle facing the center."""
    waypoints = []
    angles = np.linspace(0, 2 * np.pi, NUM_WAYPOINTS)
    for theta in angles:
        # Position on XY plane at height Z
        x = CIRCLE_CENTER[0] + CIRCLE_RADIUS * np.cos(theta)
        y = CIRCLE_CENTER[1] + CIRCLE_RADIUS * np.sin(theta)
        z = CIRCLE_CENTER[2]
        
        # Calculate Orientation: Z-axis pointing to center
        pos = np.array([x, y, z])
        z_axis = CIRCLE_CENTER - pos
        z_axis /= np.linalg.norm(z_axis)
        
        # Construct orthonormal basis
        up = np.array([0, 0, 1])
        x_axis = np.cross(up, z_axis)
        x_axis /= (np.linalg.norm(x_axis) + 1e-6)
        y_axis = np.cross(z_axis, x_axis)
        
        rot_mat = np.stack([x_axis, y_axis, z_axis], axis=1)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rot_mat.flatten())
        
        # Store as [x, y, z, q0, q1, q2, q3]
        waypoints.append((pos, quat))
    return waypoints

class CircleIDFollower:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")
        self.waypoints = generate_circular_path()
        self.wp_idx = 0
        
        # Snap reference to start
        start_pos, start_quat = self.waypoints[0]
        self.q_ref = self.solve_initial_q(start_pos, start_quat)
        self.data.qpos[:6] = self.q_ref
        mujoco.mj_forward(model, data)
        
    def solve_initial_q(self, p, q):
        """Warm-up IK to ensure we start on the circle."""
        temp_data = mujoco.MjData(self.model)
        temp_data.qpos[:6] = np.deg2rad([0, -45, -90, 0, 0, 0])
        for _ in range(100):
            mujoco.mj_forward(self.model, temp_data)
            jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, temp_data, jacp, jacr, self.ee_id)
            J = np.vstack([jacp[:, :6], jacr[:, :6]])
            
            dx = p - temp_data.body(self.ee_id).xpos
            curr_q = temp_data.body(self.ee_id).xquat
            q_inv, q_diff, dr = np.zeros(4), np.zeros(4), np.zeros(3)
            mujoco.mju_negQuat(q_inv, curr_q)
            mujoco.mju_mulQuat(q_diff, q, q_inv)
            mujoco.mju_quat2Vel(dr, q_diff, 1.0)
            
            dq = np.linalg.solve(J.T @ J + 1e-4 * np.eye(6), J.T @ np.concatenate([dx, dr]))
            temp_data.qpos[:6] += dq
        return temp_data.qpos[:6].copy()

    def update(self):
        if self.wp_idx >= len(self.waypoints): return True
        
        target_pos, target_quat = self.waypoints[self.wp_idx]
        
        # 1. Predictive IK for reference trajectory
        ik_data = mujoco.MjData(self.model)
        ik_data.qpos[:6] = self.q_ref
        mujoco.mj_forward(self.model, ik_data)
        
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, ik_data, jacp, jacr, self.ee_id)
        J = np.vstack([jacp[:, :6], jacr[:, :6]])
        
        # Cartesian error to waypoint
        dx = target_pos - ik_data.body(self.ee_id).xpos
        dist = np.linalg.norm(dx)
        
        # Orientation error
        curr_q = ik_data.body(self.ee_id).xquat
        q_inv, q_diff, dr = np.zeros(4), np.zeros(4), np.zeros(3)
        mujoco.mju_negQuat(q_inv, curr_q)
        mujoco.mju_mulQuat(q_diff, target_quat, q_inv)
        mujoco.mju_quat2Vel(dr, q_diff, 1.0)
        
        # Move q_ref toward waypoint
        v_limit = min(dist, LINEAR_VELOCITY * self.model.opt.timestep) / self.model.opt.timestep
        dq_ref = J.T @ np.linalg.solve(J @ J.T + 1e-5 * np.eye(6), np.concatenate([dx, dr]))
        self.q_ref += dq_ref * self.model.opt.timestep

        # 2. Inverse Dynamics Control Law
        self.data.qacc[:6] = KP * (self.q_ref - self.data.qpos[:6]) + KD * (dq_ref - self.data.qvel[:6])
        mujoco.mj_inverse(self.model, self.data)
        self.data.ctrl[:6] = self.data.qfrc_inverse[:6].copy()

        if dist < ARRIVAL_TOLERANCE: self.wp_idx += 1
        return False

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    follower = CircleIDFollower(model, data)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            if follower.update(): break
            
            mujoco.mj_step(model, data)
            
            # Path Visualization
            if int(data.time * 100) % 5 == 0:
                viewer.user_scn.ngeom = 0
                for wp_pos, _ in follower.waypoints:
                    mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                                        mujoco.mjtGeom.mjGEOM_SPHERE, [0.003, 0, 0], 
                                        wp_pos, np.eye(3).flatten(), [0, 1, 0, 0.5])
                    viewer.user_scn.ngeom += 1
                viewer.sync()

            # Dashboard
            if int(data.time * 100) % 10 == 0:
                err = np.linalg.norm(data.body(follower.ee_id).xpos - follower.waypoints[min(follower.wp_idx, NUM_WAYPOINTS-1)][0])
                sys.stdout.write(f"\rTracking Circle... WP: {follower.wp_idx}/{NUM_WAYPOINTS} | Error: {err*1000:.2f} mm")
                sys.stdout.flush()

            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep: time.sleep(model.opt.timestep - elapsed)

if __name__ == "__main__":
    main()