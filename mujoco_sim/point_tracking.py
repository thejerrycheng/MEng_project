import mujoco
import mujoco.viewer
import numpy as np
import time
import sys
import os
import xml.etree.ElementTree as ET

# --- Configuration ---
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURR_DIR, "assets", "iris.xml")
START_Q = np.array([0, -45, -90, 0, 0, 0], dtype=np.float64) 

LINEAR_VELOCITY = 0.08   # m/s (Slightly slower for better precision)
ANGULAR_VELOCITY = 0.3   # rad/s (Reduced to ensure IK can keep up)
ARRIVAL_TOLERANCE = 0.008 # 8mm tolerance

# Waypoints: [X, Y, Z, Roll, Pitch, Yaw]
WAYPOINTS = [
    [0.4, -0.2, 0.4, 0.0, 0.0, 0.0],
    [0.5,  0.1, 0.5, 1.57, 0.0, 0.0],
    [0.4,  0.3, 0.3, 0.0, 1.57, 0.0],
    [0.3,  0.0, 0.6, 0.0, 0.0, 1.57],
]

# --- XML Helper Functions ---
def get_basic_worldbody(original_xml_path):
    tree = ET.parse(original_xml_path)
    root = tree.getroot()
    worldbody = root.find('worldbody')
    if worldbody is None:
        worldbody = ET.SubElement(root, 'worldbody')
    return tree, root, worldbody

def save_and_load_temp(tree, original_xml_path, suffix="precision"):
    original_dir = os.path.dirname(original_xml_path)
    temp_path = os.path.join(original_dir, f'scene_{suffix}.xml')
    tree.write(temp_path, encoding='unicode')
    model = mujoco.MjModel.from_xml_path(temp_path)
    if os.path.exists(temp_path):
        os.unlink(temp_path)
    return model

def create_model_with_target_pose(original_xml_path, pos, euler, size=0.1):
    tree, root, worldbody = get_basic_worldbody(original_xml_path)
    frame_body = ET.SubElement(worldbody, 'body', {
        'name': 'target_frame', 
        'pos': f"{pos[0]} {pos[1]} {pos[2]}",
        'euler': f"{euler[0]} {euler[1]} {euler[2]}"
    })
    ET.SubElement(frame_body, 'geom', {'type': 'sphere', 'size': '0.02', 'rgba': '0 1 0 0.6', 'contype': '0', 'conaffinity': '0'})
    ET.SubElement(frame_body, 'geom', {'type': 'cylinder', 'fromto': f"0 0 0 {size} 0 0", 'size': '0.005', 'rgba': '1 0 0 0.8', 'contype': '0', 'conaffinity': '0'})
    ET.SubElement(frame_body, 'geom', {'type': 'cylinder', 'fromto': f"0 0 0 0 {size} 0", 'size': '0.005', 'rgba': '0 1 0 0.8', 'contype': '0', 'conaffinity': '0'})
    ET.SubElement(frame_body, 'geom', {'type': 'cylinder', 'fromto': f"0 0 0 0 0 {size}", 'size': '0.005', 'rgba': '0 0 1 0.8', 'contype': '0', 'conaffinity': '0'})
    return save_and_load_temp(tree, original_xml_path)

class PrecisionPoseFollower:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")
        self.target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_frame")
        
        self.data.qpos[:6] = np.deg2rad(START_Q)
        mujoco.mj_forward(model, data)
        
        # Initialize internal targets to exact EE state
        self.current_target_pos = self.data.body(self.ee_id).xpos.copy()
        self.current_target_quat = np.zeros(4)
        mujoco.mju_mat2Quat(self.current_target_quat, self.data.body(self.ee_id).xmat)
        
        self.waypoint_idx = 0
        self.damping = 1e-5 # Lower damping for higher precision

    def solve_ik(self, target_p, target_q):
        # Current State
        curr_pos = self.data.body(self.ee_id).xpos
        curr_mat = self.data.body(self.ee_id).xmat.reshape(3, 3)
        
        # Pos Error
        pos_err = target_p - curr_pos
        
        # Ori Error (High Precision Orientation)
        target_mat = np.zeros(9)
        mujoco.mju_quat2Mat(target_mat, target_q)
        rot_err_mat = target_mat.reshape(3, 3) @ curr_mat.T
        rot_err_quat = np.zeros(4)
        mujoco.mju_mat2Quat(rot_err_quat, rot_err_mat.flatten())
        # The axis-angle representation of the rotation difference
        rot_err_vec = rot_err_quat[1:] * np.sign(rot_err_quat[0])

        # Jacobian
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_id)
        J = np.vstack([jacp, jacr])[:, :6]
        
        # Solve with full integration gain for precise tracking
        error = np.concatenate([pos_err, rot_err_vec])
        dq = J.T @ np.linalg.solve(J @ J.T + self.damping * np.eye(6), error)
        
        # Apply full step (1.0) since we are tracking a moving target smoothly
        self.data.qpos[:6] += dq
        self.data.qpos[:6] = np.clip(self.data.qpos[:6], self.model.jnt_range[:6, 0], self.model.jnt_range[:6, 1])

    def update(self, dt):
        if self.waypoint_idx >= len(WAYPOINTS):
            return True

        wp = WAYPOINTS[self.waypoint_idx]
        dest_pos = np.array(wp[:3])
        dest_quat = np.zeros(4)
        mujoco.mju_euler2Quat(dest_quat, wp[3:], 'xyz')

        # 1. Linear Position Step
        pos_diff = dest_pos - self.current_target_pos
        dist = np.linalg.norm(pos_diff)
        if dist > ARRIVAL_TOLERANCE:
            self.current_target_pos += (pos_diff / dist) * LINEAR_VELOCITY * dt
        
        # 2. Angular Velocity Step (mju_quatIntegrate)
        q_inv = np.zeros(4)
        mujoco.mju_negQuat(q_inv, self.current_target_quat)
        q_diff = np.zeros(4)
        mujoco.mju_mulQuat(q_diff, dest_quat, q_inv)
        
        vel_vec = np.zeros(3)
        mujoco.mju_quat2Vel(vel_vec, q_diff, 1.0)
        
        vel_norm = np.linalg.norm(vel_vec)
        if vel_norm > 1e-4:
            # Scale the rotation to match the desired constant angular velocity
            step_vel = (vel_vec / vel_norm) * min(vel_norm, ANGULAR_VELOCITY)
            mujoco.mju_quatIntegrate(self.current_target_quat, step_vel, dt)

        # 3. Waypoint Logic & Visualization Update
        if dist <= ARRIVAL_TOLERANCE:
            print(f"WP {self.waypoint_idx} reached. Accuracy: {dist:.5f}m")
            self.waypoint_idx += 1
            if self.waypoint_idx < len(WAYPOINTS):
                self.model.body_pos[self.target_body_id] = WAYPOINTS[self.waypoint_idx][:3]
                new_q = np.zeros(4)
                mujoco.mju_euler2Quat(new_q, WAYPOINTS[self.waypoint_idx][3:], 'xyz')
                self.model.body_quat[self.target_body_id] = new_q

        # 4. Perform Precision IK Solve
        self.solve_ik(self.current_target_pos, self.current_target_quat)
        return False

def main():
    model = create_model_with_target_pose(XML_PATH, WAYPOINTS[0][:3], WAYPOINTS[0][3:])
    data = mujoco.MjData(model)
    follower = PrecisionPoseFollower(model, data)

    sys.stdout.write("\033[2J\033[H")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            finished = follower.update(model.opt.timestep)
            mujoco.mj_step(model, data)
            viewer.sync()

            if finished: break

            if int(data.time * 100) % 10 == 0:
                # Calculate current errors for the dashboard
                curr_pos = data.body(follower.ee_id).xpos
                pos_error = np.linalg.norm(curr_pos - follower.current_target_pos)
                
                # Orientation error magnitude
                curr_mat = data.body(follower.ee_id).xmat.reshape(3, 3)
                target_mat = np.zeros(9)
                mujoco.mju_quat2Mat(target_mat, follower.current_target_quat)
                rot_error_mat = target_mat.reshape(3, 3) @ curr_mat.T
                rot_error_quat = np.zeros(4)
                mujoco.mju_mat2Quat(rot_error_quat, rot_error_mat.flatten())
                # Angle difference in radians
                angle_error = 2 * np.arccos(np.clip(abs(rot_error_quat[0]), 0, 1))

                sys.stdout.write("\033[H")
                print("================ HIGH PRECISION POSE DASHBOARD ================")
                print(f" Sim Time: {data.time:6.2f}s | Target WP: {follower.waypoint_idx}")
                print(f" Pos Error (m):   {pos_error:10.6f}")
                print(f" Ori Error (rad): {angle_error:10.6f} (Approx {np.rad2deg(angle_error):.2f}°)")
                print(f" L2 Dist to WP:  {np.linalg.norm(curr_pos - WAYPOINTS[min(follower.waypoint_idx, len(WAYPOINTS)-1)][:3]):.4f}m")
                print("===============================================================")

            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == "__main__":
    main()