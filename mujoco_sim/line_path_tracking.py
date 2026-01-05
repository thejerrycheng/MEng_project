import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import argparse
import sys

# --- Configuration & Defaults ---
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURR_DIR, "assets", "iris.xml")
# Starting pose (elbow up, ready for forward reaching)
START_Q = np.array([0, -30, -90, 0, 0, 0], dtype=np.float64) 

# Controller Gains
KP = 1200.0  
KD = 90.0   

# Trajectory Speed (radians per second for the cosine interpolation)
TRAJ_SPEED = 0.8 
NUM_PATH_POINTS = 60 # Density of dots visualizing the line

def get_lookat_rotation(current_pos, target_center):
    """
    Calculates rotation matrix: Z-axis points from current_pos to target_center.
    Uses World-Z as up-vector, switching to World-Y near singularities.
    """
    z_axis = target_center - current_pos
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-6: return np.eye(3) # Prevent divide by zero if on top of target
    z_axis /= z_norm
    
    world_up = np.array([0, 0, 1])
    # If Z-axis is too close to World-Up, swap World-Up axis to avoid singularity
    if abs(np.dot(z_axis, world_up)) > 0.99:
        world_up = np.array([0, 1, 0])
        
    x_axis = np.cross(world_up, z_axis)
    x_axis /= (np.linalg.norm(x_axis) + 1e-6)
    y_axis = np.cross(z_axis, x_axis)
    
    return np.stack([x_axis, y_axis, z_axis], axis=1)

class IDController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")
        
        # Initial pose
        self.data.qpos[:6] = np.deg2rad(START_Q)
        self.data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        self.damping = 1e-3 # Slightly higher damping for stability

    def get_control_torque(self, target_pos_on_line, target_object_center):
        curr_pos = self.data.body(self.ee_id).xpos
        curr_mat = self.data.body(self.ee_id).xmat.reshape(3, 3)

        # 1. Determine desired orientation (Look-At)
        target_mat = get_lookat_rotation(curr_pos, target_object_center)

        # 2. Calculate Errors
        pos_err = target_pos_on_line - curr_pos
        
        # Orientation error via quaternions
        rot_err_mat = target_mat @ curr_mat.T
        rot_err_quat = np.zeros(4)
        mujoco.mju_mat2Quat(rot_err_quat, rot_err_mat.flatten())
        # Ensure shortest path rotation
        rot_err_vec = rot_err_quat[1:] * np.sign(rot_err_quat[0])
        
        cart_err = np.concatenate([pos_err, rot_err_vec])

        # 3. Jacobian & Inverse Dynamics
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_id)
        J = np.vstack([jacp, jacr])[:, :6]
        
        # Dampened Pseudo-inverse
        J_inv = J.T @ np.linalg.solve(J @ J.T + self.damping * np.eye(6), np.eye(6))
        
        # PD Control in Joint Space
        q_err = J_inv @ cart_err
        dq_err = -self.data.qvel[:6] # Target velocity is 0 for simple tracking
        q_accel_des = KP * q_err + KD * dq_err

        # Compute required torques
        self.data.qacc[:6] = q_accel_des
        mujoco.mj_inverse(self.model, self.data)
        
        return self.data.qfrc_inverse[:6].copy()

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="6-DOF Arm Line Tracking with Look-At Constraint")
    parser.add_argument("--object", type=float, nargs=3, required=True, metavar=('X','Y','Z'),
                        help="Center coordinates of the target cube object.")
    parser.add_argument("--start", type=float, nargs=3, required=True, metavar=('X','Y','Z'),
                        help="Starting coordinates of the line segment.")
    parser.add_argument("--dir", type=float, nargs=3, required=True, metavar=('X','Y','Z'),
                        help="Direction vector of the line (will be normalized).")
    parser.add_argument("--length", type=float, required=True, 
                        help="Total length of the line segment.")
    
    args = parser.parse_args()

    # Process inputs
    target_obj_pos = np.array(args.object)
    line_start_pos = np.array(args.start)
    line_dir_raw = np.array(args.dir)
    line_len = args.length

    # Normalize direction vector
    dir_norm = np.linalg.norm(line_dir_raw)
    if dir_norm < 1e-6:
        print("Error: Direction vector cannot be zero.")
        sys.exit(1)
    line_dir_norm = line_dir_raw / dir_norm
    
    line_end_pos = line_start_pos + line_dir_norm * line_len

    # Load model
    if not os.path.exists(XML_PATH): print(f"XML not found: {XML_PATH}"); sys.exit(1)
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    controller = IDController(model, data)

    # Pre-calculate visualization dots along the line
    path_dots = []
    for i in range(NUM_PATH_POINTS + 1):
        dist = (i / NUM_PATH_POINTS) * line_len
        path_dots.append(line_start_pos + line_dir_norm * dist)

    print("\nStarting Simulation...")
    print(f"Tracking Line from {line_start_pos} to {line_end_pos}")
    print(f"Looking at Target Cube positioned at {target_obj_pos}\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # viewer.cam.lookat[:] = target_obj_pos # Optional: center camera on target
        # viewer.cam.distance = 1.5

        while viewer.is_running():
            step_start = time.time()
            t = data.time

            # --- Trajectory Generation ---
            # Oscillate distance between 0 and line_len using cosine for smooth reversals
            # dist_along_line = line_len * (0.5 * (1 - np.cos(TRAJ_SPEED * t)))
            
            # Alternative: Linear triangle wave for constant velocity relative to line
            period = (2 * np.pi) / TRAJ_SPEED * 2 
            phase = (t % period) / period
            if phase < 0.5:
                dist_along_line = line_len * (phase * 2)
            else:
                dist_along_line = line_len * (2 - phase * 2)

            current_target_on_line = line_start_pos + line_dir_norm * dist_along_line

            # --- Control Step ---
            tau = controller.get_control_torque(current_target_on_line, target_obj_pos)
            data.ctrl[:6] = tau
            mujoco.mj_step(model, data)

            # --- Visualization Rendering ---
            viewer.user_scn.ngeom = 0 # Reset user geometry buffer

            # 1. Draw Static Target Object (Red Cube)
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_BOX, 
                size=[0.03, 0.03, 0.03], # Half-extents (0.06m edge length)
                pos=target_obj_pos, 
                mat=np.eye(3).flatten(), 
                rgba=[0.8, 0.1, 0.1, 1] # Red solid
            )
            viewer.user_scn.ngeom += 1

            # 2. Draw Static Line Path (Green transparent dots)
            for dot_pos in path_dots:
                mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.004, 0, 0], 
                    pos=dot_pos, mat=np.eye(3).flatten(), rgba=[0, 1, 0, 0.3])
                viewer.user_scn.ngeom += 1

            # 3. Draw Start (Blue) and End (Red) markers of the line
            mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0], pos=line_start_pos, 
                mat=np.eye(3).flatten(), rgba=[0, 0, 1, 0.8]) # Start Blue
            viewer.user_scn.ngeom += 1
            mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0], pos=line_end_pos, 
                mat=np.eye(3).flatten(), rgba=[1, 0, 0, 0.8]) # End Red
            viewer.user_scn.ngeom += 1

            # 4. Draw Active Target on Line (Large Bright Green Sphere)
            mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.018, 0, 0], 
                pos=current_target_on_line, 
                mat=np.eye(3).flatten(), rgba=[0, 1, 0.2, 1])
            viewer.user_scn.ngeom += 1

            viewer.sync()
            
            # Timekeeping
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == "__main__":
    main()