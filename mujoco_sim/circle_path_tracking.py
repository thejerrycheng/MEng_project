import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import argparse

# --- Configuration & Defaults ---
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURR_DIR, "assets", "iris.xml")
START_Q = np.array([0, -30, -100, 0, 0, 0], dtype=np.float64) 

# Controller Gains
KP = 1000.0  
KD = 80.0   

# Trajectory Settings
CIRCLE_SPEED = 1.0 
NUM_PATH_POINTS = 50 

def get_lookat_rotation(current_pos, target_center):
    """
    Calculates a rotation matrix where the Z-axis (end-effector blue axis) 
    points from current_pos toward the target_center.
    """
    z_axis = target_center - current_pos
    z_axis /= (np.linalg.norm(z_axis) + 1e-6)
    
    world_up = np.array([0, 0, 1])
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
        mujoco.mj_forward(model, data)
        self.damping = 1e-4

    def get_control_torque(self, target_pos, target_center):
        curr_pos = self.data.body(self.ee_id).xpos
        curr_mat = self.data.body(self.ee_id).xmat.reshape(3, 3)

        target_mat = get_lookat_rotation(curr_pos, target_center)

        pos_err = target_pos - curr_pos
        rot_err_mat = target_mat @ curr_mat.T
        rot_err_quat = np.zeros(4)
        mujoco.mju_mat2Quat(rot_err_quat, rot_err_mat.flatten())
        rot_err_vec = rot_err_quat[1:] * np.sign(rot_err_quat[0])
        
        cart_err = np.concatenate([pos_err, rot_err_vec])

        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_id)
        J = np.vstack([jacp, jacr])[:, :6]
        
        J_inv = J.T @ np.linalg.solve(J @ J.T + self.damping * np.eye(6), np.eye(6))
        q_err = J_inv @ cart_err
        dq_err = -self.data.qvel[:6]
        
        q_accel_des = KP * q_err + KD * dq_err

        self.data.qacc[:6] = q_accel_des
        mujoco.mj_inverse(self.model, self.data)
        
        return self.data.qfrc_inverse[:6].copy()

def main():
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(description="6-DOF Arm Circle Tracking with Inverse Dynamics")
    parser.add_argument("--radius", "-r", type=float, default=0.15, help="Radius of the circle (meters)")
    parser.add_argument("--center", "-c", type=float, nargs=3, default=[0.45, 0.0, 0.4], 
                        help="Center of the circle as: x y z")
    args = parser.parse_args()

    center = np.array(args.center)
    radius = args.radius

    # Load model
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    controller = IDController(model, data)

    # Pre-calculate Path Dots for Visualization
    path_dots = []
    for i in range(NUM_PATH_POINTS):
        theta = (i / NUM_PATH_POINTS) * 2 * np.pi
        path_dots.append([
            center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
            center[2]
        ])

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # Update target in X-Y Plane relative to center
            t = data.time
            target_pos = np.array([
                center[0] + radius * np.cos(CIRCLE_SPEED * t),
                center[1] + radius * np.sin(CIRCLE_SPEED * t),
                center[2]
            ])

            # Apply ID Control
            tau = controller.get_control_torque(target_pos, center)
            data.ctrl[:6] = tau
            
            mujoco.mj_step(model, data)

            # Visualization Rendering
            viewer.user_scn.ngeom = 0
            # Static Circle Path
            for dot_pos in path_dots:
                mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.005, 0, 0], 
                    pos=dot_pos, mat=np.eye(3).flatten(), rgba=[0, 1, 0, 0.3])
                viewer.user_scn.ngeom += 1

            # Red dot at center point
            mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.012, 0, 0], 
                pos=center, mat=np.eye(3).flatten(), rgba=[1, 0, 0, 1])
            viewer.user_scn.ngeom += 1

            # Bright green dot at current target
            mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0], 
                pos=target_pos, mat=np.eye(3).flatten(), rgba=[0, 1, 0, 1])
            viewer.user_scn.ngeom += 1

            viewer.sync()
            
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == "__main__":
    main()