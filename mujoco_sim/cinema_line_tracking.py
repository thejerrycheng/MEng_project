import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import argparse

# --- FIX FOR MACOS THREADING ERROR ---
import matplotlib
# Try TkAgg; if you have issues, 'Qt5Agg' is a strong alternative on Mac
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
# -------------------------------------

# --- Configuration ---
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURR_DIR, "assets", "iris.xml")
START_Q = np.array([0, -20, -100, 0, 40, 0], dtype=np.float64) 

# Gains for high-performance tracking
KP = 1500.0  
KD = 120.0   

def smoothstep(t):
    """Quintic polynomial for zero velocity AND zero acceleration at boundaries."""
    return 6*t**5 - 15*t**4 + 10*t**3

class IDController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")
        self.data.qpos[:6] = np.deg2rad(START_Q)
        mujoco.mj_forward(model, data)

    def get_control_torque(self, target_pos, target_obj):
        curr_pos = self.data.body(self.ee_id).xpos
        curr_mat = self.data.body(self.ee_id).xmat.reshape(3, 3)
        
        # Dynamic Look-At Orientation Logic
        z_axis = (target_obj - curr_pos)
        z_axis /= (np.linalg.norm(z_axis) + 1e-6)
        up = np.array([0, 0, 1])
        if abs(np.dot(z_axis, up)) > 0.99: up = np.array([0, 1, 0])
        x_axis = np.cross(up, z_axis)
        x_axis /= (np.linalg.norm(x_axis) + 1e-6)
        y_axis = np.cross(z_axis, x_axis)
        target_mat = np.stack([x_axis, y_axis, z_axis], axis=1)

        # Error Calculation
        pos_err = target_pos - curr_pos
        rot_err_mat = target_mat @ curr_mat.T
        rot_err_quat = np.zeros(4)
        mujoco.mju_mat2Quat(rot_err_quat, rot_err_mat.flatten())
        rot_err_vec = rot_err_quat[1:] * np.sign(rot_err_quat[0])
        
        cart_err = np.concatenate([pos_err, rot_err_vec])
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_id)
        J = np.vstack([jacp, jacr])[:, :6]
        
        J_inv = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.eye(6))
        q_accel_des = KP * (J_inv @ cart_err) - KD * self.data.qvel[:6]

        self.data.qacc[:6] = q_accel_des
        mujoco.mj_inverse(self.model, self.data)
        return self.data.qfrc_inverse[:6].copy(), pos_err

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['pan', 'dolly', 'crane'], default='pan')
    parser.add_argument("--duration", type=float, default=4.0, help="Duration for one-way move")
    parser.add_argument("--subject", type=float, nargs=3, help="Override focus point [x y z]")
    args = parser.parse_args()

    # Preset coordinates
    presets = {
        'pan': ([0.4, -0.25, 0.4], [0.4, 0.25, 0.4], [0.6, 0.0, 0.2]),
        'dolly': ([0.25, 0.0, 0.4], [0.55, 0.0, 0.4], [0.7, 0.0, 0.4]),
        'crane': ([0.4, 0.0, 0.15], [0.4, 0.0, 0.55], [0.6, 0.0, 0.1])
    }
    start_pt, end_pt, default_subject = presets[args.mode]
    subject = np.array(args.subject) if args.subject else np.array(default_subject)

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    controller = IDController(model, data)

    # Data collection
    times, velocities, accelerations, errors = [], [], [], []
    last_vel = np.zeros(3)
    tracking_started = False
    total_sim_time = args.duration * 2

    # Pre-calculate dots for visualization
    NUM_DOTS = 25
    dots = [np.array(start_pt) + (np.array(end_pt) - np.array(start_pt)) * (i/(NUM_DOTS-1)) for i in range(NUM_DOTS)]

    # PASSIVE VIEWER LAUNCH
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while data.time < total_sim_time and viewer.is_running():
            step_start = time.time()
            
            norm_time = data.time / args.duration
            t_smooth = smoothstep(norm_time) if norm_time <= 1.0 else smoothstep(2.0 - norm_time)
            target_pos = np.array(start_pt) + (np.array(end_pt) - np.array(start_pt)) * t_smooth
            
            tau, p_err = controller.get_control_torque(target_pos, subject)
            data.ctrl[:6] = tau
            mujoco.mj_step(model, data)

            if not tracking_started and np.linalg.norm(p_err) < 0.01:
                tracking_started = True

            if tracking_started:
                jacp = np.zeros((3, model.nv))
                mujoco.mj_jacBody(model, data, jacp, None, controller.ee_id)
                curr_vel = jacp[:, :6] @ data.qvel[:6]
                curr_acc = (curr_vel - last_vel) / model.opt.timestep
                times.append(data.time)
                velocities.append(np.linalg.norm(curr_vel))
                accelerations.append(np.linalg.norm(curr_acc))
                errors.append(np.linalg.norm(p_err) * 1000)
                last_vel = curr_vel

            # --- VISUALIZATION ---
            viewer.user_scn.ngeom = 0
            mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.04, 0.04, 0.04], pos=subject, mat=np.eye(3).flatten(), rgba=[1, 0.8, 0, 1])
            viewer.user_scn.ngeom += 1
            mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.015, 0, 0], pos=start_pt, mat=np.eye(3).flatten(), rgba=[0, 0, 1, 1])
            viewer.user_scn.ngeom += 1
            mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.015, 0, 0], pos=end_pt, mat=np.eye(3).flatten(), rgba=[1, 0, 0, 1])
            viewer.user_scn.ngeom += 1
            for d_pos in dots:
                mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.003, 0, 0], pos=d_pos, mat=np.eye(3).flatten(), rgba=[1, 1, 1, 0.3])
                viewer.user_scn.ngeom += 1
            mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.012, 0, 0], pos=target_pos, mat=np.eye(3).flatten(), rgba=[0, 1, 0, 1])
            viewer.user_scn.ngeom += 1

            viewer.sync()
            time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))

    # --- PLOTTING (Ensures this happens on the main thread after viewer closes) ---
    if times:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        fig.suptitle(f'Cinematic Tracking Report: {args.mode.upper()}', fontsize=16)

        ax1.plot(times, velocities, color='tab:blue', lw=2); ax1.set_ylabel('Velocity (m/s)'); ax1.grid(True, alpha=0.3)
        ax2.plot(times, accelerations, color='tab:red', lw=2); ax2.set_ylabel('Accel (m/s²)'); ax2.grid(True, alpha=0.3)
        ax2.set_title('Justification of Smoothness (Continuous Acceleration Profile)')
        ax3.plot(times, errors, color='tab:green', lw=2); ax3.set_ylabel('Tracking Error (mm)'); ax3.set_xlabel('Time (s)'); ax3.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

if __name__ == "__main__":
    main()