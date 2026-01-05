import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import argparse
import csv

# --- MACOS TRACE TRAP FIX ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# ----------------------------

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURR_DIR, "assets", "iris.xml")
START_Q = np.array([0, -20, -100, 0, 40, 0], dtype=np.float64) 

KP, KD = 1500.0, 120.0   

def smoothstep(t):
    return 6*t**5 - 15*t**4 + 10*t**3

class IDController:
    def __init__(self, model, data):
        self.model, self.data = model, data
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")
        self.data.qpos[:6] = np.deg2rad(START_Q)
        mujoco.mj_forward(model, data)

    def get_control_torque(self, target_pos, target_obj):
        curr_pos = self.data.body(self.ee_id).xpos
        curr_mat = self.data.body(self.ee_id).xmat.reshape(3, 3)
        
        # Look-At Orientation Logic
        z_axis = (target_obj - curr_pos)
        z_axis /= (np.linalg.norm(z_axis) + 1e-6)
        up = np.array([0, 0, 1])
        if abs(np.dot(z_axis, up)) > 0.99: up = np.array([0, 1, 0])
        x_axis = np.cross(up, z_axis)
        x_axis /= (np.linalg.norm(x_axis) + 1e-6)
        y_axis = np.cross(z_axis, x_axis)
        target_mat = np.stack([x_axis, y_axis, z_axis], axis=1)

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
        return self.data.qfrc_inverse[:6].copy(), pos_err, curr_pos.copy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['pan', 'dolly', 'crane'], default='pan')
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--subject", type=float, nargs=3)
    args = parser.parse_args()

    presets = {
        'pan': ([0.4, -0.25, 0.4], [0.4, 0.25, 0.4], [0.6, 0.0, 0.2]),
        'dolly': ([0.25, 0.0, 0.4], [0.55, 0.0, 0.4], [0.7, 0.0, 0.4]),
        'crane': ([0.4, 0.0, 0.15], [0.4, 0.0, 0.55], [0.6, 0.0, 0.1])
    }
    start_pt, end_pt, d_subj = presets[args.mode]
    subject = np.array(args.subject) if args.subject else np.array(d_subj)

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    controller = IDController(model, data)

    csv_data = []
    times, velocities, accelerations, errors = [], [], [], []
    last_vel = np.zeros(3)
    total_sim_time = args.duration * 2
    dots = [np.array(start_pt) + (np.array(end_pt) - np.array(start_pt)) * (i/24) for i in range(25)]

    print(f"\n🎥 CINEMA ROBOT: Executing {args.mode.upper()} shot...")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while data.time < total_sim_time and viewer.is_running():
                step_start = time.time()
                
                # Path Generation
                norm_time = data.time / args.duration
                t_smooth = smoothstep(norm_time) if norm_time <= 1.0 else smoothstep(2.0 - norm_time)
                target_pos = np.array(start_pt) + (np.array(end_pt) - np.array(start_pt)) * t_smooth
                
                tau, p_err, actual_pos = controller.get_control_torque(target_pos, subject)
                data.ctrl[:6] = tau
                mujoco.mj_step(model, data)

                # Data Logging
                jacp = np.zeros((3, model.nv))
                mujoco.mj_jacBody(model, data, jacp, None, controller.ee_id)
                cv = jacp[:, :6] @ data.qvel[:6]
                ca = (cv - last_vel) / model.opt.timestep
                
                csv_data.append({
                    'time': data.time,
                    'target_x': target_pos[0], 'target_y': target_pos[1], 'target_z': target_pos[2],
                    'actual_x': actual_pos[0], 'actual_y': actual_pos[1], 'actual_z': actual_pos[2],
                    'vel_x': cv[0], 'vel_y': cv[1], 'vel_z': cv[2],
                    'accel_x': ca[0], 'accel_y': ca[1], 'accel_z': ca[2]
                })

                times.append(data.time); velocities.append(np.linalg.norm(cv))
                accelerations.append(np.linalg.norm(ca)); errors.append(np.linalg.norm(p_err) * 1000)
                last_vel = cv.copy()

                # --- PATH VISUALIZATION IN MUJOCO ---
                viewer.user_scn.ngeom = 0
                
                # Subject (Yellow Cube)
                mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], 
                                    mujoco.mjtGeom.mjGEOM_BOX, [0.04, 0.04, 0.04], 
                                    subject, np.eye(3).flatten(), [1, 0.8, 0, 1])
                viewer.user_scn.ngeom += 1
                
                # Breadcrumbs (White Path)
                for d_pos in dots:
                    mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], 
                                        mujoco.mjtGeom.mjGEOM_SPHERE, [0.003, 0, 0], 
                                        d_pos, np.eye(3).flatten(), [1, 1, 1, 0.3])
                    viewer.user_scn.ngeom += 1
                
                # Active Target (Green Sphere)
                mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], 
                                    mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0], 
                                    target_pos, np.eye(3).flatten(), [0, 1, 0, 1])
                viewer.user_scn.ngeom += 1

                viewer.sync()
                time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))

    finally:
        if csv_data:
            csv_path = os.path.join(CURR_DIR, f"{args.mode}_telemetry.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                writer.writeheader()
                writer.writerows(csv_data)
            print(f"\n[SUCCESS] CSV saved: {csv_path}")

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
            ax1.plot(times, velocities, color='tab:blue'); ax1.set_ylabel('Velocity (m/s)')
            ax2.plot(times, accelerations, color='tab:red'); ax2.set_ylabel('Accel (m/s²)')
            ax3.plot(times, errors, color='tab:green'); ax3.set_ylabel('Error (mm)')
            plt.savefig(os.path.join(CURR_DIR, f"{args.mode}_report.png"), dpi=300)
            print(f"[SUCCESS] Report image saved.")

if __name__ == "__main__":
    main()