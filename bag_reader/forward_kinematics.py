#!/usr/bin/env python3
import os
import numpy as np
import mujoco
import argparse

# ==================================================
# Configuration
# ==================================================
# Update this to your actual XML path
MUJOCO_SIM_DIR = "/home/jerry/Desktop/MEng_project/mujoco_sim"
XML_PATH = os.path.join(MUJOCO_SIM_DIR, "assets", "iris.xml")

NUM_TESTS = 1000

# ==================================================
# 1. Analytical Solver (The Python Class)
# ==================================================
class IRISKinematics:
    def __init__(self):
        # Physical parameters extracted from XML
        self.link_configs = [
            {'pos': [0, 0, 0.2487],         'euler': [0, 0, 0],      'axis': [0, 0, 1]}, # Joint 1
            {'pos': [0.0218, 0, 0.059],     'euler': [0, 90, 180],   'axis': [0, 0, 1]}, # Joint 2
            {'pos': [0.299774, 0, -0.0218], 'euler': [0, 0, 0],      'axis': [0, 0, 1]}, # Joint 3
            {'pos': [0.02, 0, 0],           'euler': [0, 90, 0],     'axis': [0, 0, 1]}, # Joint 4
            {'pos': [0, 0, 0.315],          'euler': [0, -90, 0],    'axis': [0, 0, 1]}, # Joint 5
            {'pos': [0.042824, 0, 0],       'euler': [0, 90, 180],   'axis': [0, 0, 1]}, # Joint 6
            {'pos': [0, 0, 0],              'euler': [0, 0, 0],      'axis': [0, 0, 0]}  # EE Mount
        ]

    def _get_transform(self, cfg, q_rad):
        # Translation
        T_pos = np.eye(4)
        T_pos[:3, 3] = cfg['pos']

        # Rotation (Fixed)
        R_fixed = np.eye(3)
        if any(cfg['euler']):
            quat = np.zeros(4)
            mujoco.mju_euler2Quat(quat, np.deg2rad(cfg['euler']), 'xyz')
            mat = np.zeros(9)
            mujoco.mju_quat2Mat(mat, quat)
            R_fixed = mat.reshape(3, 3)
        
        T_rot_fixed = np.eye(4)
        T_rot_fixed[:3, :3] = R_fixed

        # Rotation (Joint)
        T_joint = np.eye(4)
        if np.any(cfg['axis']):
            quat_j = np.zeros(4)
            mujoco.mju_axisAngle2Quat(quat_j, np.array(cfg['axis']), q_rad)
            mat_j = np.zeros(9)
            mujoco.mju_quat2Mat(mat_j, quat_j)
            R_joint = mat_j.reshape(3, 3)
            T_joint[:3, :3] = R_joint

        return T_pos @ T_rot_fixed @ T_joint

    def forward(self, q):
        T_accumulated = np.eye(4)
        for i in range(6): 
            T_link = self._get_transform(self.link_configs[i], q[i])
            T_accumulated = T_accumulated @ T_link
        
        T_ee = self._get_transform(self.link_configs[6], 0)
        T_accumulated = T_accumulated @ T_ee
        return T_accumulated[:3, 3]

# ==================================================
# 2. MuJoCo Solver (Ground Truth)
# ==================================================
class MujocoOracle:
    def __init__(self, xml_path):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML not found: {xml_path}")
            
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # We target "ee_mount" body because that matches the end of the analytical chain
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ee_mount")
        if self.ee_id == -1:
            print("Warning: Body 'ee_mount' not found. Defaulting to last body.")
            self.ee_id = self.model.nbody - 1

    def forward(self, q):
        self.data.qpos[:6] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.xpos[self.ee_id].copy()

# ==================================================
# 3. Main Comparison Loop
# ==================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default=XML_PATH, help="Path to iris.xml")
    args = parser.parse_args()

    print(f"Initializing Comparison...")
    print(f"XML Source: {args.xml}")

    try:
        analytical = IRISKinematics()
        oracle = MujocoOracle(args.xml)
    except Exception as e:
        print(f"Initialization Failed: {e}")
        return

    errors = []
    
    print(f"\nRunning {NUM_TESTS} random configurations...\n")
    print(f"{'Test ID':<10} | {'Analytical (XYZ)':<30} | {'MuJoCo (XYZ)':<30} | {'Error (m)':<10}")
    print("-" * 90)

    for i in range(NUM_TESTS):
        # Generate random joint angles within a reasonable range (-pi to pi)
        q_rand = np.random.uniform(-np.pi, np.pi, size=6)
        
        # 1. Compute Analytical
        pos_analytical = analytical.forward(q_rand)
        
        # 2. Compute MuJoCo
        pos_mujoco = oracle.forward(q_rand)
        
        # 3. Compare
        dist = np.linalg.norm(pos_analytical - pos_mujoco)
        errors.append(dist)

        # Print first 10 for visual check
        if i < 10:
            str_ana = f"[{pos_analytical[0]:.3f}, {pos_analytical[1]:.3f}, {pos_analytical[2]:.3f}]"
            str_muj = f"[{pos_mujoco[0]:.3f}, {pos_mujoco[1]:.3f}, {pos_mujoco[2]:.3f}]"
            print(f"{i:<10} | {str_ana:<30} | {str_muj:<30} | {dist:.6f}")

    # ==================================================
    # Results
    # ==================================================
    errors = np.array(errors)
    mean_err = np.mean(errors)
    max_err = np.max(errors)
    
    print("-" * 90)
    print(f"\nSUMMARY RESULTS ({NUM_TESTS} tests):")
    print(f"  Mean Euclidean Error: {mean_err:.8f} m")
    print(f"  Max Euclidean Error:  {max_err:.8f} m")
    
    if max_err < 1e-4: # 0.1 mm tolerance
        print("\n[SUCCESS] Analytical model matches MuJoCo simulation!")
    else:
        print("\n[WARNING] Discrepancy detected. Check link offsets or frame definitions.")

if __name__ == "__main__":
    main()