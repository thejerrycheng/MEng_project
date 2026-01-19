import numpy as np
import mujoco

# ==================================================
# 1. Your Original Analytical Class (Ground Truth)
# ==================================================
class IRISKinematics:
    def __init__(self):
        self.link_configs = [
            {'pos': [0, 0, 0.2487],         'euler': [0, 0, 0],      'axis': [0, 0, 1]},
            {'pos': [0.0218, 0, 0.059],     'euler': [0, 90, 180],   'axis': [0, 0, 1]},
            {'pos': [0.299774, 0, -0.0218], 'euler': [0, 0, 0],      'axis': [0, 0, 1]},
            {'pos': [0.02, 0, 0],           'euler': [0, 90, 0],     'axis': [0, 0, 1]},
            {'pos': [0, 0, 0.315],          'euler': [0, -90, 0],    'axis': [0, 0, 1]},
            {'pos': [0.042824, 0, 0],       'euler': [0, 90, 180],   'axis': [0, 0, 1]},
            {'pos': [0, 0, 0],              'euler': [0, 0, 0],      'axis': [0, 0, 0]} 
        ]

    def _get_transform(self, cfg, q_rad):
        T_pos = np.eye(4)
        T_pos[:3, 3] = cfg['pos']
        R_fixed = np.eye(3)
        if any(cfg['euler']):
            quat = np.zeros(4)
            mujoco.mju_euler2Quat(quat, np.deg2rad(cfg['euler']), 'xyz')
            mat = np.zeros(9)
            mujoco.mju_quat2Mat(mat, quat)
            R_fixed = mat.reshape(3, 3)
        T_rot_fixed = np.eye(4)
        T_rot_fixed[:3, :3] = R_fixed
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
        T_accum = np.eye(4)
        for i in range(6): 
            T_accum = T_accum @ self._get_transform(self.link_configs[i], q[i])
        T_accum = T_accum @ self._get_transform(self.link_configs[6], 0)
        return T_accum[:3, 3]

# ==================================================
# 2. DH Table Implementation
# ==================================================
class DHKinematics:
    def __init__(self):
        # Parameters derived from your code
        # alpha(i-1), a(i-1), d(i), theta_offset
        self.dh_params = [
            [0,           0,        0.2487,  0],  # Joint 1
            [np.pi/2,     0.0218,   0.059,   0],  # Joint 2
            [0,           0.299774, -0.0218, 0],  # Joint 3
            [np.pi/2,     0.02,     0,       0],  # Joint 4
            [-np.pi/2,    0,        0.315,   0],  # Joint 5
            [np.pi/2,     0.042824, 0,       0],  # Joint 6
        ]

    def forward(self, q):
        T = np.eye(4)
        
        for i, (alpha, a, d, offset) in enumerate(self.dh_params):
            theta = q[i] + offset
            
            # Standard DH Matrix
            # Trans_z(d) * Rot_z(theta) * Trans_x(a) * Rot_x(alpha)
            # Note: The order depends on convention (Craig vs Spong). 
            # Given the link_configs structure (Pos then Rot), we use:
            
            # 1. Rot_x(alpha)
            c_al, s_al = np.cos(alpha), np.sin(alpha)
            Rx = np.array([
                [1, 0, 0, 0],
                [0, c_al, -s_al, 0],
                [0, s_al, c_al, 0],
                [0, 0, 0, 1]
            ])
            
            # 2. Trans_x(a)
            Tx = np.eye(4)
            Tx[0, 3] = a
            
            # 3. Rot_z(theta)
            c_th, s_th = np.cos(theta), np.sin(theta)
            Rz = np.array([
                [c_th, -s_th, 0, 0],
                [s_th, c_th, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            
            # 4. Trans_z(d)
            Tz = np.eye(4)
            Tz[2, 3] = d
            
            # Combine: T(i-1 -> i)
            # This specific order matches your link_config flow:
            # Shift X/Z (a/d), then Rotate frames
            step = Rx @ Tx @ Rz @ Tz 
            T = T @ step

        return T[:3, 3]

# ==================================================
# 3. Verification
# ==================================================
if __name__ == "__main__":
    original = IRISKinematics()
    dh = DHKinematics()
    
    print(f"{'Test':<5} | {'Original':<25} | {'DH Table':<25} | {'Diff'}")
    print("-" * 70)
    
    for i in range(5):
        q = np.random.uniform(-1, 1, 6)
        
        # Calculate
        pos1 = original.forward(q)
        pos2 = dh.forward(q) # Note: DH often needs tuning on matrix order
        
        diff = np.linalg.norm(pos1 - pos2)
        print(f"{i:<5} | {str(pos1[:2]):<25} | {str(pos2[:2]):<25} | {diff:.5f}")