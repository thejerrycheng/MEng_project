import numpy as np
import mujoco

class IRISKinematics:
    def __init__(self):
        self.link_configs = [
            {'pos':[0,0,0.2487],        'euler':[0,0,0],     'axis':[0,0,1]},
            {'pos':[0.0218,0,0.059],    'euler':[0,90,180],  'axis':[0,0,1]},
            {'pos':[0.299774,0,-0.0218],'euler':[0,0,0],     'axis':[0,0,1]},
            {'pos':[0.02,0,0],          'euler':[0,90,0],    'axis':[0,0,1]},
            {'pos':[0,0,0.315],         'euler':[0,-90,0],   'axis':[0,0,1]},
            {'pos':[0.042824,0,0],      'euler':[0,90,0],    'axis':[0,0,1]}
        ]

    def get_local_transform(self, cfg, q_rad):
        T = np.eye(4)
        T[:3,3] = cfg['pos']

        R_fixed = np.eye(3)
        if any(cfg['euler']):
            quat = np.zeros(4)
            mujoco.mju_euler2Quat(quat, np.deg2rad(cfg['euler']), 'xyz')
            mat = np.zeros(9)
            mujoco.mju_quat2Mat(mat, quat)
            R_fixed = mat.reshape(3,3)

        quat_j = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat_j, np.array(cfg['axis']), q_rad)
        mat_j = np.zeros(9)
        mujoco.mju_quat2Mat(mat_j, quat_j)
        R_joint = mat_j.reshape(3,3)

        T[:3,:3] = R_fixed @ R_joint
        return T

    def forward(self, q_deg):
        q_rad = np.deg2rad(q_deg)
        T = np.eye(4)
        for i in range(len(self.link_configs)):
            T = T @ self.get_local_transform(self.link_configs[i], q_rad[i])
        return T[:3,3]
