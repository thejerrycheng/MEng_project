import numpy as np
import torch
import torch.nn as nn
import mujoco

# -------------------------
# IRIS Forward Kinematics
# -------------------------

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
        return T[:3,3]   # end-effector xyz


# -------------------------
# Loss Functions
# -------------------------

mse = nn.MSELoss()
_fk = IRISKinematics()


def batch_fk(q_batch_rad):
    """
    q_batch_rad: (B,6) torch tensor in radians
    returns: (B,3) torch tensor of xyz
    """
    q_deg = torch.rad2deg(q_batch_rad).detach().cpu().numpy()
    xyz = [_fk.forward(q) for q in q_deg]
    return torch.tensor(xyz, device=q_batch_rad.device, dtype=torch.float32)


def act_loss(pred_delta, future_delta, joint_seq, goal_xyz,
             lambda_cont, lambda_goal):
    """
    pred_delta: (B,F,6) predicted Δq
    future_delta: (B,F,6) GT Δq
    joint_seq: (B,S,6) observed joints
    goal_xyz: (B,3) Cartesian EE goal
    """

    # 1. Imitation loss on action chunk
    loss_mse = mse(pred_delta, future_delta)

    # 2. Continuity loss (first predicted Δq should be zero)
    loss_cont = mse(pred_delta[:,0,:], torch.zeros_like(pred_delta[:,0,:]))

    # 3. Goal-reaching loss in Cartesian space
    q_last = joint_seq[:,-1,:]           # last observed absolute q
    q_pred_last = q_last + pred_delta[:,-1,:]  # predicted final absolute q

    xyz_pred = batch_fk(q_pred_last)     # FK
    loss_goal = mse(xyz_pred, goal_xyz)

    return loss_mse + lambda_cont*loss_cont + lambda_goal*loss_goal
