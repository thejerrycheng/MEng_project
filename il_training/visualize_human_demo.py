import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mujoco

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURR_DIR)
XML_PATH = os.path.join(PROJECT_ROOT, "mujoco_sim", "assets", "iris.xml")

NUM_JOINTS = 6


# ------------------------------------------------------------
# IRIS Forward Kinematics (your verified analytical FK)
# ------------------------------------------------------------
class IRISKinematics:
    def __init__(self):
        self.link_configs = [
            {'pos': [0, 0, 0.2487],         'euler': [0, 0, 0],    'axis': [0, 0, 1]}, 
            {'pos': [0.0218, 0, 0.059],     'euler': [0, 90, 180], 'axis': [0, 0, 1]}, 
            {'pos': [0.299774, 0, -0.0218], 'euler': [0, 0, 0],    'axis': [0, 0, 1]}, 
            {'pos': [0.02, 0, 0],           'euler': [0, 90, 0],   'axis': [0, 0, 1]}, 
            {'pos': [0, 0, 0.315],          'euler': [0, -90, 0],  'axis': [0, 0, 1]}, 
            {'pos': [0.042824, 0, 0],       'euler': [0, 90, 0],   'axis': [0, 0, 1]}  
        ]

    def get_local_transform(self, config, q_rad):
        T = np.eye(4)
        T[:3, 3] = config['pos']

        # fixed body rotation
        R_fixed = np.eye(3)
        if any(config['euler']):
            quat_e = np.zeros(4)
            mujoco.mju_euler2Quat(quat_e, np.deg2rad(config['euler']), 'xyz')
            Rm = np.zeros(9)
            mujoco.mju_quat2Mat(Rm, quat_e)
            R_fixed = Rm.reshape(3, 3)

        # joint rotation
        quat_j = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat_j, np.array(config['axis']), q_rad)
        Rj = np.zeros(9)
        mujoco.mju_quat2Mat(Rj, quat_j)
        R_joint = Rj.reshape(3, 3)

        T[:3, :3] = R_fixed @ R_joint
        return T

    def end_effector_position(self, q_deg):
        q_rad = np.deg2rad(q_deg)
        T_accum = np.eye(4)
        for i in range(len(self.link_configs)):
            T_local = self.get_local_transform(self.link_configs[i], q_rad[i])
            T_accum = T_accum @ T_local
        return T_accum[:3, 3].copy()


# ------------------------------------------------------------
# Load episodes
# ------------------------------------------------------------
def list_episode_dirs(data_root, prefix):
    return sorted([
        os.path.join(data_root, d)
        for d in os.listdir(data_root)
        if d.startswith(prefix + "_episode_")
    ])


def load_joint_csv(ep_dir):
    csv_path = os.path.join(ep_dir, "robot", "joint_states.csv")
    df = pd.read_csv(csv_path)
    pos_cols = [c for c in df.columns if c.startswith("pos_")]
    pos_cols = pos_cols[:NUM_JOINTS]
    q = df[pos_cols].to_numpy()
    return q


# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------
def visualize_episodes(data_root, bag_prefix, max_eps=20):
    kin = IRISKinematics()

    ep_dirs = list_episode_dirs(data_root, bag_prefix)
    if len(ep_dirs) == 0:
        raise RuntimeError("No episodes found")

    print(f"Found {len(ep_dirs)} episodes")

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.viridis(np.linspace(0,1,min(len(ep_dirs), max_eps)))

    for idx, ep in enumerate(ep_dirs[:max_eps]):
        q = load_joint_csv(ep)
        ee_traj = np.zeros((q.shape[0],3))

        for t in range(q.shape[0]):
            ee_traj[t] = kin.end_effector_position(q[t])

        ax.plot(
            ee_traj[:,0],
            ee_traj[:,1],
            ee_traj[:,2],
            color=colors[idx],
            linewidth=1.5,
            alpha=0.9
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Human Demonstration End-Effector Trajectories\n{bag_prefix}")

    ax.view_init(elev=25, azim=45)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True,
                        help="Path to processed_data directory")
    parser.add_argument("--bag_prefix", required=True,
                        help="Episode prefix, e.g. 0.3_0.3_0.3_goal_20260111_181031")
    parser.add_argument("--max_eps", type=int, default=20)
    args = parser.parse_args()

    visualize_episodes(args.data_root, args.bag_prefix, args.max_eps)
