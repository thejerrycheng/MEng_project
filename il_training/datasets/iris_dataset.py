import os, cv2, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from kinematics import IRISKinematics

NUM_JOINTS = 6
fk = IRISKinematics()

def list_episode_dirs(root, prefix):
    return sorted([
        os.path.join(root,d) for d in os.listdir(root)
        if d.startswith(prefix+"_episode_")
    ])

def load_episode(ep_dir):
    rgb_dir = os.path.join(ep_dir,"rgb")
    robot_csv = os.path.join(ep_dir,"robot","joint_states.csv")
    if not os.path.isfile(robot_csv):
        return None

    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png")])
    df = pd.read_csv(robot_csv)
    pos_cols = [c for c in df.columns if c.startswith("pos_")][:NUM_JOINTS]
    joints = df[pos_cols].to_numpy(dtype=np.float32)

    T = min(len(rgb_files), joints.shape[0])
    rgb_files, joints = rgb_files[:T], joints[:T]

    goal_joint = joints[-1].copy()
    goal_xyz   = fk.forward(np.rad2deg(goal_joint))

    return dict(rgb_dir=rgb_dir,
                rgb_files=rgb_files,
                joints=joints,
                goal_joint=goal_joint,
                goal_xyz=goal_xyz)

class EpisodeWindowDataset(Dataset):
    def __init__(self, episodes, seq_len, future_steps):
        self.episodes=episodes
        self.seq_len=seq_len
        self.future_steps=future_steps
        self.samples=[]
        for ei,ep in enumerate(episodes):
            T=ep["joints"].shape[0]
            for s in range(T-(seq_len+future_steps)+1):
                self.samples.append((ei,s))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        ei,s = self.samples[idx]
        ep=self.episodes[ei]

        rgb_seq=[]
        joint_seq=[]
        for i in range(self.seq_len):
            img=cv2.imread(os.path.join(ep["rgb_dir"],ep["rgb_files"][s+i]))
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img=torch.tensor(img).permute(2,0,1).float()/255.0
            rgb_seq.append(img)
            joint_seq.append(ep["joints"][s+i])

        rgb_seq=torch.stack(rgb_seq)
        joint_seq=torch.tensor(joint_seq,dtype=torch.float32)

        q_last=ep["joints"][s+self.seq_len-1]
        fut=ep["joints"][s+self.seq_len:s+self.seq_len+self.future_steps]
        fut=torch.tensor(fut-q_last[None,:],dtype=torch.float32)

        goal_xyz=torch.tensor(ep["goal_xyz"],dtype=torch.float32)

        return rgb_seq, joint_seq, goal_xyz, fut
