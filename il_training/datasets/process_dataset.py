import os, cv2, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset

from losses.loss import IRISKinematics   # your FK is inside loss.py

NUM_JOINTS = 6
fk = IRISKinematics()

# =========================================================
# Configuration (global dataset hyperparameters)
# =========================================================
SEQ_LEN = 5
FUTURE_STEPS = 8


# =========================================================
# Episode Discovery on SSD
# =========================================================

def list_episode_dirs(root, prefix):
    return sorted([
        os.path.join(root, d)
        for d in os.listdir(root)
        if d.startswith(prefix + "_episode_")
    ])


# =========================================================
# Load Single Episode
# =========================================================

def load_episode(ep_dir):
    rgb_dir = os.path.join(ep_dir, "rgb")
    robot_csv = os.path.join(ep_dir, "robot", "joint_states.csv")

    if not os.path.isfile(robot_csv):
        return None

    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png")])

    df = pd.read_csv(robot_csv)
    pos_cols = [c for c in df.columns if c.startswith("pos_")][:NUM_JOINTS]
    joints = df[pos_cols].to_numpy(dtype=np.float32)

    # Align length
    T = min(len(rgb_files), joints.shape[0])
    rgb_files = rgb_files[:T]
    joints = joints[:T]

    # -----------------------------------------------------
    # Goal definition = FK of final joint configuration
    # -----------------------------------------------------
    goal_joint = joints[-1].copy()
    goal_xyz = fk.forward(np.rad2deg(goal_joint))

    return {
        "rgb_dir": rgb_dir,
        "rgb_files": rgb_files,
        "joints": joints,
        "goal_xyz": goal_xyz
    }


# =========================================================
# Episode Split (80 / 10 / 10)
# =========================================================

def split_episodes(episodes, save_manifest_path=None):
    N = len(episodes)
    n_train = int(0.8 * N)
    n_val   = int(0.1 * N)

    train_eps = episodes[:n_train]
    val_eps   = episodes[n_train:n_train+n_val]
    test_eps  = episodes[n_train+n_val:]

    # Optional: save split manifest for reproducibility
    if save_manifest_path is not None:
        with open(save_manifest_path, "w") as f:
            for split, eps in zip(["train","val","test"], [train_eps,val_eps,test_eps]):
                for e in eps:
                    f.write(f"{split},{e['rgb_dir']}\n")

    return train_eps, val_eps, test_eps


# =========================================================
# Sliding Window Dataset
# =========================================================

class EpisodeWindowDataset(Dataset):
    def __init__(self, episodes, seq_len=SEQ_LEN, future_steps=FUTURE_STEPS):
        self.episodes = episodes
        self.seq_len = seq_len
        self.future_steps = future_steps

        # Build (episode_index, start_index) pairs
        self.samples = []
        for ei, ep in enumerate(self.episodes):
            T = ep["joints"].shape[0]
            max_start = T - (seq_len + future_steps) + 1
            for s in range(max_start):
                self.samples.append((ei, s))

        print(f"Built {len(self.samples)} windows from {len(self.episodes)} episodes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ei, s = self.samples[idx]
        ep = self.episodes[ei]

        # -------- RGB sequence --------
        rgb_seq = []
        for i in range(self.seq_len):
            img_path = os.path.join(ep["rgb_dir"], ep["rgb_files"][s+i])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128,128))
            img = torch.tensor(img).permute(2,0,1).float() / 255.0
            rgb_seq.append(img)

        rgb_seq = torch.stack(rgb_seq)   # (S,3,128,128)

        # -------- Joint history --------
        joint_seq = torch.tensor(
            ep["joints"][s:s+self.seq_len],
            dtype=torch.float32
        )                                 # (S,6)

        # -------- Future Δq target --------
        q_last = ep["joints"][s+self.seq_len-1]
        fut = ep["joints"][s+self.seq_len : s+self.seq_len+self.future_steps]
        fut_delta = fut - q_last[None,:]
        future = torch.tensor(fut_delta, dtype=torch.float32)  # (F,6)

        # -------- Goal Cartesian --------
        goal_xyz = torch.tensor(ep["goal_xyz"], dtype=torch.float32)  # (3,)

        return rgb_seq, joint_seq, goal_xyz, future


# =========================================================
# Convenience Loader (One-call entry point)
# =========================================================

def build_datasets_from_ssd(ssd_root, bag_prefix):
    dirs = list_episode_dirs(ssd_root, bag_prefix)
    episodes = [load_episode(d) for d in dirs]
    episodes = [e for e in episodes if e is not None]

    if len(episodes) == 0:
        raise RuntimeError("No valid episodes found on SSD!")

    train_eps, val_eps, test_eps = split_episodes(
        episodes,
        save_manifest_path="configs/dataset_split_manifest.txt"
    )

    train_ds = EpisodeWindowDataset(train_eps)
    val_ds   = EpisodeWindowDataset(val_eps)
    test_ds  = EpisodeWindowDataset(test_eps)

    return train_ds, val_ds, test_ds
