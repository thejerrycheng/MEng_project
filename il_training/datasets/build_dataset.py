import os, argparse, pickle
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from losses.loss import IRISKinematics
from process_dataset import (
    list_episode_dirs,
    load_episode,
    split_episodes,
    EpisodeWindowDataset,
    SEQ_LEN,
    FUTURE_STEPS
)

# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssd_root", required=True, help="Root SSD processed_data directory")
    parser.add_argument("--prefix", required=True, help="Bag prefix for episodes")
    parser.add_argument("--out_dir", default="new_processed_dataset")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Scanning SSD for episodes...")
    dirs = list_episode_dirs(args.ssd_root, args.prefix)
    episodes = [load_episode(d) for d in dirs]
    episodes = [e for e in episodes if e is not None]

    if len(episodes) == 0:
        raise RuntimeError("No valid episodes found.")

    print(f"Loaded {len(episodes)} episodes")

    # ------------------------------------------------------------
    # Split episodes
    train_eps, val_eps, test_eps = split_episodes(
        episodes,
        save_manifest_path=os.path.join(args.out_dir, "dataset_manifest.txt")
    )

    print(f"Split: {len(train_eps)} train | {len(val_eps)} val | {len(test_eps)} test")

    # ------------------------------------------------------------
    # Build windowed datasets
    print("Building sliding windows...")
    train_ds = EpisodeWindowDataset(train_eps, SEQ_LEN, FUTURE_STEPS)
    val_ds   = EpisodeWindowDataset(val_eps, SEQ_LEN, FUTURE_STEPS)
    test_ds  = EpisodeWindowDataset(test_eps, SEQ_LEN, FUTURE_STEPS)

    # ------------------------------------------------------------
    # Save lightweight pickles
    with open(os.path.join(args.out_dir, "train.pkl"), "wb") as f:
        pickle.dump(train_ds, f)

    with open(os.path.join(args.out_dir, "val.pkl"), "wb") as f:
        pickle.dump(val_ds, f)

    with open(os.path.join(args.out_dir, "test.pkl"), "wb") as f:
        pickle.dump(test_ds, f)

    print("====================================")
    print(" Dataset build complete")
    print(" Saved to:", args.out_dir)
    print(" seq_len =", SEQ_LEN)
    print(" future_steps =", FUTURE_STEPS)
    print("====================================")


if __name__ == "__main__":
    main()
