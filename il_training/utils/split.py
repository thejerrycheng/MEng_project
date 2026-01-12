def split_episodes(episodes, train_ratio=0.8, val_ratio=0.1):
    N = len(episodes)
    n_train = int(N * train_ratio)
    n_val   = int(N * val_ratio)
    train_eps = episodes[:n_train]
    val_eps   = episodes[n_train:n_train+n_val]
    test_eps  = episodes[n_train+n_val:]
    return train_eps, val_eps, test_eps
