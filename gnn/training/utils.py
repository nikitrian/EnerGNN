# gnn/training/utils.py
import random
import torch
import numpy as np
import os

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_device_tensor(x, device):
    return x.to(device) if isinstance(x, torch.Tensor) else x

def to_device_graph(container, device):
    if isinstance(container, dict):
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in container.items()}
    else:
        return container.to(device)

def save_checkpoint(model, opt, path, epoch, best=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "best": best,
    }
    torch.save(ckpt, path)

def load_checkpoint(model, opt, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if opt is not None:
        opt.load_state_dict(ckpt["opt_state"])
    return ckpt

def split_instance_ids(instance_ids, val_frac=0.1, test_frac=0.1, seed=42):
    """
    Randomly split instance IDs into train/val/test.
    """
    rng = random.Random(seed)
    ids = list(instance_ids)
    rng.shuffle(ids)

    n_total = len(ids)
    n_val = int(n_total * val_frac)
    n_test = int(n_total * test_frac)

    val_ids = ids[:n_val]
    test_ids = ids[n_val:n_val+n_test]
    train_ids = ids[n_val+n_test:]

    return train_ids, val_ids, test_ids

