import numpy as np
import torch
import os
import h5py
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from fluidlab.utils.misc import get_src_dir

MAX_N_TRAJS = 2500
N_TRAJS_PER_FILE = 500
EXP_NAME = "exp_latteart"
AC_DIM = 3


class TrajectoryDataset(Dataset):
    def __init__(self, trajs_dir):
        trajs = torch.linspace(
            0, MAX_N_TRAJS, MAX_N_TRAJS // N_TRAJS_PER_FILE + 1, dtype=int
        )
        self.traj_paths = []
        for i in range(len(trajs) - 1):
            end = trajs[i + 1]
            try:
                start = trajs[i]
                f_name = f"trajs{start:04}_{end:04}_%d.hdf5"
                path = os.path.join(get_src_dir(), "..", trajs_dir, f_name)
                file = h5py.File(path, driver="family")
            except:
                try:
                    start = trajs[i] + 1
                    f_name = f"trajs{start:04}_{end:04}_%d.hdf5"
                    path = os.path.join(get_src_dir(), "..", trajs_dir, f_name)
                    file = h5py.File(path, driver="family")
                except FileNotFoundError as f_err:
                    print(f_err)
                    raise FileNotFoundError
            self.traj_paths.append(file)

    def __len__(self):
        return MAX_N_TRAJS

    def __getitem__(self, idx):
        # first find which traj file this traj would be found in
        adj_idx = idx // N_TRAJS_PER_FILE
        f = self.traj_paths[adj_idx]

        # second get the specific trajectory within that file
        trajs = f[EXP_NAME]
        traj_idxs = list(trajs.keys())
        traj_idx = idx % N_TRAJS_PER_FILE
        try:
            traj = trajs[traj_idxs[traj_idx]]
        except ValueError as e:
            print(idx)
            raise ec
        img_obs_matrix = []
        action_matrix = []
        sim_state_matrix = []
        def fn(name, obj):
            if type(obj) != h5py.Group or "sim_state" in name:
                return
            if len(obj["action"]) == 0:
                action_matrix.append([0] * AC_DIM)
            else:
                action_matrix.append(obj["action"][:])
            img_obs_matrix.append(obj["img_obs"][:])
            # sim_state_matrix.append([obj["sim_state"]['x'][:], obj["sim_state"]["v"][:]])

        traj.visititems(fn)
        f.close()
        img_obs_matrix = torch.Tensor(np.array(img_obs_matrix))
        action_matrix = torch.Tensor(np.array(action_matrix))
        # sim_state_matrix = torch.Tensor(np.array(sim_state_matrix))
        return img_obs_matrix, action_matrix, sim_state_matrix


if __name__ == "__main__":
    ds = TrajectoryDataset("data")
    train_dataloader = DataLoader(ds, batch_size=64, shuffle=True)
    img_obs_batch, action_batch, sim_state_batch = next(iter(train_dataloader))
    for f in ds.traj_paths:
        f.close()