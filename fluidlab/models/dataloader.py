import numpy as np
import torch
import os
import h5py
from torch.utils.data import Dataset, DataLoader

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
                path = os.path.join("fluidlab/", "..", trajs_dir, f_name)
                file = h5py.File(path, driver="family")
            except:
                try:
                    start = trajs[i] + 1
                    f_name = f"trajs{start:04}_{end:04}_%d.hdf5"
                    path = os.path.join("fluidlab/", "..", trajs_dir, f_name)
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
        try:
            trajs = f[EXP_NAME]
        except ValueError as e:
            print(f)
            raise e
        traj_idxs = list(trajs.keys())
        traj_idx = idx % N_TRAJS_PER_FILE
        try:
            traj = trajs[traj_idxs[traj_idx]]
        except ValueError as e:
            print(idx)
            raise e
        img_obs_matrix = []
        action_matrix = []
        sim_state_matrix = []
        # pick a random timestep
        tstep = np.random.choice(list(traj.keys()))
        img_obs_matrix = traj[tstep]["img_obs"][:]
        img_obs_matrix = np.transpose(img_obs_matrix, (2, 0, 1))
        img_obs_matrix = torch.Tensor(img_obs_matrix)
        action_matrix = torch.Tensor(np.array(action_matrix))
        sim_state_matrix = torch.Tensor(np.array(sim_state_matrix))
        return img_obs_matrix, action_matrix, sim_state_matrix


if __name__ == "__main__":
    ds = TrajectoryDataset("data")
    train_dataloader = DataLoader(ds, batch_size=32, num_workers=2)
    img_obs_batch, action_batch, sim_state_batch = next(iter(train_dataloader))
    # print(action_batch.size())
    for f in ds.traj_paths:
        f.close()
