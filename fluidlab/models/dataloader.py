import numpy as np
import torch
import os
import h5py
from torch.utils.data import Dataset, DataLoader

MAX_N_TRAJS = 2500
N_TRAJS_PER_FILE = 500
N_TSTEPS_PER_TRAJ = 330
EXP_NAME = "exp_latteart"
AC_DIM = 3


class TrajectoryDataset(Dataset):
    def __init__(self, trajs_dir, train=True):
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
        return N_TSTEPS_PER_TRAJ * MAX_N_TRAJS

    def __getitem__(self, idx):
        # first get which traj this tstep idx would correspond to
        traj = idx // N_TSTEPS_PER_TRAJ
        # first find which traj file this traj would be found in
        adj_idx = traj // N_TSTEPS_PER_TRAJ
        f = self.traj_paths[adj_idx]

        # second get the specific trajectory within that file
        try:
            trajs = f[EXP_NAME]
        except ValueError as e:
            print(f)
            raise e
        traj_idxs = list(trajs.keys())
        traj_idx = traj % N_TRAJS_PER_FILE
        try:
            tstep_key = traj_idxs[traj_idx]
            traj = trajs[tstep_key]
        except ValueError as e:
            print(idx, traj)
            raise e
        sim_state_matrix = []
        # pick a random timestep
        tstep_keys = list(traj.keys())
        # random_idx = np.random.choice(range(len(tstep_keys) - 1))
        # tstep = tstep_keys[random_idx]
        tstep = tstep_keys[idx % N_TSTEPS_PER_TRAJ]
        # tstep_next = tstep_keys[random_idx + 1]
        if (idx % N_TSTEPS_PER_TRAJ) + 1 == len(tstep_keys):
            tstep_next = tstep
        else:
            tstep_next = tstep_keys[(idx % N_TSTEPS_PER_TRAJ) + 1]
        img_obs = traj[tstep]["img_obs"][:]
        img_obs_next = traj[tstep_next]["img_obs"][:]
        img_obs = np.transpose(img_obs, (2, 0, 1))
        img_obs = torch.Tensor(img_obs)
        img_obs_next = np.transpose(img_obs_next, (2, 0, 1))
        img_obs_next = torch.Tensor(img_obs)
        action = traj[tstep]["action"][:]
        if len(action) == 0:
            action = np.array([0] * 3)
        action = torch.Tensor(action)
        sim_state_matrix = torch.Tensor(np.array(sim_state_matrix))
        return img_obs, img_obs_next, action, sim_state_matrix


class NumPyTrajectoryDataset(Dataset):
    def __init__(self, trajs_dir, train=True) -> None:
        self.trajs_dir = trajs_dir
        self.offset = 0
        if not train:
            self.offset = 1250
        self.prefix = "traj_"

    def __len__(self):
        return (MAX_N_TRAJS * N_TSTEPS_PER_TRAJ) // 2

    def __getitem__(self, idx):
        traj = idx // N_TSTEPS_PER_TRAJ
        traj += self.offset
        adj_idx = idx % N_TSTEPS_PER_TRAJ
        path = os.path.join("fluidlab/", "..", self.trajs_dir)
        f = np.load(f"{path}/{self.prefix}{traj:04d}.npz", mmap_mode="r")
        img_obs = f["img_obs"][adj_idx]
        next_img_obs = f["img_obs"][adj_idx]
        img_obs = np.transpose(img_obs, (2, 0, 1))
        next_img_obs = np.transpose(next_img_obs, (2, 0, 1))
        actions = f["actions"][adj_idx]
        f.close()
        return img_obs, next_img_obs, actions, []


def time_dataloading(batch_size):
    import time

    train = NumPyTrajectoryDataset("trajs", train=True)
    a = time.time()
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=0
    )
    i = 0
    for batch_idx, (img_obs, next_img_obs, actions, _) in enumerate(train_loader):
        print(img_obs.size())
        print(next_img_obs.size())
        print(actions.size())
        if i == 5:
            break
        i += 1
    print(
        f"Dataloading with batch size = {batch_size} took {time.time() - a:6f} seconds"
    )


if __name__ == "__main__":
    # ds = NumPyTrajectoryDataset("trajs")
    # train_dataloader = DataLoader(ds, batch_size=2, num_workers=1)
    # img_obs_batch, action_batch, sim_state_batch = next(iter(train_dataloader))
    time_dataloading(batch_size=32)
    # for f in ds.traj_paths:
    #     f.close()
