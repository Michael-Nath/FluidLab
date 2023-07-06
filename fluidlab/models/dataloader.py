import numpy as np
import time
import torch
import os
import h5py
from torchvision.utils import save_image
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader, SequentialSampler, BatchSampler
import ffmpegio
MAX_N_TRAJS = 2500
N_TRAJS_PER_FILE = 500
N_TSTEPS_PER_TRAJ = 250
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
    def __init__(self, trajs_dir, lookahead_amnt=1, train=True, amnt_trajs_test=500) -> None:
        self.max_lookahead_amnt = lookahead_amnt
        self.trajs_dir = trajs_dir
        self.cache = dict()
        self.offset = 0
        self.train = train
        if not train:
            self.offset = MAX_N_TRAJS - amnt_trajs_test
        self.prefix = "traj_"
        self.amnt_trajs_test = amnt_trajs_test

    def __len__(self):
        if self.train:
            return (MAX_N_TRAJS - self.amnt_trajs_test) * N_TSTEPS_PER_TRAJ
        else:
            return self.amnt_trajs_test * N_TSTEPS_PER_TRAJ

    def __getitem__(self, idx):
        traj = idx // N_TSTEPS_PER_TRAJ
        traj += self.offset
        adj_idx = idx % N_TSTEPS_PER_TRAJ
        path = os.path.join("fluidlab/", "..", self.trajs_dir)
        f_prefix = f"{path}/{self.prefix}{traj:04d}" 
        if f_prefix not in self.cache.keys():
            f_o = np.load(f"{path}/{self.prefix}{traj:04d}_o.npy", mmap_mode="r")
            f_a = np.load(f"{path}/{self.prefix}{traj:04d}_a.npy", mmap_mode="r")
            # self.cache[f_prefix] = (f_o, f_a)
        else:
            f_o, f_a = self.cache[f_prefix]
        img_obs = f_o[adj_idx].copy()
        actions = f_a[adj_idx].copy()
        allowable_lookahead_amnt = min(N_TSTEPS_PER_TRAJ - adj_idx - 1, self.max_lookahead_amnt)
        if allowable_lookahead_amnt == 0:
            sampled_lookahead_amnt = 0
        else:
            sampled_lookahead_amnt = np.random.randint(1, allowable_lookahead_amnt + 1)
        goal_img_obs = f_o[adj_idx + sampled_lookahead_amnt].copy()
        return img_obs, goal_img_obs, actions, []
        return ToTensor()(img_obs), ToTensor()(goal_img_obs), actions, []        

def time_dataloading(batch_size):
    train = NumPyTrajectoryDataset("npy_trajs", train=True)
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=4
    )
    i = 0
    a = time.time()
    for batch_idx, (img_obs, next_img_obs, actions, _) in enumerate(train_loader):
        print(
            f"Dataloading with batch size = {batch_size} took {time.time() - a:6f} seconds"
        )
        if i == 10:
            break
        i += 1
        a = time.time()

def get_mean_img():
    train = NumPyTrajectoryDataset("low_var_trajs", lookahead_amnt=1, train=True)
    mean_imgs = []
    train_loader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True, num_workers=2)
    for batch_idx, (img_obs, _, _, _,) in enumerate(train_loader):
        img_obs = torch.Tensor(img_obs)
        mean_img = img_obs.mean(dim=0)
        mean_imgs.append(mean_img)
        if batch_idx == 16:
            break
    save_image(mean_imgs, "latteart-recon/mean_imgs/mean_imgs.png", nrow=4)

def viz_trajs():
    ds = NumPyTrajectoryDataset("low_var_trajs", lookahead_amnt=1, train=True, amnt_trajs_test=0)
    max_tsteps = len(ds)
    random_starts = np.random.choice(range(2500), 5, replace=False)
    random_starts *= 250
    for start in random_starts:
        a = []
        for i in range(250):
            img, _, _, _ = ds[start + i]
            a.append(img)
        a = np.array(a)
        ffmpegio.video.write(f"fluidlab/models/viz/movie_{start//250}.mp4", 60, a, show_log=True,overwrite=True)
        print(f"Stored trajectory {start//250} at fluidlab/models/viz/movie_{start//250}.mp4!")
    


if __name__ == "__main__":
    # ds = NumPyTrajectoryDataset("trajs")
    # train_dataloader = DataLoader(ds, batch_size=2, num_workers=1)
    # img_obs_batch, action_batch, sim_state_batch = next(iter(train_dataloader))

    # time_dataloading(batch_size=256)
    # get_mean_img()
    # for f in ds.traj_paths:
    #     f.close()
    viz_trajs()
