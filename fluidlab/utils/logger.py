import os
from PIL import Image
import cv2
import h5py 
import numpy as np
import pickle as pkl
from time import time
from torch.utils.tensorboard import SummaryWriter as TorchWriter
from fluidlab.utils.misc import get_src_dir
class SummaryWriter:
    def __init__(self, exp_name):
        self.dir = os.path.join(get_src_dir(), '..', 'logs', 'logs', exp_name)
        os.makedirs(self.dir, exist_ok=True)
        self.writer = TorchWriter(log_dir=self.dir)

    def write(self, iteration, info):
        for key in info:
            self.writer.add_scalar(key, info[key], iteration)

class ImageWriter:
    def __init__(self, exp_name):
        self.dir = os.path.join(get_src_dir(), '..', 'logs', 'imgs', exp_name)

    def write(self, img, iteration, step):
        img_dir = os.path.join(self.dir, f'{iteration}')
        img_path = os.path.join(self.dir, f'{iteration}/{step:04d}.png')
        os.makedirs(img_dir, exist_ok=True)
        cv2.imwrite(img_path, img[:, :, ::-1])
class TrajectoryWriter:
    def __init__(self, exp_name, output_ds):
        self.trajs_fname = output_ds
        self.exp_name = exp_name
        self.dir_name = os.path.join(get_src_dir(), '..', self.trajs_fname)
    def write(self, action, sim_state, img_obs, iteration: int, t: int):
        f = h5py.File(self.dir_name, "a", driver="family")
        g = f.require_group(self.exp_name)
        traj = g.require_group("traj" + str(iteration))
        traj.pop("t_{t:04d}", None)
        tstep = traj.require_group(f"t_{t:04d}")
        tstep.pop("sim_state", None)
        tstep.pop("img_obs", None)
        tstep.pop("action", None) 
        sim_state_g = tstep.require_group("sim_state")
        sim_state_g.create_dataset("x", data=sim_state["x"], dtype='float32', compression="gzip", chunks=True, compression_opts=9)
        sim_state_g.create_dataset("v", data=sim_state["v"], dtype='float32', compression="gzip", chunks=True, compression_opts=9)
        tstep.create_dataset("img_obs", data=img_obs, dtype="float32", compression="gzip", chunks=True, compression_opts=9)
        action = action.astype('float32') if action is not None else []
        tstep.create_dataset("action", data=action, dtype="float32", compression="gzip", chunks=True, compression_opts=9)
        f.close()
    def print_trajs():
        with h5py.File(TrajectoryWriter.dir_name, "r") as f:
            def p(x):
                num_indents = x.count("/")
                print('\t' * num_indents + x)
            f.visit(p)
class Logger:
    def __init__(self, exp_name, output_ds):
        self.exp_name = exp_name
        self.summary_writer = SummaryWriter(exp_name)
        self.image_writer = ImageWriter(exp_name)
        self.traj_writer = TrajectoryWriter(exp_name, output_ds)
        self.last_step_t = time()
    
    def resize_img(self, img):
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((256, 256), resample=Image.Resampling.LANCZOS) 
        return np.array(pil_img)
    def write_img(self, img, iteration, step):
        img = self.resize_img(img)
        self.image_writer.write(img, iteration, step)
    def write_traj(self, action, sim_state, img_obs, iteration: int, t: int):
        img = self.resize_img(img_obs)
        self.traj_writer.write(action, sim_state, img, iteration, t)
    def save_policy(self, policy, iteration):
        policy_dir = os.path.join(get_src_dir(), '..', 'logs', 'policies', self.exp_name)
        os.makedirs(policy_dir, exist_ok=True)
        pkl.dump(policy, open(os.path.join(policy_dir, f'{iteration:04d}.pkl'), 'wb'))

    def log(self, iteration, info):
        cur_t = time()
        print_msg = f'Iteration: {iteration}, '
        tb_info = dict()
        for key in info:
            val = info[key]
            if type(val) is int:
                print_msg += f'{key}: {info[key]}, '
                tb_info[key] = info[key]
            elif type(val) is float or type(val) is np.float32:
                print_msg += f'{key}: {info[key]:.3f}, '
                tb_info[key] = info[key]
            else:
                pass

        print_msg += f'Step time: {cur_t-self.last_step_t:.2f}s'
        print(print_msg)

        self.summary_writer.write(iteration, tb_info)
        self.last_step_t = cur_t

if __name__ == "__main__":
    # Create a test TrajectoryWriter
    writer = TrajectoryWriter("latteart")
    # writer.write([1], [2], 1)
    TrajectoryWriter.print_trajs()
