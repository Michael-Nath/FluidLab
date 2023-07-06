# import os
# import cv2
import wandb
from datetime import datetime
import numpy as np
import itertools

import taichi as ti
from fluidlab.utils.misc import is_on_server
from h5py import File
import torch
import torch.optim as optim

# from fluidlab.fluidengine.taichi_env import TaichiEnv
from fluidlab.models.dataloader import NumPyTrajectoryDataset
from fluidlab.models.gc_bc import GCBCAgent, GCBCVAEAgent

from PIL import Image

class Solver:
    def __init__(self, env, logger=None, cfg=None, agent_type="gcbc", in_vae_weights=None):
        self.cfg = cfg
        self.env = env
        self.target_file = env.target_file
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if agent_type == "gcbc":
            self.agent = GCBCAgent(3)
        elif agent_type == "gcbc_vae":
            self.agent = GCBCVAEAgent(3, in_vae_weights)
        self.agent = self.agent.to(self.device)
        # self.agent = None

    def create_trajs(self, iteration,beta,test=False):
        taichi_env = self.env.taichi_env
        horizon = self.env.horizon
        policy = self.env.random_policy(self.cfg.init_range, beta)
        horizon_action = self.env.horizon_action
        init_state = taichi_env.get_state()
        taichi_env.set_state(**init_state)
        taichi_env.apply_agent_action_p(policy.get_action())
        action_matrix = []
        sim_state_matrix = []
        img_obs_matrix = []
        for i in range(horizon):
            if i < horizon_action:
                action = policy.get_action(i=i, agent=taichi_env.agent, update=True)
            else:
                break
            sim_state = self.env.taichi_env.get_state_RL()
            img = taichi_env.render("rgb_array")
            taichi_env.step(action)
            action = action if action is not None else [0] * 3
            if test:
                self.logger.write_traj(action, sim_state, img, iteration, i)
            if not test:
                sim_state_matrix.append(sim_state)
                action_matrix.append(action if action is not None else [0] * 3)
                img_obs_matrix.append(self.logger.resize_img(img))
            self.logger.write_img(img, iteration, i)
        if not test:
            np.save(
                f"{self.logger.traj_writer.trajs_fname}/traj_{iteration:04d}_a.npy",
                action_matrix,
            )
            np.save(
                f"{self.logger.traj_writer.trajs_fname}/traj_{iteration:04d}_o.npy",
                img_obs_matrix,
            )
            return
        else:
            return

    def train_bc(self, epoch, in_trajs_file, lookahead_amnt):
        self.agent.train()
        train = NumPyTrajectoryDataset(
            in_trajs_file, train=True, lookahead_amnt=lookahead_amnt
        )
        print(f"\nEpoch: {epoch+1:d} {datetime.now()}")
        train_loader = torch.utils.data.DataLoader(
            train, batch_size=64, shuffle=True, num_workers=2
        )
        gcbc_optim = optim.Adam(
            self.agent.parameters()
        )
        for batch_idx, (img_obs, goal_img_obs, action, _) in enumerate(train_loader):
            gcbc_optim.zero_grad()
            dist = self.agent.forward(img_obs, goal_img_obs)
            action = action.detach().to(self.device)
            log_probs = dist.log_prob(action)
            loss = -torch.sum(log_probs, dim=1).mean().to(self.device)
            wandb.log({"train_loss": loss})
            wandb.log({"log_prob": -loss})
            loss.backward()
            gcbc_optim.step()
            if batch_idx == 250:
                break

    def eval_bc(self, epoch, trajs_file, lookahead_amnt):
        self.agent.eval()
        taichi_env = self.env.taichi_env
        f = File(trajs_file, driver="family")
        traj_keys = np.array(list(f["exp_latteart"].keys()))
        sampled_trajs = np.random.choice(traj_keys, size=100)
        val_loss = 0
        cnt = 0
        for traj in sampled_trajs:
            tsteps = f["exp_latteart"][traj]
            tsteps_keys = np.array(list(tsteps.keys()))
            tsteps_keys_adj = tsteps_keys[:-lookahead_amnt]
            sampled_tsteps_idxs = np.random.choice(len(tsteps_keys_adj), 10)
            traj_loss = 0
            for i in sampled_tsteps_idxs:
                tstep_key = tsteps_keys_adj[i]
                tstep = tsteps[tstep_key]
                loaded_sim_state = dict(tstep["sim_state"])
                taichi_env_state = taichi_env.get_state()
                # Preparing the new sim state
                loaded_sim_state["x"] = loaded_sim_state["x"][:]
                loaded_sim_state["v"] = loaded_sim_state["v"][:]
                loaded_sim_state["used"] = loaded_sim_state["used"][:].astype("int32")
                loaded_sim_state["F"] = taichi_env_state["state"]["F"]
                loaded_sim_state["C"] = taichi_env_state["state"]["C"]
                loaded_sim_state["agent"] = taichi_env_state["state"]["agent"]
                taichi_env.set_state(loaded_sim_state)
                # Preparing input for BC policy
                next_tstep = tsteps[tsteps_keys[i + lookahead_amnt]]
                cur_img_obs = taichi_env.render("rgb_array")
                cur_img_obs = torch.Tensor(self.logger.resize_img(cur_img_obs))
                goal_img_obs = torch.Tensor(next_tstep["img_obs"][:])
                cur_img_obs = cur_img_obs.movedim(2, 0)
                goal_img_obs = goal_img_obs.movedim(2, 0)
                with torch.no_grad():
                    # inpt = torch.cat((cur_img_obs, goal_img_obs), 0).to(self.device)
                    # inpt = inpt.type("torch.cuda.FloatTensor")
                    # inpt = inpt.unsqueeze(0)
                    pred_a = self.agent.forward(cur_img_obs, goal_img_obs).sample()
                actual_a = tstep["action"][:]
                pred_a = pred_a[0].detach().cpu()
                loss = self.env.get_loss(pred_a, actual_a)
                val_loss += loss
                traj_loss += loss
                cnt += 1
            print(f"Epoch {epoch + 1} | End of Trajectory Val Loss: {traj_loss}")
        wandb.log({"End of Epoch Val Loss": val_loss / cnt})
        f.close()

    def solve(self):
        taichi_env = self.env.taichi_env
        policy = self.env.trainable_policy(self.cfg.optim, self.cfg.init_range)

        taichi_env_state = taichi_env.get_state()

        def forward_backward(sim_state, policy, horizon, horizon_action):
            taichi_env.set_state(sim_state, grad_enabled=True)

            # forward pass
            from time import time

            t1 = time()
            taichi_env.apply_agent_action_p(policy.get_actions_p())
            cur_horizon = taichi_env.loss.temporal_range[1]
            for i in range(cur_horizon):
                if i < horizon_action:
                    action = policy.get_action_v(i, agent=taichi_env.agent, update=True)
                else:
                    action = None
                taichi_env.step(action)

                # print(i, taichi_env.get_step_loss())
                # self.env._get_obs()

            loss_info = taichi_env.get_final_loss()
            t2 = time()

            # backward pass
            taichi_env.reset_grad()
            taichi_env.get_final_loss_grad()

            for i in range(cur_horizon - 1, policy.freeze_till - 1, -1):
                if i < horizon_action:
                    action = policy.get_action_v(i)
                else:
                    action = None
                taichi_env.step_grad(action)

            taichi_env.apply_agent_action_p_grad(policy.get_actions_p())
            t3 = time()
            print(f"=======> forward: {t2-t1:.2f}s backward: {t3-t2:.2f}s")
            return loss_info, taichi_env.agent.get_grad(horizon_action)

        for iteration in range(self.cfg.n_iters):
            self.logger.save_policy(policy, iteration)
            if iteration % 50 == 0:
                self.render_policy(
                    taichi_env,
                    taichi_env_state,
                    policy,
                    self.env.horizon,
                    self.env.horizon_action,
                    iteration,
                )
            loss_info, grad = forward_backward(
                taichi_env_state["state"],
                policy,
                self.env.horizon,
                self.env.horizon_action,
            )
            loss = loss_info["loss"]
            loss_info["iteration"] = iteration
            policy.optimize(grad, loss_info)

            if self.logger is not None:
                loss_info["lr"] = policy.optim.lr
                self.logger.log(iteration, loss_info)

    def render_policy(
        self, taichi_env, init_state, policy, horizon, horizon_action, iteration
    ):
        taichi_env.set_state(**init_state)
        
        taichi_env.apply_agent_action_p(policy.get_actions_p())

        for i in range(horizon):
            if i < horizon_action:
                action = policy.get_action_v(i, agent=taichi_env.agent, update=True)
            else:
                action = None
            taichi_env.step(action)
            # print(i, taichi_env.get_step_loss())

            save = True
            save = False
            if save:
                img = taichi_env.render("rgb_array")
                self.logger.write_img(img, iteration, i)
            else:
                taichi_env.render("human")
                
    def run_policy(self, policy_name, in_weights_file=None, in_trajs_file=None):
        taichi_env = self.env.taichi_env
        horizon = self.env.horizon
        if policy_name == "random":
            policy = self.env.random_policy()
        elif policy_name in ["gcbc", "gcbc_vae"]:
            assert in_trajs_file is not None
            goal_img_obs = Image.open("goals/0056.png")
            goal_img_obs = np.asarray(goal_img_obs)
            goal_img_obs = self.logger.resize_img(goal_img_obs)
            policy = self.env.bc_policy(goal_img_obs, in_weights_file, policy_name)
        horizon_action = self.env.horizon_action
        init_state = taichi_env.get_state()
        taichi_env.set_state(**init_state)
        img = taichi_env.render("rgb_array")
        img = self.logger.resize_img(img)
        action = policy.get_action(cur_img_obs=img)
        if torch.is_tensor(action):
            action = action.cpu()
        taichi_env.apply_agent_action_p(action)
        for i in range(horizon):
            img = taichi_env.render("rgb_array")
            self.logger.write_img(img, "goals/0056/"+in_weights_file.split("/")[-1].split(".")[0], i)
            if i < horizon_action:
                cur_img_obs = self.logger.resize_img(img)
                action = policy.get_action(cur_img_obs=cur_img_obs)
                if (torch.is_tensor(action)):
                    action = action.cpu()
            else:
                action = None
            taichi_env.step(action)

def solve_policy(env, logger, cfg):
    env.reset()
    solver = Solver(env, logger, cfg)
    solver.solve()


def gen_trajs_from_policy(env, logger, cfg, n_trajs, start_iter, test, beta):
    for i in range(n_trajs):
        env.reset()
        solver = Solver(env, logger, cfg)
        print(test)
        solver.create_trajs(start_iter + i, test, beta)
        print(
            f"Finished creating trajectory {i + 1} at {datetime.now().strftime('%H:%M:%S')}"
        )


def run_bc(env, logger, cfg, weights_file, trajs_file):
    env.reset()
    solver = Solver(env, logger, cfg)
    solver.run_bc(weights_file, trajs_file)


def train_bc(
    env,
    logger,
    cfg,
    out_weights_file,
    in_trajs_file_train,
    in_trajs_file_eval,
    lookahead_amnt,
    agent_type="gcbc_vae",
    in_vae_weights=None
):
    print(agent_type)
    wandb.init(
        project=f"{agent_type}-training-fluid-manip",
        name=out_weights_file,
        config={"Lookahead Amnt": lookahead_amnt, "# Training Epochs": 15},
    )
    solver = Solver(env, logger, cfg, agent_type, in_vae_weights)
    for i in range(15):
        solver.train_bc(i, in_trajs_file_train, lookahead_amnt)
        torch.save(solver.agent.state_dict(), out_weights_file)
        solver.eval_bc(i, in_trajs_file_eval, lookahead_amnt)
def run_policy(env, logger, cfg, policy_name, weights_file, in_trajs_file):
    solver = Solver(env, logger, cfg)
    solver.run_policy(policy_name=policy_name, in_weights_file=weights_file, in_trajs_file=in_trajs_file)
