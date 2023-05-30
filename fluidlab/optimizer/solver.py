# import os
# import cv2
from datetime import datetime
import numpy as np

# import taichi as ti
from fluidlab.utils.misc import is_on_server
from h5py import File

# from fluidlab.fluidengine.taichi_env import TaichiEnv


class Solver:
    def __init__(self, env, logger=None, cfg=None):
        self.cfg = cfg
        self.env = env
        self.target_file = env.target_file
        self.logger = logger

    def create_trajs(self, iteration):
        taichi_env = self.env.taichi_env
        horizon = self.env.horizon
        policy = self.env.random_policy(self.cfg.init_range)
        horizon_action = self.env.horizon_action
        init_state = taichi_env.get_state()
        taichi_env.set_state(**init_state)
        taichi_env.apply_agent_action_p(policy.get_actions_p())
        action_matrix = []
        sim_state_matrix = []
        img_obs_matrix = []
        for i in range(horizon):
            if i < horizon_action:
                action = policy.get_action_v(i, agent=taichi_env.agent, update=True)
            else:
                action = None
            # sim_state = self.env.taichi_env.get_state_RL()
            img = taichi_env.render("rgb_array")
            taichi_env.step(action)
            action_matrix.append(action if action is not None else [0] * 3)
            # sim_state_matrix.append(sim_state)
            self.logger.write_img(img, iteration, i)
            img_obs_matrix.append(self.logger.resize_img(img))
        action_matrix = np.array(action_matrix)
        img_obs_matrix = np.array(img_obs_matrix)
        np.savez(
            f"{self.logger.traj_writer.trajs_fname}/traj_{iteration:04d}",
            actions=action_matrix,
            img_obs=img_obs_matrix,
        )
        # np.save(f"{self.logger.traj_writer.trajs_fname}/traj_{iteration:04d}_action.npy", action_matrix)
        # np.save(f"{self.logger.traj_writer.trajs_fname}/traj_{iteration:04d}_img_obs.npy", img_obs_matrix)
        return img_obs_matrix, action_matrix

    def run_bc(self, weights_file, trajs_file):
        taichi_env = self.env.taichi_env
        horizon = self.env.horizon
        policy = self.env.bc_policy(weights_file)
        f = File(trajs_file, driver="family")
        traj_keys = list(f["exp_latteart"].keys())
        traj = traj_keys[0]
        tsteps = f["exp_latteart"][traj]
        tstep = tsteps["t_0000"]
        loaded_sim_state = dict(tstep["sim_state"])
        taichi_env_state = taichi_env.get_state()
        loaded_sim_state["x"] = loaded_sim_state["x"][:]
        loaded_sim_state["v"] = loaded_sim_state["v"][:]
        loaded_sim_state["F"] = taichi_env_state["state"]["F"]
        loaded_sim_state["C"] = taichi_env_state["state"]["C"]
        loaded_sim_state["used"] = taichi_env_state["state"]["used"]
        loaded_sim_state["agent"] = taichi_env_state["state"]["agent"]
        taichi_env.set_state(loaded_sim_state)
        next_tstep = tsteps["t_0001"]
        cur_img_obs = taichi_env.render("rgb_array")
        cur_img_obs = self.logger.resize_img(cur_img_obs)
        goal_img_obs = next_tstep["img_obs"][:]
        pred_a = policy.get_action(cur_img_obs, goal_img_obs)
        actual_a = tstep["action"][:]
        pred_a = pred_a[0].detach().cpu()
        loss = self.env.get_loss(pred_a, actual_a)
        print(loss)
        f.close()
        raise NotImplementedError

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
        if is_on_server():
            return

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


def solve_policy(env, logger, cfg):
    env.reset()
    solver = Solver(env, logger, cfg)
    solver.solve()


def gen_trajs_from_policy(env, logger, cfg, n_trajs, start_iter):
    # actions_matrix = []
    # img_obs_matrix = []
    for i in range(n_trajs):
        env.reset()
        solver = Solver(env, logger, cfg)
        img_obs_i, action_i = solver.create_trajs(start_iter + i)
        # actions_matrix.append(action_i)
        # img_obs_matrix.append(img_obs_i)
        print(
            f"Finished creating trajectory {i + 1} at {datetime.now().strftime('%H:%M:%S')}"
        )
    # actions_matrix = np.array(actions_matrix)
    # img_obs_matrix = np.array(img_obs_matrix)


def run_bc(env, logger, cfg, weights_file, trajs_file):
    env.reset()
    solver = Solver(env, logger, cfg)
    solver.run_bc(weights_file, trajs_file)
