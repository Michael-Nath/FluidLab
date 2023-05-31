import os
import gym
import torch
import random
import argparse
import numpy as np

import fluidlab.envs
from fluidlab.envs.fluid_env import FluidEnv
from fluidlab.utils.logger import Logger
from fluidlab.optimizer.solver import solve_policy, gen_trajs_from_policy, run_bc
from fluidlab.optimizer.recorder import record_target, replay_policy, replay_target
from fluidlab.utils.config import load_config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='exp_latteart')
    parser.add_argument("--env_name", type=str, default='')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg_file", type=str, default='configs/exp_latteart.yaml')
    parser.add_argument("--record", action='store_true')
    parser.add_argument("--user_input", action='store_true')
    parser.add_argument("--replay_policy", action='store_true')
    parser.add_argument("--replay_target", action='store_true')
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--renderer_type", type=str, default='GGUI')
    parser.add_argument("--gen_trajs", action="store_true", default=True)
    parser.add_argument("--n_trajs", type=int, default=1)
    parser.add_argument("--out_ds", type=str, default="trajs.hdf5")
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--in_weights_file", type=str, default="gcbc_weights.pt")
    parser.add_argument("--in_trajs_file", type=str, default="data/trajs0000_0500_%d.hdf5")

    args = parser.parse_args()

    return args

def main():
    args = get_args()
    if args.cfg_file is not None:
        cfg = load_config(args.cfg_file)
    else:
        cfg = None

    if args.record:
        if cfg is not None:
            env = gym.make(cfg.EXP.env_name, seed=cfg.EXP.seed, loss=False, loss_type='diff', renderer_type=args.renderer_type)
        else:
            env = gym.make(args.env_name, seed=args.seed, loss=False, loss_type='diff', renderer_type=args.renderer_type)
        record_target(env, path=args.path, user_input=args.user_input)
    elif args.replay_target:
        if cfg is not None:
            env = gym.make(cfg.EXP.env_name, seed=cfg.EXP.seed, loss=False, loss_type='diff', renderer_type=args.renderer_type)
        else:
            env = gym.make(args.env_name, seed=args.seed, loss=False, loss_type='diff', renderer_type=args.renderer_type)
        replay_target(env)
    elif args.replay_policy:
        if cfg is not None:
            env = gym.make(cfg.EXP.env_name, seed=cfg.EXP.seed, loss=False, loss_type='diff', renderer_type=args.renderer_type)
        else:
            env = gym.make(args.env_name, seed=args.seed, loss=False, loss_type='diff', renderer_type=args.renderer_type)
        replay_policy(env, path=args.path)

    elif args.gen_trajs:
        if cfg is not None:
            env = gym.make(cfg.EXP.env_name, seed=cfg.EXP.seed, loss=False, loss_type="diff", renderer_type=args.renderer_type)
        else:
            env = gym.make(args.env_name, seed=args.seed, loss=False, loss_type="diff", renderer_type=args.renderer_type)
        logger = Logger(args.exp_name, args.out_ds)
        gen_trajs_from_policy(env, logger, cfg.SOLVER, args.n_trajs, args.start_iter, args.test)
    else:
        logger = Logger(args.exp_name)
        env = gym.make(cfg.EXP.env_name, seed=cfg.EXP.seed, loss=True, loss_type='diff', renderer_type=args.renderer_type)
        solve_policy(env, logger, cfg.SOLVER)


def main2():
    args = get_args()
    cfg = None
    if args.cfg_file is not None:
        cfg = load_config(args.cfg_file)
    logger = Logger(args.exp_name, "blooblah")
    if cfg is not None:
        env = gym.make(cfg.EXP.env_name, seed=cfg.EXP.seed, loss=False, loss_type="diff", renderer_type=args.renderer_type)
    else:
        env = gym.make(args.env_name, seed=args.seed, loss=False, loss_type="diff", renderer_type=args.renderer_type)
    run_bc(env, logger, cfg, args.in_weights_file, args.in_trajs_file)
    


if __name__ == '__main__':
    # main()
    main2()