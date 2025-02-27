import os
import cv2
import gym
import torch
import random
import argparse
import numpy as np

from fluidlab.envs.latteart_env import LatteArtEnv
from fluidlab.envs.pouring_env import PouringEnv
from fluidlab.envs.spreading_env import SpreadingEnv
from fluidlab.configs.macros import *
from fluidlab.utils.logger import Logger
from fluidlab.optimizer.solver import (
    solve_policy,
    gen_trajs_from_policy,
    train_bc,
    run_policy,
)
from fluidlab.optimizer.recorder import record_target, replay_policy, replay_target
from fluidlab.utils.config import load_config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="exp_latteart")
    parser.add_argument("--env_name", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg_file", type=str, default="configs/exp_latteart.yaml")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--user_input", action="store_true")
    parser.add_argument("--replay_policy", action="store_true")
    parser.add_argument("--replay_target", action="store_true")
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--renderer_type", type=str, default="GGUI")
    parser.add_argument("--gen_trajs", action="store_true", default=True)
    parser.add_argument("--n_trajs", type=int, default=1)
    parser.add_argument("--out_ds", type=str, default="trajs.hdf5")
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--beta", type=float, default="0.85")
    parser.add_argument("--gcbc_type", default="gcbc")
    parser.add_argument("--in_weights_file", type=str, default=None)
    parser.add_argument(
        "--eval_trajs_file", type=str, default="eval_trajs/eval_trajs_0000_0049_%d.hdf5"
    )
    parser.add_argument("--roll_trajs_file", type=str, default=None)
    parser.add_argument("--in_vae_weights", type=str, default=None)
    parser.add_argument("--train_trajs_folder", type=str, default="trajs_no_padded_acs")
    parser.add_argument("--out_weights_file", type=str, default="dummy_weights.pt")
    parser.add_argument("--lookahead_amnt", type=int, default=1)
    parser.add_argument("--base", type=float, default=0)
    parser.add_argument("--inj", type=float, default=0)
    parser.add_argument("--phys", type=str)
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
            env = gym.make(
                cfg.EXP.env_name,
                seed=cfg.EXP.seed,
                loss=False,
                loss_type="diff",
                renderer_type=args.renderer_type,
            )
        else:
            env = gym.make(
                args.env_name,
                seed=args.seed,
                loss=False,
                loss_type="diff",
                renderer_type=args.renderer_type,
            )
        record_target(env, exp_name=args.path, path="demo", user_input=args.user_input)
    elif args.replay_target:
        if cfg is not None:
            env = gym.make(
                cfg.EXP.env_name,
                seed=cfg.EXP.seed,
                loss=False,
                loss_type="diff",
                renderer_type=args.renderer_type,
            )
        else:
            env = gym.make(
                args.env_name,
                seed=args.seed,
                loss=False,
                loss_type="diff",
                renderer_type=args.renderer_type,
            )
        replay_target(env)
    elif args.replay_policy:
        if cfg is not None:
            env = gym.make(
                cfg.EXP.env_name,
                seed=cfg.EXP.seed,
                loss=False,
                loss_type="diff",
                renderer_type=args.renderer_type,
            )
        else:
            env = gym.make(
                args.env_name,
                seed=args.seed,
                loss=False,
                loss_type="diff",
                renderer_type=args.renderer_type,
            )
        replay_policy(env, path=args.path)

    elif args.gen_trajs:
        if cfg is not None:
            env = gym.make(
                cfg.EXP.env_name,
                seed=cfg.EXP.seed,
                loss=False,
                loss_type="diff",
                renderer_type=args.renderer_type,
            )
        else:
            env = gym.make(
                args.env_name,
                seed=args.seed,
                loss=False,
                loss_type="diff",
                renderer_type=args.renderer_type,
            )
        logger = Logger(args.exp_name, args.out_ds)
        gen_trajs_from_policy(
            env, logger, cfg.SOLVER, args.n_trajs, args.start_iter, args.beta, args.test
        )
    else:
        logger = Logger(args.exp_name)
        env = gym.make(
            cfg.EXP.env_name,
            seed=cfg.EXP.seed,
            loss=True,
            loss_type="diff",
            renderer_type=args.renderer_type,
        )
        solve_policy(env, logger, cfg.SOLVER)


def main2():
    args = get_args()
    cfg = None
    if args.cfg_file is not None:
        cfg = load_config(args.cfg_file)
    logger = Logger(args.exp_name, args.out_ds)
    if cfg is not None:
        env = gym.make(
            cfg.EXP.env_name,
            seed=cfg.EXP.seed,
            loss=False,
            loss_type="diff",
            renderer_type=args.renderer_type,
        )
    else:
        env = gym.make(
            args.env_name,
            seed=args.seed,
            loss=False,
            loss_type="diff",
            renderer_type=args.renderer_type,
        )
    train_bc(
        env,
        logger,
        cfg,
        args.out_weights_file,
        args.train_trajs_folder,
        args.eval_trajs_file,
        args.lookahead_amnt,
        agent_type=args.gcbc_type,
        in_vae_weights=args.in_vae_weights,
    )


def main3():
    args = get_args()
    cfg = None
    if args.cfg_file is not None:
        cfg = load_config(args.cfg_file)
    logger = Logger(args.exp_name, args.out_ds)
    if cfg is not None:
        env = gym.make(
            cfg.EXP.env_name,
            seed=cfg.EXP.seed,
            loss=False,
            loss_type="diff",
            renderer_type=args.renderer_type,
        )
    else:
        env = gym.make(
            args.env_name,
            seed=args.seed,
            loss=False,
            loss_type="diff",
            renderer_type=args.renderer_type,
        )
    weights_file = args.in_weights_file
    in_trajs_file = args.roll_trajs_file
    agent_type = args.agent_type
    run_policy(
        env,
        logger,
        cfg,
        agent_type,
        weights_file=weights_file,
        in_trajs_file=in_trajs_file,
    )


# testing only the env building
def main4():
    args = get_args()
    if args.cfg_file is not None:
        cfg = load_config(args.cfg_file)
    logger = Logger(args.exp_name, args.out_ds)
    if cfg is not None:
        # env = gym.make(
        #     cfg.EXP.env_name,
        #     seed=cfg.EXP.seed,
        #     loss=False,
        #     loss_type="diff",
        #     renderer_type=args.renderer_type,
        # )
        pass
    else:
        # env = gym.make(
        #     args.env_name,
        #     seed=args.seed,
        #     loss=False,
        #     loss_type="diff",
        #     renderer_type=args.renderer_type,
        # )
        pass
    if args.visc:
        MU[MILK] = args.inj
        MU[COFFEE] = args.base
        MU[ICECREAM] = args.inj
        env = gym.make(
            cfg.EXP.env_name,
            seed=cfg.EXP.seed,
            loss=False,
            loss_type="diff",
            renderer_type=args.renderer_type,
        )
        record_target(
            env,
            exp_name=args.path,
            path=f"visc_base_{args.base:.2f}_visc_inj_{args.inj:.2f}",
            user_input=False,
        )
    if args.lamda:
        LAMDA[MILK] = args.inj
        LAMDA[ICECREAM] = args.inj
        LAMDA[COFFEE] = args.base
        env = gym.make(
            cfg.EXP.env_name,
            seed=cfg.EXP.seed,
            loss=False,
            loss_type="diff",
            renderer_type=args.renderer_type,
        )
        record_target(
            env,
            exp_name=args.path,
            path=f"lambda_base_{args.base:.2f}_lambda_inj_{args.inj:.2f}",
            user_input=False,
        )


def main5():
    args = get_args()
    if args.cfg_file is not None:
        cfg = load_config(args.cfg_file)
    logger = Logger(args.exp_name, args.out_ds)     
    if args.phys == "rho":
        RHO[WATER] = args.base
        RHO[MILK] = args.inj
    elif args.phys == "lambda":
        LAMDA[WATER] = args.base
        LAMDA[MILK] = args.inj
    elif args.phys == "mu":
        MU[WATER] = args.base
        MU[MILK] = args.inj
    env = PouringEnv(None, False, "diff")
    record_target(
           env,
           exp_name=args.path,
           path=f"{args.phys}_base_{args.base:.2f}_{args.phys}_inj_{args.inj:.2f}",
           user_input=False,
        )   

def main6():
    args = get_args()
    if args.cfg_file is not None:
        cfg = load_config(args.cfg_file)
    logger = Logger(args.exp_name, args.out_ds)
    MU[WATER] = -2
    env = SpreadingEnv(None, loss=False)
    record_target(
        env,
        exp_name=args.path,
        path=f"demo",
        user_input=False,
    ) 

if __name__ == "__main__":
    # main()
    # main2()
    # main3()
    # breakpoint()
    # main4()
    # main5()
    main6()
