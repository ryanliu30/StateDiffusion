import torch.nn as nn
from cs285.infrastructure import pytorch_util as ptu
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gym.wrappers.rescale_action import RescaleAction
from gym.wrappers.clip_action import ClipAction
import gym
import torch
from typing import Optional


def dynamics_config(
    env_name: str,
    exp_name: str,
    log_dir: str,
    hidden_size: int = 128,
    num_layers: int = 3,
    learning_rate: float = 1e-3,
    ensemble_size: int = 3,
    initial_batch_size: int = 20000,  # number of transitions to collect with random policy at the start
    batch_size: int = 8000,  # number of transitions to collect per per iteration thereafter
    train_batch_size: int = 512,  # number of transitions to train each dynamics model per iteration
    num_iters: int = 20,
    replay_buffer_capacity: int = 1000000,
    num_agent_train_steps_per_iter: int = 20,
    num_eval_trajectories: int = 10,
):

    def make_dynamics_model(ob_dim: int, ac_dim: int) -> nn.Module:
        return ptu.build_mlp(
            input_size=ob_dim + ac_dim,
            output_size=ob_dim + 1,
            n_layers=num_layers,
            size=hidden_size,
        )

    def make_optimizer(params: nn.ParameterList):
        return torch.optim.Adam(params, lr=learning_rate)

    def make_env(render: bool = False): 
        env = RecordEpisodeStatistics(
            ClipAction(
                RescaleAction(
                    gym.make(
                        id=env_name,
                        render_mode="single_rgb_array" if render else None
                    ),  -1, 1
                )
            )
        )
        return env

    log_string = f"{env_name}_{exp_name}_l{num_layers}_h{hidden_size}"

    return {
        "agent_kwargs": {
            "make_dynamics_model": make_dynamics_model,
            "make_optimizer": make_optimizer,
            "ensemble_size": ensemble_size,
        },
        "log_dir": log_dir,
        "make_env": make_env,
        "replay_buffer_capacity": replay_buffer_capacity,
        "log_name": log_string,
        "num_iters": num_iters,
        "batch_size": batch_size,
        "initial_batch_size": initial_batch_size,
        "train_batch_size": train_batch_size,
        "num_agent_train_steps_per_iter": num_agent_train_steps_per_iter,
        "num_eval_trajectories": num_eval_trajectories,
    }
