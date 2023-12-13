from typing import Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn

from cs285.agents.ddpm import DDPM
from cs285.agents.elucidated import ElucidatedDiffusion
import cs285.infrastructure.pytorch_util as ptu


def diffusion_config(
    sample_what: str = "state",
    model_name: str = "DDPM",
    hidden_size: int = 1024,
    num_layers: int = 5,
    learning_rate: float = 3e-4,
    n_steps: int = 100,
    batch_size: int = 128,
    steps_per_iter: int = 5000,
    initial_steps: int = 10000,
    elucidated_kwargs: Dict[str, object] = {},
):
    def make_diffusion(observation_dim: int, action_dim: int) -> nn.Module:
        if sample_what == "state":
            shape = observation_dim
        elif sample_what == "all":
            shape = 2 * observation_dim + action_dim  + 1
        else:
            raise ValueError("sampling option not implemented")

        if model_name == "DDPM":
            return DDPM(
                shape=shape,
                hidden_size=hidden_size,
                num_layers=num_layers,
                learning_rate=learning_rate, 
                n_steps=n_steps
            )
        elif model_name == "Elucidated":
            return ElucidatedDiffusion(
                shape=shape,
                T0=steps_per_iter,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_size=batch_size,
                learning_rate=learning_rate,
                **elucidated_kwargs
            )
        else:
            raise ValueError("model not implemented")
    
    def join_predictions(
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray
    ):
        if sample_what == "state":
            return ptu.from_numpy(observations)
        elif sample_what == "all":
            return torch.cat([
                ptu.from_numpy(observations), 
                ptu.from_numpy(actions), 
                ptu.from_numpy(rewards)[:, None],
                ptu.from_numpy(next_observations), 
            ], dim=1)

    def split_predictions(all_states, observation_dim, action_dim):
        if sample_what == "state":
            return {
                "observations": all_states
            }
        elif sample_what == "all":
            return {
                "observations": all_states[:, :observation_dim],
                "actions": all_states[:, observation_dim: observation_dim + action_dim],
                "rewards": all_states[:, observation_dim + action_dim],
                "next_observations": all_states[:, observation_dim + action_dim + 1 : 2 * observation_dim + action_dim + 1],
                "dones": np.zeros_like(all_states[:, observation_dim + action_dim]),
            }

    return {
        "sample_what": sample_what,
        "make_diffusion": make_diffusion,
        "join_predictions": join_predictions,
        "split_predictions": split_predictions,
        "initial_steps": initial_steps,
        "steps_per_iter": steps_per_iter,
        "batch_size": batch_size,
    }
