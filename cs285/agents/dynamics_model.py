from typing import Callable, Optional, Tuple
import numpy as np
import torch.nn as nn
import torch
import gym
from cs285.infrastructure import pytorch_util as ptu


class DynamicsModel(nn.Module):
    def __init__(
        self,
        env: gym.Env,
        make_dynamics_model: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[nn.ParameterList], torch.optim.Optimizer],
        ensemble_size: int,
    ):
        super().__init__()
        self.env = env

        # ensure the environment is state-based
        assert len(env.observation_space.shape) == 1
        assert len(env.action_space.shape) == 1

        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]

        self.ensemble_size = ensemble_size
        self.dynamics_models = nn.ModuleList(
            [
                make_dynamics_model(
                    self.ob_dim,
                    self.ac_dim,
                )
                for _ in range(ensemble_size)
            ]
        )
        self.optimizer = make_optimizer(self.dynamics_models.parameters())
        self.loss_fn = nn.MSELoss()

        self.eps = 1e-12

        # keep track of statistics for both the model input (obs & act) and
        # output (obs delta)
        self.register_buffer(
            "obs_acs_mean", torch.zeros(self.ob_dim + self.ac_dim, device=ptu.device)
        )
        self.register_buffer(
            "obs_acs_std", torch.ones(self.ob_dim + self.ac_dim, device=ptu.device)
        )
        self.register_buffer(
            "obs_delta_mean", torch.zeros(self.ob_dim, device=ptu.device)
        )
        self.register_buffer(
            "obs_delta_std", torch.ones(self.ob_dim, device=ptu.device)
        )
        self.register_buffer(
            "rwd_mean", torch.zeros(1, device=ptu.device)
        )
        self.register_buffer(
            "rwd_std", torch.ones(1, device=ptu.device)
        )

    def update(self, i: int, obs: np.ndarray, acs: np.ndarray, next_obs: np.ndarray, rwds: np.ndarray):
        """
        Update self.dynamics_models[i] using the given batch of data.

        Args:
            i: index of the dynamics model to update
            obs: (batch_size, ob_dim)
            acs: (batch_size, ac_dim)
            next_obs: (batch_size, ob_dim)
            rwds: (batch_size, )
        """
        
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        next_obs = ptu.from_numpy(next_obs)
        rwds = ptu.from_numpy(rwds)
        # TODO(student): update self.dynamics_models[i] using the given batch of data
        # HINT: make sure to normalize the NN input (observations and actions)
        # *and* train it with normalized outputs (observation deltas) ]
        obs_acs = torch.cat([obs, acs], dim = -1)
        obs_delta = next_obs - obs
        norm_obs_acs = (obs_acs - self.obs_acs_mean[None]) / (self.eps + self.obs_acs_std[None])
        norm_obs_delta = (obs_delta - self.obs_delta_mean[None]) / (self.eps + self.obs_delta_std[None])
        norm_rwd = (rwds - self.rwd_mean) / (self.eps + self.rwd_std)

        # HINT 2: make sure to train it with observation *deltas*, not next_obs
        # directly
        # HINT 3: make sure to avoid any risk of dividing by zero when
        # normalizing vectors by adding a small number to the denominator!
        outputs = self.dynamics_models[i](norm_obs_acs)
        loss = self.loss_fn(torch.cat([norm_obs_delta, norm_rwd[:, None]], dim = -1), outputs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return ptu.to_numpy(loss)

    @torch.no_grad()
    def update_statistics(self, obs: np.ndarray, acs: np.ndarray, next_obs: np.ndarray, rwds: np.ndarray):
        """
        Update the statistics used to normalize the inputs and outputs of the dynamics models.

        Args:
            obs: (n, ob_dim)
            acs: (n, ac_dim)
            next_obs: (n, ob_dim)
            rwds: (n, )
        """
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        next_obs = ptu.from_numpy(next_obs)
        rwds = ptu.from_numpy(rwds)
        # TODO(student): update the statistics
        obs_acs = torch.cat([obs, acs], dim = -1)
        obs_delta = next_obs - obs
        self.obs_acs_mean = obs_acs.mean(0)
        self.obs_acs_std = obs_acs.std(0)
        self.obs_delta_mean = obs_delta.mean(0)
        self.obs_delta_std = obs_delta.std(0)
        self.rwd_mean = rwds.mean(0)
        self.rwd_std = rwds.std(0)

    @torch.no_grad()
    def get_dynamics_predictions(
        self, i: int, obs: np.ndarray, acs: np.ndarray
    ) -> np.ndarray:
        """
        Takes a batch of each current observation and action and outputs the
        predicted next observations from self.dynamics_models[i].

        Args:
            obs: (batch_size, ob_dim)
            acs: (batch_size, ac_dim)
        Returns: (batch_size, ob_dim), (batch_size, )
        """
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        # TODO(student): get the model's predicted `next_obs`
        # HINT: make sure to *unnormalize* the NN outputs (observation deltas)
        # Same hints as `update` above, avoid nasty divide-by-zero errors when
        # normalizing inputs!
        obs_acs = torch.cat([obs, acs], dim = -1)
        norm_obs_acs = (obs_acs - self.obs_acs_mean[None]) / (self.eps + self.obs_acs_std[None])
        outputs = self.dynamics_models[i](norm_obs_acs)
        norm_obs_delta_hat, norm_rwds = outputs[:, :-1], outputs[:, -1]
        pred_next_obs = obs + norm_obs_delta_hat * (self.eps + self.obs_delta_std[None]) + self.obs_delta_mean[None]
        rwds = self.rwd_mean + norm_rwds * (self.eps + self.rwd_std)
        return ptu.to_numpy(pred_next_obs), ptu.to_numpy(rwds)
