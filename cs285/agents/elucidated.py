"""
The implementation of the Elucidated Diffuser is based on the SynthER repo:
https://github.com/conglu1997/SynthER/blob/main/synther/diffusion/elucidated_diffusion.py
"""

from math import sqrt
from random import random
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import cs285.infrastructure.pytorch_util as ptu

from cs285.agents.utils import ResidualMLPDenoiser

from tqdm import tqdm
from einops import rearrange, reduce

# helpers

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# tensor helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# main class

class ElucidatedDiffusion(nn.Module):
    def __init__(
        self,
        shape,
        T0 = 10000,
        hidden_size = 32,
        num_layers = 2,
        batch_size = 32,
        learning_rate = 3e-4,
        self_condition = False,
        n_steps = 32,          # number of sampling steps
        sigma_min = 0.002,     # min noise level
        sigma_max = 80,        # max noise level
        sigma_data = 0.5,      # standard deviation of data distribution
        rho = 7,               # controls the sampling schedule
        P_mean = -1.2,         # mean of log-normal distribution from which noise is drawn for training
        P_std = 1.2,           # standard deviation of log-normal distribution from which noise is drawn for training
        S_churn = 80,          # parameters for stochastic sampling - depends on dataset, Table 5 in apper
        S_tmin = 0.05,
        S_tmax = 50,
        S_noise = 1.003,
    ):
        super().__init__()
        # assert net.random_or_learned_sinusoidal_cond
        # self.self_condition = net.self_condition
        
        self.shape = shape

        self.net = ResidualMLPDenoiser(
            d_in=self.shape,
            dim_t=n_steps,
            mlp_width=hidden_size,
            num_layers=num_layers
        )
        self.optimizer = torch.optim.AdamW(params=self.net.parameters(), lr=learning_rate)
        self.self_condition = self_condition

        # image dimensions
        self.batch_size = batch_size

        # parameters

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.rho = rho

        self.P_mean = P_mean
        self.P_std = P_std

        self.num_sample_steps = n_steps  # otherwise known as N in the paper

        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T0)

        self.eps = 1e-12

        # keep track of statistics for both the model input (obs & act) and
        # output (obs delta)
        self.register_buffer(
            "input_mean", torch.zeros(shape, device=ptu.device)
        )
        self.register_buffer(
            "input_std", torch.ones(shape, device=ptu.device)
        )
    
    def update_statistics(self, inputs: torch.Tensor):
        # TODO(student): update the statistics
        self.input_mean = inputs.mean(0)
        self.input_std = inputs.std(0)

    # derived preconditioning params - Table 1

    def c_skip(self, sigma):
        return (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma):
        return 1 * (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25

    # preconditioned network output
    # equation (7) in the paper

    def preconditioned_network_forward(self, noised_states, sigma, self_cond = None, clamp = False):
        batch = noised_states.shape[0]

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device = ptu.device)

        padded_sigma = rearrange(sigma, 'b -> b 1')

        net_out = self.net(
            self.c_in(padded_sigma) * noised_states,
            self.c_noise(sigma),
            self_cond
        )

        out = self.c_skip(padded_sigma) * noised_states +  self.c_out(padded_sigma) * net_out

        if clamp:
            out = out.clamp(-1., 1.)

        return out

    # sampling

    # sample schedule
    # equation (5) in the paper

    def sample_schedule(self, num_sample_steps = None):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        N = num_sample_steps
        inv_rho = 1 / self.rho

        steps = torch.arange(num_sample_steps, device = ptu.device, dtype = torch.float32)
        sigmas = (self.sigma_max ** inv_rho + steps / (N - 1) * (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value = 0.) # last step is sigma value of 0.
        return sigmas

    @torch.no_grad()
    def synthesize(self, n, num_sample_steps = None, clamp = False):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        shape = (n, self.shape)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma

        sigmas = self.sample_schedule(num_sample_steps)

        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, sqrt(2) - 1),
            0.
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # states is noise at the beginning

        init_sigma = sigmas[0]

        states = init_sigma * torch.randn(shape, device = ptu.device)

        # for self conditioning

        x_start = None

        # gradually denoise

        for sigma, sigma_next, gamma in sigmas_and_gammas:
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            eps = self.S_noise * torch.randn(shape, device = ptu.device) # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            states_hat = states + sqrt(sigma_hat ** 2 - sigma ** 2) * eps

            self_cond = x_start if self.self_condition else None

            model_output = self.preconditioned_network_forward(states_hat, sigma_hat, self_cond, clamp = clamp)
            denoised_over_sigma = (states_hat - model_output) / sigma_hat

            states_next = states_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            # second order correction, if not the last timestep

            if sigma_next != 0:
                self_cond = model_output if self.self_condition else None

                model_output_next = self.preconditioned_network_forward(states_next, sigma_next, self_cond, clamp = clamp)
                denoised_prime_over_sigma = (states_next - model_output_next) / sigma_next
                states_next = states_hat + 0.5 * (sigma_next - sigma_hat) * (denoised_over_sigma + denoised_prime_over_sigma)

            states = states_next
            x_start = model_output_next if sigma_next != 0 else model_output
        return self.input_mean + states * (self.eps + self.input_std)

    def loss_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2

    def noise_distribution(self):
        return (self.P_mean + self.P_std * torch.randn((self.batch_size,), device = ptu.device)).exp()

    def forward(self, states):
        assert states.shape[1] == self.shape, f'shape must be {self.shape}'

        sigmas = self.noise_distribution()
        padded_sigmas = rearrange(sigmas, 'b -> b 1')

        noise = torch.randn_like(states)

        noised_states = states + padded_sigmas * noise  # alphas are 1. in the paper

        self_cond = None

        if self.self_condition and random() < 0.5:
            # from hinton's group's bit diffusion paper
            with torch.no_grad():
                self_cond = self.preconditioned_network_forward(noised_states, sigmas)
                self_cond.detach_()

        denoised = self.preconditioned_network_forward(noised_states, sigmas, self_cond)

        losses = F.mse_loss(denoised, states, reduction = 'none')
        losses = reduce(losses, 'b ... -> b', 'mean')

        losses = losses * self.loss_weight(sigmas)

        return losses.mean()

    def update(self, states):
        states = (states - self.input_mean) / (self.eps + self.input_std)
        state_loss = self.forward(states)
        self.optimizer.zero_grad()
        state_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return {"state_loss": state_loss.item(), "lr": self.optimizer.param_groups[0]["lr"]}