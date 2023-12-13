"""
implmentation taken from labml.ai
---
title: Denoising Diffusion Implicit Models (DDIM) Sampling
summary: >
 Annotated PyTorch implementation/tutorial of
 Denoising Diffusion Implicit Models (DDIM) Sampling
 for stable diffusion model.
---

# Denoising Diffusion Implicit Models (DDIM) Sampling

This implements DDIM sampling from the paper
[Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
"""

from typing import Optional, List

import numpy as np
import torch

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn
import numpy as np
import math

from labml import monit
from cs285.infrastructure.pytorch_util import build_mlp
import cs285.infrastructure.pytorch_util as ptu
from cs285.diffusion.denoiser_network import ResidualMLPDenoiser


def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1)

def get_positional_encoding(d_model: int, max_len: int = 5000):
    # Empty encodings vectors
    encodings = torch.zeros(max_len, d_model)
    # Position indexes
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    # $2 * i$
    two_i = torch.arange(0, d_model, 2, dtype=torch.float32).unsqueeze(0)
    # $10000^{\frac{2i}{d_{model}}}$
    div_term = torch.exp(two_i * -(math.log(10000.0) / d_model))
    # $PE_{p,2i} = sin\Bigg(\frac{p}{10000^{\frac{2i}{d_{model}}}}\Bigg)$
    encodings[:, 0::2] = torch.sin(position * div_term)
    # $PE_{p,2i + 1} = cos\Bigg(\frac{p}{10000^{\frac{2i}{d_{model}}}}\Bigg)$
    encodings[:, 1::2] = torch.cos(position * div_term)

    # Add batch dimension
    encodings = encodings.requires_grad_(False)

    return encodings

class BatchNormWithInverse(nn.Module):

    def __init__(
            self, 
            normalized_shape: int, 
            momentum: float = 1e-2,
            ):
        super().__init__()
        self.register_buffer(
            "mean",
            torch.zeros((1, normalized_shape)),
            True
        )
        self.register_buffer(
            "std",
            torch.ones((1, normalized_shape)),
            True
        )
        self.momentum = momentum
    
    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.mean = (1 - self.momentum) * self.mean + self.momentum * x.mean(0, keepdim=True)
                self.std = (1 - self.momentum) * self.std + self.momentum * x.std(0,  keepdim=True)
        return (x - self.mean) / self.std
    
    def inverse(self, y):
        return self.mean + y * self.std


class DDIM(nn.Module):
    """
    ## DDIM Sampler

    This extends the [`DiffusionSampler` base class](index.html).

    DDIM samples images by repeatedly removing noise by sampling step by step using,

    \begin{align}
    x_{\tau_{i-1}} &= \sqrt{\alpha_{\tau_{i-1}}}\Bigg(
            \frac{x_{\tau_i} - \sqrt{1 - \alpha_{\tau_i}}\epsilon_\theta(x_{\tau_i})}{\sqrt{\alpha_{\tau_i}}}
            \Bigg) \\
            &+ \sqrt{1 - \alpha_{\tau_{i- 1}} - \sigma_{\tau_i}^2} \cdot \epsilon_\theta(x_{\tau_i}) \\
            &+ \sigma_{\tau_i} \epsilon_{\tau_i}
    \end{align}

    where $\epsilon_{\tau_i}$ is random noise,
    $\tau$ is a subsequence of $[1,2,\dots,T]$ of length $S$,
    and
    $$\sigma_{\tau_i} =
    \eta \sqrt{\frac{1 - \alpha_{\tau_{i-1}}}{1 - \alpha_{\tau_i}}}
    \sqrt{1 - \frac{\alpha_{\tau_i}}{\alpha_{\tau_{i-1}}}}$$

    Note that, $\alpha_t$ in DDIM paper refers to ${\color{lightgreen}\bar\alpha_t}$ from [DDPM](ddpm.html).
    """
    
    def __init__(
            self,
            obs_shape: int,
            ac_shape: int,
            hidden_size: int = 1024,
            num_layers: int = 5,
            learning_rate: float = 3e-4,
            n_steps: int = 256,
            batch_size: int = 256,
            ddim_discretize: str = "uniform",
            ddim_eta: float = 0.
        ):
        """
        :param model: is the model to predict noise $\epsilon_\text{cond}(x_t, c)$
        :param n_steps: is the number of DDIM sampling steps, $S$
        :param ddim_discretize: specifies how to extract $\tau$ from $[1,2,\dots,T]$.
            It can be either `uniform` or `quad`.
        :param ddim_eta: is $\eta$ used to calculate $\sigma_{\tau_i}$. $\eta = 0$ makes the
            sampling process deterministic.
        """
        super().__init__()
        self.n_steps = n_steps
        self.obs_shape = obs_shape
        self.ac_shape = ac_shape
        self.device = ptu.device
        
        self.eps_model = ResidualMLPDenoiser(
            d_in=self.obs_shape*2 + self.ac_shape + 2,
            dim_t=self.n_steps,
            mlp_width=hidden_size,
            num_layers=num_layers
        )
        self.optimizer = torch.optim.AdamW(params=self.eps_model.parameters(), lr=learning_rate)
        self.norm = BatchNormWithInverse(self.obs_shape*2 + self.ac_shape + 2)

        # Calculate $\tau$ to be uniformly distributed across $[1,2,\dots,T]$
        if ddim_discretize == 'uniform':
            c = self.n_steps // n_steps
            self.time_steps = np.asarray(list(range(0, self.n_steps, c))) #+ 1
        # Calculate $\tau$ to be quadratically distributed across $[1,2,\dots,T]$
        elif ddim_discretize == 'quad':
            self.time_steps = ((np.linspace(0, np.sqrt(self.n_steps * .8), n_steps)) ** 2).astype(int) #+ 1
        else:
            raise NotImplementedError(ddim_discretize)

        with torch.no_grad():
            # Get ${\color{lightgreen}\bar\alpha_t}$
            # Create $\beta_1, \dots, \beta_T$ linearly increasing variance schedule
            beta = torch.linspace(0.00085 ** 0.5, 0.0120 ** 0.5, n_steps, dtype=torch.float64) ** 2
            self.beta = nn.Parameter(beta.to(torch.float32), requires_grad=False)
            # $\alpha_t = 1 - \beta_t$
            alpha = 1. - beta
            # $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
            alpha_bar = torch.cumprod(alpha, dim=0)
            self.alpha_bar = nn.Parameter(alpha_bar.to(torch.float32), requires_grad=False)

            # $\alpha_{\tau_i}$
            self.ddim_alpha = self.alpha_bar[self.time_steps].clone().to(torch.float32)
            # $\sqrt{\alpha_{\tau_i}}$
            self.ddim_alpha_sqrt = torch.sqrt(self.ddim_alpha)
            # $\alpha_{\tau_{i-1}}$
            self.ddim_alpha_prev = torch.cat([self.alpha_bar[0:1], self.alpha_bar[self.time_steps[:-1]]])

            # $$\sigma_{\tau_i} =
            # \eta \sqrt{\frac{1 - \alpha_{\tau_{i-1}}}{1 - \alpha_{\tau_i}}}
            # \sqrt{1 - \frac{\alpha_{\tau_i}}{\alpha_{\tau_{i-1}}}}$$
            self.ddim_sigma = (ddim_eta *
                               ((1 - self.ddim_alpha_prev) / (1 - self.ddim_alpha) *
                                (1 - self.ddim_alpha / self.ddim_alpha_prev)) ** .5)

            # $\sqrt{1 - \alpha_{\tau_i}}$
            self.ddim_sqrt_one_minus_alpha = (1. - self.ddim_alpha) ** .5

    @torch.no_grad()
    def sample(self,
               num_samples: int,
               repeat_noise: bool = False,
               temperature: float = 1.,
               x_last: Optional[torch.Tensor] = None,
               skip_steps: int = 0,
               ):
        """
        ### Sampling Loop

        :param shape: is the shape of the generated images in the
            form `[batch_size, channels, height, width]`
        :param temperature: is the noise temperature (random noise gets multiplied by this)
        :param x_last: is $x_{\tau_S}$. If not provided random noise will be used.
        :param skip_steps: is the number of time steps to skip $i'$. We start sampling from $S - i'$.
            And `x_last` is then $x_{\tau_{S - i'}}$.
        """

        # Get device and batch size
        shape = [num_samples, 2 * self.obs_shape + self.ac_shape + 2]
        device = self.device
        bs = shape[0]

        # Get $x_{\tau_S}$
        x = x_last if x_last is not None else torch.randn(shape, device=device)

        # Time steps to sample at $\tau_{S - i'}, \tau_{S - i' - 1}, \dots, \tau_1$
        time_steps = np.flip(self.time_steps)[skip_steps:]

        for i, step in enumerate(time_steps):
            # Index $i$ in the list $[\tau_1, \tau_2, \dots, \tau_S]$
            index = len(time_steps) - i - 1
            # Time step $\tau_i$
            ts = x.new_full((bs,), step, dtype=torch.long)

            # Sample $x_{\tau_{i-1}}$
            x, pred_x0, e_t = self.p_sample(x, ts, index=index,
                                            repeat_noise=repeat_noise,
                                            temperature=temperature)

        # Return $x_0$
        sampled_result = self.norm.inverse(x)
        return {
            "observations": sampled_result[:, :self.obs_shape],
            "actions": sampled_result[:, self.obs_shape: self.obs_shape + self.ac_shape],
            "rewards": sampled_result[:, self.obs_shape + self.ac_shape],
            "next_observations": sampled_result[:, self.obs_shape + self.ac_shape + 1 : 2 * self.obs_shape + self.ac_shape + 1],
            "dones": sampled_result[:, 2 * self.obs_shape + self.ac_shape + 1],
        }

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, index: int, *,
                 repeat_noise: bool = False,
                 temperature: float = 1.):
        """
        ### Sample $x_{\tau_{i-1}}$

        :param x: is $x_{\tau_i}$ of shape `[batch_size, channels, height, width]`
        :param t: is $\tau_i$ of shape `[batch_size]`
        :param step: is the step $\tau_i$ as an integer
        :param index: is index $i$ in the list $[\tau_1, \tau_2, \dots, \tau_S]$
        :param repeat_noise: specified whether the noise should be same for all samples in the batch
        :param temperature: is the noise temperature (random noise gets multiplied by this)
        """

        # Get $\epsilon_\theta(x_{\tau_i})$
        e_t = self.eps_model(x, t)

        # Calculate $x_{\tau_{i - 1}}$ and predicted $x_0$
        x_prev, pred_x0 = self.get_x_prev_and_pred_x0(e_t, index, x,
                                                      temperature=temperature,
                                                      repeat_noise=repeat_noise)

        #
        return x_prev, pred_x0, e_t

    def get_x_prev_and_pred_x0(self, e_t: torch.Tensor, index: int, x: torch.Tensor, *,
                               temperature: float,
                               repeat_noise: bool):
        """
        ### Sample $x_{\tau_{i-1}}$ given $\epsilon_\theta(x_{\tau_i})$
        """

        # $\alpha_{\tau_i}$
        alpha = self.ddim_alpha[index]
        # $\alpha_{\tau_{i-1}}$
        alpha_prev = self.ddim_alpha_prev[index]
        # $\sigma_{\tau_i}$
        sigma = self.ddim_sigma[index]
        # $\sqrt{1 - \alpha_{\tau_i}}$
        sqrt_one_minus_alpha = self.ddim_sqrt_one_minus_alpha[index]

        # Current prediction for $x_0$,
        # $$\frac{x_{\tau_i} - \sqrt{1 - \alpha_{\tau_i}}\epsilon_\theta(x_{\tau_i})}{\sqrt{\alpha_{\tau_i}}}$$
        pred_x0 = (x - sqrt_one_minus_alpha * e_t) / (alpha ** 0.5)
        # Direction pointing to $x_t$
        # $$\sqrt{1 - \alpha_{\tau_{i- 1}} - \sigma_{\tau_i}^2} \cdot \epsilon_\theta(x_{\tau_i})$$
        dir_xt = (1. - alpha_prev - sigma ** 2).sqrt() * e_t

        # No noise is added, when $\eta = 0$
        if sigma == 0.:
            noise = 0.
        # If same noise is used for all samples in the batch
        elif repeat_noise:
            noise = torch.randn((1, *x.shape[1:]), device=x.device)
            # Different noise for each sample
        else:
            noise = torch.randn(x.shape, device=x.device)

        # Multiply noise by the temperature
        noise = noise * temperature

        #  \begin{align}
        #     x_{\tau_{i-1}} &= \sqrt{\alpha_{\tau_{i-1}}}\Bigg(
        #             \frac{x_{\tau_i} - \sqrt{1 - \alpha_{\tau_i}}\epsilon_\theta(x_{\tau_i})}{\sqrt{\alpha_{\tau_i}}}
        #             \Bigg) \\
        #             &+ \sqrt{1 - \alpha_{\tau_{i- 1}} - \sigma_{\tau_i}^2} \cdot \epsilon_\theta(x_{\tau_i}) \\
        #             &+ \sigma_{\tau_i} \epsilon_{\tau_i}
        #  \end{align}
        x_prev = (alpha_prev ** 0.5) * pred_x0 + dir_xt + sigma * noise

        #
        return x_prev, pred_x0

    @torch.no_grad()
    def q_sample(self, x0: torch.Tensor, index: int, noise: Optional[torch.Tensor] = None):
        """
        ### Sample from $q_{\sigma,\tau}(x_{\tau_i}|x_0)$

        $$q_{\sigma,\tau}(x_t|x_0) =
         \mathcal{N} \Big(x_t; \sqrt{\alpha_{\tau_i}} x_0, (1-\alpha_{\tau_i}) \mathbf{I} \Big)$$

        :param x0: is $x_0$ of shape `[batch_size, channels, height, width]`
        :param index: is the time step $\tau_i$ index $i$
        :param noise: is the noise, $\epsilon$
        """

        # Random noise, if noise is not specified
        if noise is None:
            noise = torch.randn_like(x0)

        # Sample from
        #  $$q_{\sigma,\tau}(x_t|x_0) =
        #          \mathcal{N} \Big(x_t; \sqrt{\alpha_{\tau_i}} x_0, (1-\alpha_{\tau_i}) \mathbf{I} \Big)$$
        # print(
        #     f"""
        #     ATTENTION:
        #     self.ddim_alpha_sqrt: {self.ddim_alpha_sqrt.shape}
        #     self.ddim_alpha_sqrt[index]: {self.ddim_alpha_sqrt[index].shape}
        #     (self.ddim_alpha_sqrt[index] * x0.T).T : {(self.ddim_alpha_sqrt[index] * x0.T).T.shape}
        #     x0: {x0.shape}
        #     noise: {noise.shape}
        #     """
        # )
        
        return (self.ddim_alpha_sqrt[index] * x0.T).T + (self.ddim_sqrt_one_minus_alpha[index] * noise.T).T
        # return self.ddim_alpha_sqrt[index] * x0 + self.ddim_sqrt_one_minus_alpha[index] * noise
    
    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None):
        # Get batch size
        batch_size = x0.shape[0]
        # Get random $t$ for each sample in the batch
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if noise is None:
            noise = torch.randn_like(x0)

        # Sample $x_t$ for $q(x_t|x_0)$
        xt = self.q_sample(x0, t, noise=noise)
        # Get $\textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)$
        eps_theta = self.eps_model(xt, t)

        # MSE loss
        return F.mse_loss(noise, eps_theta)
    
    def update(self, obs, action, rewards, next_obs, terminal):
        all_states = torch.cat([obs, action, rewards[:, None], next_obs , terminal[:, None]], axis=-1)
        all_states = self.norm(all_states)
        state_loss = self.loss(all_states)
        self.optimizer.zero_grad()
        state_loss.backward()
        self.optimizer.step()

        return {
            "state_loss": state_loss.item(),
        }
