"""
implmentation taken from labml.ai
"""

from typing import Tuple, Optional

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn
import numpy as np
import math

from cs285.infrastructure.pytorch_util import build_mlp
import cs285.infrastructure.pytorch_util as ptu

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

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.register_buffer("encodings", get_positional_encoding(d_model, max_len))
    def forward(self, t):
        return self.encodings[t]

class EpsModel(nn.Module):
    def __init__(
        self,
        input_shape: int,
        output_shape: int,
        hidden_shape: int,
        num_layers: int,
        n_steps: int,
    ):
        super().__init__()
        self.input_layer = nn.Linear(
            input_shape,
            hidden_shape,
            True
        )
        self.timestep_embedding = PositionalEncoding(
            n_steps,
            hidden_shape,
        ) #nn.Embedding

        self.activation = nn.ReLU()

        self.mlp = build_mlp(
            input_size=hidden_shape,
            output_size=output_shape,
            n_layers=num_layers,
            size=hidden_shape,
            activation=self.activation
        )

        

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor
        ) -> torch.Tensor:
        x = self.input_layer(x) + self.timestep_embedding(t)
        x = self.activation(x)
        x = self.mlp(x)
        return x
        
class DDPM(nn.Module):
    """
    ## Denoise Diffusion
    """

    def __init__(
            self,
            shape: int,
            hidden_size: int = 1024,
            num_layers: int = 5,
            learning_rate: float = 3e-4,
            n_steps: int = 256,
            ):
        super().__init__()
        # $T$
        self.n_steps = n_steps

        # Create $\beta_1, \dots, \beta_T$ linearly increasing variance schedule
        self.beta = torch.linspace(0.0001, 0.04, self.n_steps).to(ptu.device)

        # $\alpha_t = 1 - \beta_t$
        self.alpha = 1. - self.beta
        # $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        # $\sigma^2 = \beta$
        self.sigma2 = self.beta
        
        self.shape = shape
        
        self.eps_model = EpsModel(
            input_shape=shape,
            output_shape=shape,
            hidden_shape=hidden_size,
            num_layers=num_layers,
            n_steps=self.n_steps
        )
        self.optimizer = torch.optim.AdamW(params=self.eps_model.parameters(), lr=learning_rate)

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

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # [gather](utils.html) $\alpha_t$ and compute $\sqrt{\bar\alpha_t} x_0$
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        # $(1-\bar\alpha_t) \mathbf{I}$
        var = 1 - gather(self.alpha_bar, t)
        #
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if eps is None:
            eps = torch.randn_like(x0)

        # get $q(x_t|x_0)$
        mean, var = self.q_xt_x0(x0, t)
        # Sample from $q(x_t|x_0)$
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        
        # $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
        eps_theta = self.eps_model(xt, t)
        # [gather](utils.html) $\bar\alpha_t$
        alpha_bar = gather(self.alpha_bar, t)
        # $\alpha_t$
        alpha = gather(self.alpha, t)
        # $\frac{\beta}{\sqrt{1-\bar\alpha_t}}$
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        # $$\frac{1}{\sqrt{\alpha_t}} \Big(x_t -
        #      \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        # $\sigma^2$
        var = gather(self.sigma2, t)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        eps = torch.randn(xt.shape, device=xt.device)
        # Sample
        return mean + (var ** .5) * eps

    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None):
        # Get batch size
        batch_size = x0.shape[0]
        # Get random $t$ for each sample in the batch
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if noise is None:
            noise = torch.randn_like(x0)

        # Sample $x_t$ for $q(x_t|x_0)$
        xt = self.q_sample(x0, t, eps=noise)
        # Get $\textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)$
        eps_theta = self.eps_model(xt, t)

        # MSE loss
        return F.mse_loss(noise, eps_theta)
    
    def update(self, all_states):
        all_states = (all_states - self.input_mean) / (self.eps + self.input_std)
        state_loss = self.loss(all_states)
        self.optimizer.zero_grad()
        state_loss.backward()
        self.optimizer.step()

        return {
            "state_loss": state_loss.item(),
        }
    
    @torch.no_grad()
    def synthesize(self, n):
        sampled_result = torch.randn(
            (n, self.shape),
            device = ptu.device
        )
        for t_ in range(self.n_steps):
            t = self.n_steps - t_ - 1
            sampled_result = self.p_sample(sampled_result, torch.tensor([t], device = ptu.device))

        sampled_result = self.input_mean + sampled_result * (self.eps + self.input_std)
        return sampled_result