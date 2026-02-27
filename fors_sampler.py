import math
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm
import torch


@dataclass
class FORSConfig:
    B: float = 1.0
    max_resample: int = 20


class FORSSampler:
    """
    Minimal DDPM-like FORS sampler (Section 4.2 of fors.pdf).

    This implements Algorithm 1 (FORS) sequentially in the backward process
    (Algorithm 2) with the DDPM-like proposal q_t described in Section 4.2:
      q_t = N( alpha_t^{-1} x_{t+1} + eta_t s_t(alpha_t^{-1} x_{t+1}), eta_t I )

    Notes:
    - Assumes epsilon-prediction UNet (diffusers DDPM-style).
    """

    def __init__(self, model, scheduler, config: Optional[FORSConfig] = None, device: Optional[str] = None):
        self.model = model
        self.scheduler = scheduler
        self.config = config or FORSConfig()
        self.device = device or next(model.parameters()).device
        self.model.eval()

        # Cache alphas_cumprod on device for fast access
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

        if getattr(self.scheduler.config, "prediction_type", "epsilon") != "epsilon":
            raise ValueError("FORSSampler only supports epsilon-prediction models.")

    def set_timesteps(self, num_inference_steps: int):
        self.scheduler.set_timesteps(num_inference_steps)
        self.timesteps = self.scheduler.timesteps

    @torch.no_grad()
    def sample(self, batch_size: int, num_inference_steps: int, generator: Optional[torch.Generator] = None):
        self.set_timesteps(num_inference_steps)

        in_channels = self.model.config.in_channels
        sample_size = self.model.config.sample_size
        if isinstance(sample_size, int):
            height = width = sample_size
        else:
            height, width = sample_size

        t_start = int(self.timesteps[0])
        sigma_start = torch.sqrt(1.0 - self.alphas_cumprod[t_start])

        x = torch.randn(
            (batch_size, in_channels, height, width),
            generator=generator,
            device=self.device,
            dtype=self.alphas_cumprod.dtype,
        ) * sigma_start

        # Iterate over timesteps, skipping the last (t=0)
        for i in tqdm(range(len(self.timesteps) - 1), desc='FORS Sampling'):
            t_cur = int(self.timesteps[i])
            t_prev = int(self.timesteps[i + 1])
            x = self._fors_step(x, t_cur, t_prev)

        return x

    def _fors_step(self, x_next: torch.Tensor, t_cur: int, t_prev: int) -> torch.Tensor:
        """One reverse step: sample x_{t_prev} given x_{t_cur}."""
        alpha_bar_cur = self.alphas_cumprod[t_cur]
        alpha_bar_prev = self.alphas_cumprod[t_prev]

        # alpha_t corresponds to sqrt(alpha_bar_cur / alpha_bar_prev)
        alpha_step = torch.sqrt(alpha_bar_cur / alpha_bar_prev)

        # eta_t: eta = sigma_cur^2 / alpha_step^2 - sigma_prev^2
        sigma_cur_sq = 1.0 - alpha_bar_cur
        sigma_prev_sq = 1.0 - alpha_bar_prev
        eta = sigma_cur_sq / (alpha_step ** 2) - sigma_prev_sq
        eta = torch.clamp(eta, min=1e-42)

        bar_x = x_next / alpha_step
        s_bar = self._score(bar_x, t_prev)

        x_prev = torch.empty_like(bar_x)
        for i in range(bar_x.shape[0]):
            x_prev[i] = self._fors_sample_one(bar_x[i], s_bar[i], t_prev, eta)

        return x_prev

    def _fors_sample_one(
        self,
        bar_x: torch.Tensor,
        s_bar: torch.Tensor,
        t_prev: int,
        eta: torch.Tensor,
    ) -> torch.Tensor:
        B = float(self.config.B)
        sqrt_eta = torch.sqrt(eta)
        mean = bar_x + eta * s_bar

        for i in range(self.config.max_resample):
            x = mean + sqrt_eta * torch.randn_like(bar_x)

            # Algorithm 1: J ~ Poisson(2B)
            J = int(torch.poisson(torch.tensor([2.0 * B], device=self.device)).item())
            if J == 0:
                return x

            # Sample r, z for each W_j
            r = torch.rand((J,), device=self.device)
            z = torch.randn((J, *x.shape), device=self.device) * sqrt_eta

            view_shape = (J,) + (1,) * x.ndim
            a = torch.sin(0.5 * math.pi * r).view(view_shape)
            b = torch.cos(0.5 * math.pi * r).view(view_shape)

            gamma = a * x + (1.0 - a) * bar_x + b * z

            a_prime = 0.5 * math.pi * b
            b_prime = -0.5 * math.pi * a
            dot_gamma = a_prime * (x - bar_x) + b_prime * z

            s_gamma = self._score(gamma, t_prev)
            diff = s_gamma - s_bar.unsqueeze(0)

            # sum in paper, but mean is more stable for large dims
            # note: the paper's formulation is sum_j dot_gamma_j^T (s(gamma_j) - s_bar), 
            # but we take the mean over j because the sum blows up comically.
            # pretty much the only change from the paper.
            w = (dot_gamma * diff).flatten(1).mean(1) 
            
            w = torch.clamp(w, -B, B)
            probs = (B + w) / (2.0 * B)
            log_prob = torch.sum(torch.log(probs + 1e-12))
            accept_prob = torch.exp(log_prob)
            # print(accept_prob.item())

            if torch.rand((), device=self.device) < accept_prob:
                # print('Accepted sample after {} resamples!'.format(i + 1))
                return x

        # Fallback if we exceed max_resample
        # print('Exceeded max resample attempts :(')
        return x

    def _score(self, x: torch.Tensor, t_index: int) -> torch.Tensor:
        # Scale input if scheduler expects it (identity for DDPM)
        x_in = self.scheduler.scale_model_input(x, t_index)

        t = torch.full((x_in.shape[0],), t_index, device=self.device, dtype=torch.long)
        eps = self.model(x_in, t).sample

        sigma = torch.sqrt(1.0 - self.alphas_cumprod[t_index])
        sigma = torch.clamp(sigma, min=1e-12)
        sigma = sigma.reshape((1,) * eps.ndim)

        score = -eps / sigma
        return score
