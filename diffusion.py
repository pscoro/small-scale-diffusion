from typing import Tuple

import torch
import torch.nn as nn


class Diffusion(nn.Module):
    def __init__(self, steps: int, start: float = 0.0001, end: float = 0.02):
        super().__init__()
        self.steps = steps

        self.beta = nn.Parameter(torch.linspace(start, end, steps), requires_grad=False)
        self.alpha_cumprod = nn.Parameter(torch.cumprod(1 - self.beta, 0), requires_grad=False)

    def forward(self, samples: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alphas = self.alpha_cumprod[timesteps]

        noise = torch.randn_like(samples)
        scaled_samples = alphas.sqrt()[:, None, None, None] * samples
        scaled_noise = (1 - alphas).sqrt()[:, None, None, None] * noise

        return scaled_samples + scaled_noise, noise

    def backward(self, sample: torch.Tensor, timestep: int, noise_pred: torch.Tensor) -> torch.Tensor:
        beta = self.beta[timestep]
        alpha = 1 - beta
        noise_pred_scale = beta / torch.sqrt(1 - self.alpha_cumprod[timestep])
        denoised_sample = (1 / torch.sqrt(alpha)) * (sample - noise_pred_scale * noise_pred)
        if timestep > 0:
            denoised_sampled = denoised_sample + torch.sqrt(beta) * torch.randn_like(sample)
        return denoised_sample


def main() -> None:
    diffusion = Diffusion(1000)

    samples = torch.randn(64, 3, 32, 32)
    timesteps = torch.randint(0, 1000, (64,))
    noisy_samples, noise = diffusion(samples, timesteps)
    print(noisy_samples.shape, noise.shape)

    sample = torch.randn(3, 32, 32)
    timestep = 100
    noise_pred = torch.randn(3, 32, 32)
    denoised_sample = diffusion.backward(sample, timestep, noise_pred)
    print(denoised_sample.shape)


if __name__ == "__main__":
    main()
