import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseScheduler(nn.Module):
    def __init__(
        self,
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
    ):
        super().__init__()

        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32
            )
        elif beta_schedule == "quadratic":
            self.betas = (
                torch.linspace(
                    beta_start**0.5, beta_end**0.5, num_timesteps, dtype=torch.float32
                )
                ** 2
            )

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod**0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(1 / self.alphas_cumprod - 1)

        # required for q_posterior
        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

        # Make all the tensors Parameters that don't require gradients
        self.betas = nn.Parameter(self.betas, requires_grad=False)
        self.alphas = nn.Parameter(self.alphas, requires_grad=False)
        self.alphas_cumprod = nn.Parameter(self.alphas_cumprod, requires_grad=False)
        self.alphas_cumprod_prev = nn.Parameter(self.alphas_cumprod_prev, requires_grad=False)
        self.sqrt_alphas_cumprod = nn.Parameter(self.sqrt_alphas_cumprod, requires_grad=False)
        self.sqrt_one_minus_alphas_cumprod = nn.Parameter(self.sqrt_one_minus_alphas_cumprod, requires_grad=False)
        self.sqrt_inv_alphas_cumprod = nn.Parameter(self.sqrt_inv_alphas_cumprod, requires_grad=False)
        self.sqrt_inv_alphas_cumprod_minus_one = nn.Parameter(self.sqrt_inv_alphas_cumprod_minus_one, requires_grad=False)
        self.posterior_mean_coef1 = nn.Parameter(self.posterior_mean_coef1, requires_grad=False)
        self.posterior_mean_coef2 = nn.Parameter(self.posterior_mean_coef2, requires_grad=False)

    def reconstruct_x0(self, x_t, t, noise):
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1[None, None, None, None]
        s2 = s2[None, None, None, None]
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        s1 = s1[None, None, None, None]
        s2 = s2[None, None, None, None]
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        if t == 0:
            return 0

        variance = (
            self.betas[t]
            * (1.0 - self.alphas_cumprod_prev[t])
            / (1.0 - self.alphas_cumprod[t])
        )
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

        s1 = s1[:, None, None, None]
        s2 = s2[:, None, None, None]

        return s1 * x_start + s2 * x_noise

    def __len__(self):
        return self.num_timesteps
