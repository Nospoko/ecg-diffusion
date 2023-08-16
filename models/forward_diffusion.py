import torch
import torch.nn as nn
import torch.nn.functional as F


class ForwardDiffusion(nn.Module):
    def __init__(self, beta_start: float, beta_end: float, timesteps: int, schedule_type: str = "cosine"):
        super().__init__()

        # attributes
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.schedule_type = schedule_type

        # extracting beta values
        betas = self._get_beta_schedule()
        self.register_buffer("betas", betas)

        # extracting alpha values
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        sqrt_recip_alphas = (1.0 / alphas) ** 0.5
        self.register_buffer("sqrt_recip_alphas", sqrt_recip_alphas)

        # diffusion
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod**0.5)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod) ** 0.5)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

    def _get_beta_schedule(self) -> torch.Tensor:
        """
        Get beta schedule given schedule_type
        """

        def cosine_beta_schedule():
            x = torch.linspace(0, self.timesteps, self.timesteps + 1)
            alphas_hat = torch.cos(((x / self.timesteps) + 0.008) / (1.008) * torch.pi * 0.5) ** 2
            alphas_hat = alphas_hat / alphas_hat[0]
            betas = 1 - (alphas_hat[1:] / alphas_hat[:-1])
            return torch.clip(betas, 0.0001, 0.9999)

        def linear_beta_schedule():
            return torch.linspace(self.beta_start, self.beta_end, self.timesteps)

        def quadratic_beta_schedule():
            return torch.linspace(self.beta_start**0.5, self.beta_end**0.5, self.timesteps) ** 2

        def sigmoid_beta_schedule():
            betas = torch.linspace(-6, 6, self.timesteps)
            return torch.sigmoid(betas) * (self.beta_end - self.beta_start) + self.beta_start

        schedules = {
            "cosine": cosine_beta_schedule,
            "linear": linear_beta_schedule,
            "quadratic": quadratic_beta_schedule,
            "sigmoid": sigmoid_beta_schedule,
        }
        beta_schedule = schedules[self.schedule_type]

        return beta_schedule()

    def forward(self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Diffuse data given diffusion timesteps

        Args:
            x (torch.Tensor): data to diffuse
            t (torch.Tensor): timesteps
            noise (torch.Tensor, optional): noise used for diffusion. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (diffused image, noise to predict)
        """
        if noise is None:
            noise = torch.randn_like(x)

        # get diffusion values for each batch
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]

        x_hat = sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise

        return x_hat, noise
