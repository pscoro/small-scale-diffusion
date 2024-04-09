from pathlib import Path
from typing import Tuple

import lightning
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Food101
from torchvision.transforms import v2

from noise_scheduler import NoiseScheduler
from diffusion import Diffusion
from unet_model import UNet


DIFFUSION_STEPS = 1000


class DiffusionModel(lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self.scheduler = NoiseScheduler()
        self.model = UNet(3, 64)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(x, t)

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        images = batch[0]

        timesteps = torch.randint(0, DIFFUSION_STEPS, (images.shape[0],), device=images.device)
        noise = torch.randn_like(images)
        noisy_images = self.scheduler.add_noise(images, noise, timesteps)

        noise_pred = self.model(noisy_images, timesteps)

        loss = F.mse_loss(noise_pred, noise)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


def load_dataset() -> Dataset:
    train_transform = v2.Compose([
        v2.RandomResizedCrop(size=(128, 128), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.ToTensor(),
        v2.ToDtype(torch.bfloat16, scale=True),
        v2.Lambda(lambda x: x * 2 - 1),
    ])
    train_dataset = Food101(
        root="./food101",
        split="train",
        download=True,
        transform=train_transform
    )
    return train_dataset


def main() -> None:
    train_dataset = load_dataset()
    dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )

    model = DiffusionModel()

    trainer = lightning.Trainer(
        max_epochs=10,
        accelerator="cuda",
        devices=2,
        strategy="ddp",
        precision="bf16-mixed",
    )
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()
