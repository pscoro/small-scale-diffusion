import os
import torch
from PIL import Image
from tqdm import tqdm

from diffusion import Diffusion
from unet_model import UNet
from noise_scheduler import NoiseScheduler

def load_model() -> UNet:
    model = UNet(3, 64).eval()
    weights = torch.load("diffusion.pt", map_location=torch.device("cpu"))
    weights = {
        k[len("model."):]: v
        for (k, v) in weights["state_dict"].items()
        if k.startswith("model")
    }
    model.load_state_dict(weights)
    return model


@torch.no_grad()
def main() -> None:
    model = load_model().cuda()

    scheduler = NoiseScheduler().cuda()

    os.makedirs("final", exist_ok=True)
    inception_scores = []
    fid_scores = []
    for i in range(10):
        noise = torch.randn(1, 3, 128, 128).cuda()
        for timestep in tqdm(reversed(range(1000))):
            noise_pred = model(noise, torch.tensor([timestep], dtype=torch.long).cuda())
            noise = scheduler.step(noise_pred, timestep, noise)

        print(noise)
        noise = noise.cpu()[0].permute(1, 2, 0).numpy()
        noise = (noise.clip(-1, 1) + 1) / 2
        noise = (255 * noise).astype("uint8")
        img = Image.fromarray(noise)
        img.resize((512, 512)).save(f"final/{i}.png")

if __name__ == "__main__":
    main()
