from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import torch
from torchvision.utils import save_image

# Load the trained model
model = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=3)
diffusion = GaussianDiffusion(model, image_size=128, timesteps=1000)

# Load checkpoint
data = torch.load('./results/model-100.pt')
diffusion.load_state_dict(data['model'])

# Generate 16 new car images!
samples = diffusion.sample(batch_size=16)
save_image(samples, 'generated_cars.png', nrow=4)