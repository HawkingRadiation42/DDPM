#!/usr/bin/env python3
"""Generate samples from a trained checkpoint (with flash attention disabled)"""

import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from torchvision import utils
import math
from pathlib import Path
import sys

def generate_samples(checkpoint_path='./results/model-1.pt', num_samples=25, batch_size=4):
    """Generate samples from a checkpoint with flash attention disabled."""

    print(f"\n{'='*70}")
    print(f"{'GENERATING SAMPLES':^70}")
    print(f"{'='*70}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Num samples: {num_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"{'='*70}\n")

    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model architecture (must match training config)
    print("\nLoading model architecture...")
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        flash_attn=False  # Disable flash attention for sampling
    ).to(device)

    diffusion = GaussianDiffusion(
        model,
        image_size=128,
        timesteps=1000,
        sampling_timesteps=250,
        objective='pred_noise'
    )

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    data = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Load model weights
    if 'ema' in data:
        print("Loading EMA model weights...")
        from ema_pytorch import EMA
        ema = EMA(diffusion, beta=0.995, update_every=10)
        ema.load_state_dict(data['ema'])
        model_to_sample = ema.ema_model
    else:
        print("Loading model weights...")
        diffusion.load_state_dict(data['model'])
        model_to_sample = diffusion

    model_to_sample.eval()

    print(f"\nStep: {data.get('step', 'unknown')}")
    print(f"Version: {data.get('version', 'unknown')}")

    # Generate samples
    print(f"\nGenerating {num_samples} samples...")
    print("This may take a few minutes...\n")

    with torch.inference_mode():
        from denoising_diffusion_pytorch.denoising_diffusion_pytorch import num_to_groups

        batches = num_to_groups(num_samples, batch_size)
        all_images_list = []

        for i, n in enumerate(batches):
            print(f"  Batch {i+1}/{len(batches)}: generating {n} samples...")
            batch_images = model_to_sample.sample(batch_size=n)
            all_images_list.append(batch_images)

        all_images = torch.cat(all_images_list, dim=0)

    # Save images
    output_path = Path(checkpoint_path).parent / 'generated_samples.png'
    print(f"\nSaving samples to {output_path}...")
    utils.save_image(all_images, str(output_path), nrow=int(math.sqrt(num_samples)))

    print(f"\n{'='*70}")
    print("✓ Generation complete!")
    print(f"  Output: {output_path}")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate samples from checkpoint')
    parser.add_argument('--checkpoint', type=str, default='./results/model-1.pt',
                       help='Path to checkpoint file')
    parser.add_argument('--num_samples', type=int, default=25,
                       help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for generation')

    args = parser.parse_args()

    generate_samples(args.checkpoint, args.num_samples, args.batch_size)
