import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import time
import sys
from gpu_monitor import GPUMemoryMonitor, print_gpu_info


# ---------- device selection ----------
def get_device():
    """Select CUDA GPU if available, otherwise exit."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        return device
    else:
        print("\n" + "="*70)
        print("ERROR: No CUDA GPU detected!")
        print("This script is optimized for CUDA GPUs only.")
        print("Please run on a machine with NVIDIA GPU and CUDA installed.")
        print("="*70 + "\n")
        sys.exit(1)


# ========== Enhanced Trainer with Memory Monitoring ==========
class MonitoredTrainer:
    """Wraps the Trainer with real-time GPU memory monitoring in tqdm bar."""

    def __init__(self, trainer, memory_monitor):
        self.trainer = trainer
        self.memory_monitor = memory_monitor
        self.start_time = None
        self._patch_trainer()

    def _patch_trainer(self):
        """Monkey-patch the trainer's train method to inject GPU stats into tqdm."""
        def patched_train():
            """Patched train method that updates tqdm with GPU stats."""
            accelerator = self.trainer.accelerator
            device = accelerator.device

            # Import tqdm here to match the library
            from tqdm import tqdm

            with tqdm(initial=self.trainer.step, total=self.trainer.train_num_steps,
                     disable=not accelerator.is_main_process) as pbar:

                while self.trainer.step < self.trainer.train_num_steps:
                    self.trainer.model.train()
                    total_loss = 0.

                    for _ in range(self.trainer.gradient_accumulate_every):
                        data = next(self.trainer.dl).to(device)

                        with self.trainer.accelerator.autocast():
                            loss = self.trainer.model(data)
                            loss = loss / self.trainer.gradient_accumulate_every
                            total_loss += loss.item()

                        self.trainer.accelerator.backward(loss)

                    # Update tqdm with loss AND GPU stats
                    gpu_stats = self.memory_monitor.get_stats_string()
                    pbar.set_description(f'loss: {total_loss:.4f}')
                    pbar.set_postfix_str(gpu_stats)

                    accelerator.wait_for_everyone()
                    accelerator.clip_grad_norm_(self.trainer.model.parameters(), self.trainer.max_grad_norm)

                    self.trainer.opt.step()
                    self.trainer.opt.zero_grad()

                    accelerator.wait_for_everyone()

                    self.trainer.step += 1
                    if accelerator.is_main_process:
                        self.trainer.ema.update()

                        if self.trainer.step != 0 and self.trainer.step % self.trainer.save_and_sample_every == 0:
                            milestone = self.trainer.step // self.trainer.save_and_sample_every

                            # ALWAYS save checkpoint first (before sampling which might crash)
                            accelerator.print(f"Saving checkpoint at step {self.trainer.step}...")
                            self.trainer.save(milestone)

                            # Now try sampling (safe to fail - checkpoint already saved)
                            self.trainer.ema.ema_model.eval()

                            with torch.inference_mode():
                                from denoising_diffusion_pytorch.denoising_diffusion_pytorch import num_to_groups

                                # Temporarily disable flash attention for sampling to avoid kernel issues
                                # This is necessary because sampling uses different dtypes than training
                                def set_flash_attn(model, enable):
                                    """Recursively enable/disable flash attention in all attention modules."""
                                    for module in model.modules():
                                        if hasattr(module, 'flash'):
                                            module.flash = enable

                                set_flash_attn(self.trainer.ema.ema_model, False)

                                try:
                                    accelerator.print("Generating samples...")
                                    batches = num_to_groups(self.trainer.num_samples, self.trainer.batch_size)
                                    all_images_list = list(map(lambda n: self.trainer.ema.ema_model.sample(batch_size=n), batches))
                                    accelerator.print("✓ Sample generation successful")
                                except Exception as e:
                                    accelerator.print(f"✗ Sampling failed: {e}")
                                    accelerator.print("  Checkpoint saved, skipping sample generation")
                                    all_images_list = None
                                finally:
                                    # Always restore flash attention for training
                                    set_flash_attn(self.trainer.ema.ema_model, True)

                            # Only save images if sampling succeeded
                            if all_images_list is not None:
                                all_images = torch.cat(all_images_list, dim=0)

                                from torchvision import utils
                                import math
                                utils.save_image(all_images, str(self.trainer.results_folder / f'sample-{milestone}.png'),
                                               nrow=int(math.sqrt(self.trainer.num_samples)))
                                accelerator.print(f"✓ Saved samples to sample-{milestone}.png")

                                if self.trainer.calculate_fid:
                                    try:
                                        fid_score = self.trainer.fid_scorer.fid_score()
                                        accelerator.print(f'fid_score: {fid_score}')
                                    except Exception as e:
                                        accelerator.print(f"FID calculation failed: {e}")

                                if self.trainer.save_best_and_latest_only:
                                    if self.trainer.best_fid > fid_score:
                                        self.trainer.best_fid = fid_score
                                        self.trainer.save("best")
                                    self.trainer.save("latest")

                    pbar.update(1)

            accelerator.print('training complete')

        self.trainer.train = patched_train

    def train(self):
        """Train with background memory monitoring and detailed logging."""
        self.start_time = time.time()

        # Print training configuration
        print(f"\n{'='*70}")
        print(f"{'STARTING TRAINING':^70}")
        print(f"{'='*70}")
        print(f"  Training steps:   {self.trainer.train_num_steps:,}")
        print(f"  Batch size:       {self.trainer.batch_size}")
        print(f"  Gradient accum:   {self.trainer.gradient_accumulate_every}")
        print(f"  Effective batch:  {self.trainer.batch_size * self.trainer.gradient_accumulate_every}")
        print(f"  Learning rate:    {self.trainer.opt.param_groups[0]['lr']:.2e}")
        print(f"  Save interval:    Every {self.trainer.save_and_sample_every} steps")
        print(f"  AMP enabled:      {self.trainer.accelerator.mixed_precision != 'no'}")
        print(f"{'='*70}\n")

        # Start background memory monitoring
        print("Starting GPU memory monitoring...")
        self.memory_monitor.start_background_monitoring()

        try:
            # Run patched training with GPU stats in tqdm
            self.trainer.train()
        except KeyboardInterrupt:
            print("\n\n" + "="*70)
            print("Training interrupted by user!")
            print("="*70)
        except Exception as e:
            print(f"\n\n" + "="*70)
            print(f"Training failed with error: {e}")
            print("="*70)
            raise
        finally:
            # Stop monitoring and print final stats
            self.memory_monitor.stop_background_monitoring()
            self.print_final_stats()

    def print_final_stats(self):
        """Print final training statistics."""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)

        print(f"\n{'='*70}")
        print(f"{'TRAINING SUMMARY':^70}")
        print(f"{'='*70}")
        print(f"  Total training time:  {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"{'='*70}\n")

        self.memory_monitor.print_stats()


if __name__ == '__main__':
    # ---------- device setup ----------
    device = get_device()
    print_gpu_info(device)

    # Initialize memory monitor
    memory_monitor = GPUMemoryMonitor(device)

    # ---------- model ----------
    print("Loading model...")
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        flash_attn=True  # Enable flash attention on CUDA
    ).to(device)

    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    memory_monitor.update()
    stats = memory_monitor.get_stats()
    print(f"Model VRAM usage: {stats['current']:.2f} GB\n")

    # ---------- diffusion ----------
    diffusion = GaussianDiffusion(
        model,
        image_size=64,
        timesteps=1000,
        sampling_timesteps=250,
        objective='pred_noise'  # L2 loss on noise prediction
    )

    # ---------- monkey-patch cpu_count to limit workers ----------
    # Set to 0 to use main process - avoids DataLoader bottleneck on HPC
    import denoising_diffusion_pytorch.denoising_diffusion_pytorch as ddp_module
    original_cpu_count_fn = ddp_module.cpu_count
    ddp_module.cpu_count = lambda: 16  # Use main process for data loading

    # ---------- trainer ----------
    trainer = Trainer(
        diffusion,
        folder='./sysu-shape-dataset/combined/',
        train_batch_size=256,  # Increased - monitor VRAM in tqdm bar
        train_lr=1e-4,
        train_num_steps=30000,
        gradient_accumulate_every=1,
        ema_decay=0.995,
        amp=False,  # Enable mixed precision on CUDA
        save_and_sample_every=2000,
        results_folder='./results_combinedV2',
        num_samples=25,
        calculate_fid=True,
    )

    # Restore original cpu_count
    ddp_module.cpu_count = original_cpu_count_fn

    # ---------- run with monitoring ----------
    print("Initializing monitored trainer...\n")
    monitored_trainer = MonitoredTrainer(trainer, memory_monitor)
    monitored_trainer.train()
