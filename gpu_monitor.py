"""GPU Memory Monitoring Utilities for Training"""

import torch
from collections import deque
import threading
import time

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


class GPUMemoryMonitor:
    """Tracks GPU memory usage: current, peak, and average in real-time."""

    def __init__(self, device, update_interval=1.0):
        self.device = device
        self.is_cuda = device.type == "cuda"
        self.update_interval = update_interval
        self.memory_history = deque(maxlen=1000)  # Rolling average
        self.peak_memory = 0
        self.monitoring = False
        self.monitor_thread = None
        self.nvml_handle = None

        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(device)

            # Initialize NVML for GPU utilization
            if NVML_AVAILABLE:
                try:
                    pynvml.nvmlInit()
                    # Get device index properly
                    if hasattr(device, 'index') and device.index is not None:
                        device_idx = device.index
                    else:
                        # For cuda:0 device, extract the number
                        device_idx = torch.cuda.current_device()
                    self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
                except Exception as e:
                    print(f"Warning: Could not initialize NVML for GPU utilization: {e}")
                    print("GPU stats will show memory only (no utilization %)")
                    self.nvml_handle = None

    def update(self):
        """Update memory statistics."""
        if not self.is_cuda:
            return

        current = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
        peak = torch.cuda.max_memory_allocated(self.device) / 1024**3  # GB

        self.memory_history.append(current)
        self.peak_memory = max(self.peak_memory, peak)

    def get_stats(self):
        """Get current memory statistics including GPU utilization."""
        if not self.is_cuda:
            return {"current": 0, "peak": 0, "average": 0, "total": 0, "utilization": 0, "memory_percent": 0}

        # Use reserved memory (includes cached) for more accurate representation
        current = torch.cuda.memory_reserved(self.device) / 1024**3  # GB
        peak = torch.cuda.max_memory_reserved(self.device) / 1024**3  # GB
        total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3  # GB
        average = sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0
        memory_percent = (current / total * 100) if total > 0 else 0

        # Get GPU utilization via NVML (and memory from NVML if available)
        utilization = 0
        nvml_memory_used = None
        if self.nvml_handle:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                utilization = util.gpu
                # Get actual memory usage from NVML (more accurate)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                nvml_memory_used = mem_info.used / 1024**3  # GB
            except Exception:
                pass

        # Prefer NVML memory if available (more accurate than PyTorch reserved)
        if nvml_memory_used is not None:
            current = nvml_memory_used
            memory_percent = (current / total * 100) if total > 0 else 0

        return {
            "current": current,
            "peak": peak,
            "average": average,
            "total": total,
            "utilization": utilization,
            "memory_percent": memory_percent
        }

    def get_stats_string(self):
        """Get formatted stats string for tqdm."""
        stats = self.get_stats()
        return (f"GPU: {stats['utilization']:>3.0f}% | "
                f"VRAM: {stats['current']:.1f}/{stats['total']:.1f}GB ({stats['memory_percent']:.1f}%)")

    def print_stats(self):
        """Print formatted memory statistics."""
        stats = self.get_stats()
        if not self.is_cuda:
            print("GPU memory monitoring not available (not using CUDA)")
            return

        print(f"\n{'='*70}")
        print(f"{'GPU MEMORY STATISTICS':^70}")
        print(f"{'='*70}")
        print(f"  Total VRAM:    {stats['total']:>6.2f} GB")
        print(f"  Current:       {stats['current']:>6.2f} GB  ({stats['current']/stats['total']*100:>5.1f}%)")
        print(f"  Peak:          {stats['peak']:>6.2f} GB  ({stats['peak']/stats['total']*100:>5.1f}%)")
        print(f"  Average:       {stats['average']:>6.2f} GB  ({stats['average']/stats['total']*100:>5.1f}%)")
        print(f"  Available:     {stats['total']-stats['current']:>6.2f} GB")
        print(f"{'='*70}\n")

    def start_background_monitoring(self):
        """Start background thread for continuous monitoring."""
        if not self.is_cuda or self.monitoring:
            return

        self.monitoring = True

        def monitor_loop():
            while self.monitoring:
                self.update()
                time.sleep(self.update_interval)

        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_background_monitoring(self):
        """Stop background monitoring thread."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

    def __del__(self):
        """Cleanup NVML on deletion."""
        if self.nvml_handle and NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def print_gpu_info(device):
    """Print detailed GPU information."""
    if device.type != "cuda":
        print("Not using CUDA GPU")
        return

    props = torch.cuda.get_device_properties(device)

    print(f"\n{'='*70}")
    print(f"{'GPU CONFIGURATION':^70}")
    print(f"{'='*70}")
    print(f"  Device:           {torch.cuda.get_device_name(device)}")
    print(f"  CUDA Version:     {torch.version.cuda}")
    print(f"  Total VRAM:       {props.total_memory / 1024**3:.2f} GB")
    print(f"  Compute Cap:      {props.major}.{props.minor}")
    print(f"  Multi-Processors: {props.multi_processor_count}")
    print(f"{'='*70}\n")
