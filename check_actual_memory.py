#!/usr/bin/env python3
"""Quick script to check actual GPU memory during training"""

import subprocess
import time

print("Checking GPU memory every 2 seconds (Ctrl+C to stop)...")
print("="*80)

try:
    while True:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            output = result.stdout.strip()
            mem_used, mem_total, gpu_util = output.split(',')
            mem_used_gb = float(mem_used) / 1024
            mem_total_gb = float(mem_total) / 1024
            mem_percent = (float(mem_used) / float(mem_total)) * 100

            print(f"GPU: {gpu_util.strip():>3}% | "
                  f"VRAM: {mem_used_gb:>5.1f}/{mem_total_gb:>5.1f}GB ({mem_percent:>5.1f}%) | "
                  f"{time.strftime('%H:%M:%S')}")
        else:
            print("Error running nvidia-smi")

        time.sleep(2)

except KeyboardInterrupt:
    print("\n\nStopped.")
