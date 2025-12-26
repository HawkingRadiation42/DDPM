# DDPM Implementation on SysU-Shape Dataset
**Course:** MLIA (Machine Learning in Image Analysis)  
**Project:** Denoising Diffusion Probabilistic Models (DDPM)



---

## 📝 Abstract
This project implements a Denoising Diffusion Probabilistic Model (DDPM) for unconditional image synthesis on the SYSU-Shapes dataset. We investigate the impact of model capacity, learning rate, and data augmentation on generation quality. Through an ablation study of five configurations, we found that increasing model capacity (from 64 to 128 dimensions) significantly stabilized training and improved results, countering overfitting observed in smaller models. Our best configuration achieved a Fréchet Inception Distance (FID) of **59.01** and an Inception Score (IS) of **6.57**.

---

## 📂 Project Structure

```
├── train.py                  # Baseline training script (Dim 64, No Aug)
├── train_exp1_low_lr.py      # Exp 1: Low Learning Rate (2e-5)
├── train_exp2_hybrid.py      # Exp 2: Low LR + Mild Augmentation
├── train_exp3_deeper.py      # Exp 3: Deeper Model (Dim 96)
├── train_exp4_dim128.py      # Exp 4: Best Model (Dim 128 + Aug)
│
├── run.sbatch                # SLURM script for baseline
├── run_exp[1-4].sbatch       # SLURM scripts for experiments
│
├── evaluation_scripts/       # FID/IS calculation and plotting
│   ├── eval_utils.py         # Shared evaluation logic
│   ├── plot_all_results.py   # Generates comparison plots
│   └── run_eval_*.sbatch     # Evaluation job scripts
│
├── plot_training_loss.py     # Generate loss curves
├── gpu_monitor.py            # GPU usage tracking
└── requirements.txt          # Dependencies
```

> **Note:** Full result logs and checkpoints are stored on Rivanna shared storage due to size constraints.

---

## 📊 Experiments & Methodology

We utilized the **SYSU-Shapes Dataset** (1,541 images across 5 categories: Airplanes, Bicycles, Boats, Cars, Motorbikes), resized to $64 \times 64$.

We conducted 5 experiments to optimize generation quality:

| Experiment ID | Dim | LR | Augmentation | Key Finding |
|--------------|-----|----|--------------|-------------|
| **Baseline** | 64 | 1e-4 | None | Early plateau, unstable loss. |
| **Exp 1 (Low LR)** | 64 | 2e-5 | None | Improved stability, but still limited capacity. |
| **Exp 2 (Low LR+Aug)** | 64 | 2e-5 | H-Flip | **Worst performance**. Model too small for augmentation. |
| **Exp 3 (Dim 96)** | 96 | 1e-4 | H-Flip | Significant improvement in loss and FID. |
| **Exp 4 (Dim 128)** | **128** | 1e-4 | **H-Flip** | **Best Result**. Stable convergence, sharpest images. |

### Training Details
- **Architecture**: U-Net with Group Normalization and Self-Attention (16x16).
- **Steps**: 100,000 iterations (Baseline stopped at 70k).
- **Batch Size**: 64.
- **Optimizer**: Adam.
- **EMA**: Decay rate 0.995.

---

## 📈 Results

### Quantitative Metrics
| Metric | Baseline (Dim 64) | Best Model (Dim 128) |
|--------|-------------------|----------------------|
| **FID** (Lower is better) | ~110 | **59.01** |
| **IS** (Higher is better) | ~5.4 | **6.57** |

### Qualitative Analysis
- **Dim 64 Models**: Produced blurry, "dream-like" shapes and suffered from mode collapse/memorization.
- **Dim 128 Model**: Generated sharp, coherent objects with distinct structural details (e.g., wings, wheels) and clear background separation.

*(See `Report.pdf` for full loss curves and generated samples)*

---

## 🚀 Usage Instructions

### 1. Setup Environment
Ensure you are on a GPU node.
```bash
module load miniforge/24.11.3-py3.12
module load cuda/12.8.0
module load cudnn/9.8.0-CUDA-12.8.0

pip install -r requirements.txt
```

### 2. Training
To train the best model (Experiment 4):
```bash
sbatch run_exp4.sbatch
```
To train other versions, use their respective `sbatch` scripts (e.g., `sbatch run.sbatch` for baseline).

### 3. Evaluation
To compute FID and IS for a trained model:
```bash
cd evaluation_scripts
sbatch run_eval_exp4.sbatch
```
This produces `evaluation_results.json` in the results directory.

### 4. Visualization
To generate comparison plots (FID/IS curves):
```bash
cd evaluation_scripts
python plot_all_results.py
```
Outputs: `comparison_fid.png`, `comparison_is.png`.

To generate loss curves:
```bash
python plot_training_loss.py
```
Outputs saved in `loss_plots/`.

---

## 🔮 Future Work
- **Classifier-Free Guidance**: To improve class separation.
- **Cascade Diffusion**: Super-resolution (64x64 -> 128x128) to add fine textures.
- **Latent Diffusion**: To scale to higher resolutions more efficiently.
