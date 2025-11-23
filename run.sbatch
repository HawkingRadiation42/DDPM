#!/bin/bash
#SBATCH -J mlia
#SBATCH -p gpu
#SBATCH -A orsdardencomputing
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH -o logs/mlia.out
#SBATCH -e logs/mlia.err
#SBATCH --mail-type=ALL                # Mail events (END, FAIL)
#SBATCH --mail-user=zjy6us@virginia.edu     # Where to send mail


cd /home/zjy6us/mlia


module load miniforge/24.11.3-py3.12
module load cuda/12.8.0
module load cudnn/9.8.0-CUDA-12.8.0

source .venv/bin/activate

python train.py 

