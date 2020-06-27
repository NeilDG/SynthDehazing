#!/bin/bash
#SBATCH -J NeilDN
#SBATCH --partition=gpu
#SBATCH --qos=12c-1h_2gpu
#SBATCH -G 1 
#SBATCH --ntasks=1
#SBATCH --output=neil_job.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph 
#SBATCH --mail-type=END 

# check which GPU device was allocated
echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

# prepare working environment
module load anaconda
module load cuda/10.1_cudnn-7.6.5

g_skip=10
d_skip=10
iteration=7

echo "g_skip=$g_skip d_skip=$d_skip iteration=$iteration"

# activate your python environment
source activate NeilGAN
srun python denoise_main.py --img_to_load=72000 --coare=1 --gen_skips=$g_skip --disc_skips=$d_skip --style_iteration=$iteration
source deactivate