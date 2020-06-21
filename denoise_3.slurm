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

i=1.0
c=100.0
a=1.0
t=100.0
iteration=3

echo "c=$c a=$a t=$t iteration=$iteration"

# activate your python environment
source activate NeilGAN
srun python denoise_main.py --coare=1 --img_to_load=-1 --style_iteration=$iteration --identity_weight=$i --cycle_weight=$c --adv_weight=$a --tv_weight=$t
source deactivate