#!/bin/bash
#SBATCH -J NeilDN
#SBATCH --partition=gpu
#SBATCH --qos=12c-1h_2gpu 
#SBATCH -G 1 
#SBATCH --ntasks=1
#SBATCH --output=dn_job_2.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph 
#SBATCH --mail-type=END 

# check which GPU device was allocated
echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

# prepare working environment
module load anaconda
module load cuda/10.1_cudnn-7.6.5

# activate your python environment
source activate NeilGAN
srun python denoise_main.py --style_iteration=11 --coare=1 --img_to_load=72296 --cycle_weight=100.0 --identity_weight=1.0 --tv_weight=3.0 --adv_weight=1.0 --load_previous=0
source deactivate