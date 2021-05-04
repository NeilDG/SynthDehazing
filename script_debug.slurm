#!/bin/bash
#SBATCH -J ALBEDO_DEBUG
#SBATCH --partition=debug
#SBATCH --qos=240c-1h_debug
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --output=script_debug.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph
#SBATCH --mail-type=END

ITERATION=2

#About this script:
# CycleGAN using new dataset between synth hazy and places dataset
echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

# prepare working environment
module load anaconda
module load cuda/10.1_cudnn-7.6.5

source activate NeilGAN_V2

srun python cyclegan_main.py --iteration=$ITERATION --coare=1 --likeness_weight=1.0 --psnr_loss_weight=10.0 --use_psnr=1 --img_to_load=-1 --load_previous=0
conda deactivate