#!/bin/bash
#SBATCH -J AL_6
#SBATCH --partition=gpu
#SBATCH --qos=12c-1h_2gpu
#SBATCH --cpus-per-task=6
#SBATCH -G 1
#SBATCH --ntasks=1
#SBATCH --output=script_6.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph
#SBATCH --mail-type=END

ITERATION=6

#About this script:
# CycleGAN using new dataset between synth hazy and places dataset
echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

# prepare working environment
module load anaconda
module load cuda/10.1_cudnn-7.6.5

source activate NeilGAN_V2

srun python albedo_main.py --iteration=$ITERATION --coare=1 --likeness_weight=10.0 --psnr_loss_weight=0.0 --adv_weight=2.5 --g_lr=0.0005 --d_lr=0.0005 --img_to_load=-1 --load_previous=0
conda deactivate