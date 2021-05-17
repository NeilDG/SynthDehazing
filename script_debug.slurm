#!/bin/bash
#SBATCH -J TR_DB
#SBATCH --partition=debug
#SBATCH --qos=240c-1h_debug
#SBATCH --ntasks=1
#SBATCH --output=script_debug.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph
#SBATCH --mail-type=END

ITERATION=7

#About this script:
# CycleGAN using new dataset between synth hazy and places dataset
echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

# prepare working environment
module load anaconda
module load cuda/10.1_cudnn-7.6.5

source activate NeilGAN_V2

srun python transmission_main.py --iteration=$ITERATION --coare=1 --use_disc_l1=0 --edge_weight=2.0 --img_to_load=-1 --load_previous=0
conda deactivate