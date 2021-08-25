#!/bin/bash
#SBATCH -J CYCLE_DB
#SBATCH --partition=debug
#SBATCH --qos=240c-1h_debug
#SBATCH --ntasks=1
#SBATCH --output=script_debug.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph
#SBATCH --mail-type=END

ITERATION=2

echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

# prepare working environment
module load anaconda
module load cuda/10.1_cudnn-7.6.5

source activate NeilGAN_V2

srun python cyclegan_main.py \
--iteration=$ITERATION \
--server_config=1 \
--num_workers=6 \
--batch_size=512 \
--img_to_load=-1 --load_previous=0

conda deactivate