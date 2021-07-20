#!/bin/bash
#SBATCH -J DZ_2
#SBATCH --partition=gpu
#SBATCH --qos=12c-1h_2gpu
#SBATCH --cpus-per-task=6
#SBATCH -G 1
#SBATCH --ntasks=1
#SBATCH --output=script_2.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph
#SBATCH --mail-type=END

ITERATION=2

echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

# prepare working environment
module load anaconda
module load cuda/10.1_cudnn-7.6.5

source activate NeilGAN_V2

srun python dehaze_main.py --iteration=$ITERATION --server_config=1 --num_workers=6 --likeness_weight=10.0 --edge_weight=10.0 --img_to_load=-1 --load_previous=0 --batch_size=32
conda deactivate