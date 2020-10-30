#!/bin/bash
#SBATCH -J PATCH_4
#SBATCH --partition=gpu
#SBATCH --qos=12c-1h_2gpu
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --ntasks=1
#SBATCH --output=patch_4.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph
#SBATCH --mail-type=END

ITERATION=5

# check which GPU device was allocated
echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

# prepare working environment
module load anaconda
module load cuda/10.1_cudnn-7.6.5

source activate NeilGAN

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install -c conda-forge scikit-image
srun python hybrid_dehaze_main.py --iteration=$ITERATION --clarity_weight=500.0 --color_weight=1000.0 ----cycle_weight=100.0 --coare=1 --img_to_load=-1 --load_previous=0
conda deactivate

for i in {1..500}
    do
        source activate NeilGAN
        srun python hybrid_dehaze_main.py --iteration=$ITERATION --clarity_weight=500.0 --color_weight=1000.0 ----cycle_weight=100.0 --coare=1 --img_to_load=-1 --load_previous=1
        conda deactivate
    done