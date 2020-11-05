#!/bin/bash
#SBATCH -J RESIDE_1
#SBATCH --partition=gpu
#SBATCH --qos=12c-1h_2gpu
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --ntasks=1
#SBATCH --output=color_1.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph 
#SBATCH --mail-type=END 

ITERATION=2

# check which GPU device was allocated
echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

# prepare working environment
module load anaconda
module load cuda/10.1_cudnn-7.6.5

source activate NeilGAN

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install -c conda-forge scikit-image
srun python div2k_main.py --iteration=$ITERATION --coare=1 --identity_weight=1.0 --likeness_weight=1.0 --cycle_weight=50.0 --img_to_load=-1 --load_previous=0
source deactivate

for i in {1..300}
    do
        source activate NeilGAN
        srun python div2k_main.py --iteration=$ITERATION --coare=1 --identity_weight=1.0 --likeness_weight=1.0 --cycle_weight=50.0 --img_to_load=-1 --load_previous=1
        source deactivate
    done