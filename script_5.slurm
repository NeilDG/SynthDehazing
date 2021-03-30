#!/bin/bash
#SBATCH -J LIGHT_9
#SBATCH --partition=gpu
#SBATCH --qos=12c-1h_2gpu
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --ntasks=1
#SBATCH --output=script_9.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph
#SBATCH --mail-type=END

ITERATION=9

#About this script:
# CycleGAN using new dataset between synth hazy and places dataset
echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

# prepare working environment
module load anaconda
module load cuda/10.1_cudnn-7.6.5

source activate NeilGAN_V2

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
#conda install -c conda-forge scikit-image

srun python airlight_main.py --iteration=$ITERATION --coare=1 --d_lr=0.00005 --img_to_load=-1 --batch_size=32 --load_previous=0
conda deactivate

for i in {1..25}
    do
        source activate NeilGAN_V2
        srun python airlight_main.py --iteration=$ITERATION --coare=1 --d_lr=0.00005 --img_to_load=-1 --batch_size=32 --load_previous=1
        conda deactivate
    done