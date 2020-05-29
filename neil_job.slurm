#!/bin/bash
#SBATCH -J NeilGAN/style
#SBATCH --partition=gpu
#SBATCH --qos=12c-1h_2gpu
#SBATCH --gres=gpu:0,gpu:1
#SBATCH --output=neil_job.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
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

#import necessary modules
#conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
#conda install tensorflow
#conda install matplotlib
#conda install opencv

# execute your application
srun python style_main.py --coare=1 --img_to_load=-1
source deactivate