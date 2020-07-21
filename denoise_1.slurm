#!/bin/bash
#SBATCH -J GTA_DN
#SBATCH --partition=gpu
#SBATCH --qos=12c-1h_2gpu 
#SBATCH -G 1 
#SBATCH --ntasks=1
#SBATCH --output=dn_job_1.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph 
#SBATCH --mail-type=END 

# check which GPU device was allocated
echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

# prepare working environment
module load anaconda
module load cuda/10.1_cudnn-7.6.5

for i in {1..500}
    do
        source activate NeilGAN
        srun python denoise_main.py --iteration=1 --coare=1 --img_to_load=72296 --identity_weight=10.0 --likeness_weight=1.0 --adv_weight=1.0 --load_previous=1
        source deactivate
    done