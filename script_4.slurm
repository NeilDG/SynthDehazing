#!/bin/bash
#SBATCH -J DH_12
#SBATCH --partition=gpu
#SBATCH --qos=12c-1h_2gpu
#SBATCH --cpus-per-task=4
#SBATCH -G 1 
#SBATCH --ntasks=1
#SBATCH --output=dn_job_12.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph 
#SBATCH --mail-type=END 

# check which GPU device was allocated
echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

# prepare working environment
module load anaconda
module load cuda/10.1_cudnn-7.6.5

source activate NeilGAN

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install -c conda-forge scikit-image
srun python hybrid_dehaze_main.py --iteration=12 --coare=1 --clarity_weight=1000.0 --adv_weight=4.0 --img_to_load=119900 --load_previous=0
source deactivate

for i in {1..500}
    do
        source activate NeilGAN
        srun python hybrid_dehaze_main.py --iteration=12 --coare=1 --clarity_weight=1000.0 --adv_weight=4.0 --img_to_load=119900 --load_previous=1
        source deactivate
    done