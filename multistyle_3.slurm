#!/bin/bash
#SBATCH -J NeilMST
#SBATCH --partition=gpu
#SBATCH --qos=12c-1h_2gpu
#SBATCH -G 1 
#SBATCH --ntasks=1
#SBATCH --output=neil_job.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph 
#SBATCH --mail-type=END 

# check which GPU device was allocated
echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

# prepare working environment
module load anaconda
module load cuda/10.1_cudnn-7.6.5

c=10.0
s=100.0
iteration=3

echo "content_weight=$c style_weight=$s iteration=$iteration"

# activate your python environment
source activate NeilGAN
srun python multistyle_main_slurm.py --coare=1 --img_to_load=-1 --style_iteration=$iteration --content_weight=$c --style_weight=$s
source deactivate

#simulate 20 epoch entry due to dataloader bug
source activate NeilGAN
srun python multistyle_main_slurm.py --coare=1 --img_to_load=-1 --load_previous=1 --style_iteration=$iteration --content_weight=$c --style_weight=$s
source deactivate

source activate NeilGAN
srun python multistyle_main_slurm.py --coare=1 --img_to_load=-1 --load_previous=1 --style_iteration=$iteration --content_weight=$c --style_weight=$s
source deactivate

source activate NeilGAN
srun python multistyle_main_slurm.py --coare=1 --img_to_load=-1 --load_previous=1 --style_iteration=$iteration --content_weight=$c --style_weight=$s
source deactivate

source activate NeilGAN
srun python multistyle_main_slurm.py --coare=1 --img_to_load=-1 --load_previous=1 --style_iteration=$iteration --content_weight=$c --style_weight=$s
source deactivate

source activate NeilGAN
srun python multistyle_main_slurm.py --coare=1 --img_to_load=-1 --load_previous=1 --style_iteration=$iteration --content_weight=$c --style_weight=$s
source deactivate

source activate NeilGAN
srun python multistyle_main_slurm.py --coare=1 --img_to_load=-1 --load_previous=1 --style_iteration=$iteration --content_weight=$c --style_weight=$s
source deactivate

source activate NeilGAN
srun python multistyle_main_slurm.py --coare=1 --img_to_load=-1 --load_previous=1 --style_iteration=$iteration --content_weight=$c --style_weight=$s
source deactivate

source activate NeilGAN
srun python multistyle_main_slurm.py --coare=1 --img_to_load=-1 --load_previous=1 --style_iteration=$iteration --content_weight=$c --style_weight=$s
source deactivate

source activate NeilGAN
srun python multistyle_main_slurm.py --coare=1 --img_to_load=-1 --load_previous=1 --style_iteration=$iteration --content_weight=$c --style_weight=$s
source deactivate

source activate NeilGAN
srun python multistyle_main_slurm.py --coare=1 --img_to_load=-1 --load_previous=1 --style_iteration=$iteration --content_weight=$c --style_weight=$s
source deactivate

source activate NeilGAN
srun python multistyle_main_slurm.py --coare=1 --img_to_load=-1 --load_previous=1 --style_iteration=$iteration --content_weight=$c --style_weight=$s
source deactivate

source activate NeilGAN
srun python multistyle_main_slurm.py --coare=1 --img_to_load=-1 --load_previous=1 --style_iteration=$iteration --content_weight=$c --style_weight=$s
source deactivate

source activate NeilGAN
srun python multistyle_main_slurm.py --coare=1 --img_to_load=-1 --load_previous=1 --style_iteration=$iteration --content_weight=$c --style_weight=$s
source deactivate

source activate NeilGAN
srun python multistyle_main_slurm.py --coare=1 --img_to_load=-1 --load_previous=1 --style_iteration=$iteration --content_weight=$c --style_weight=$s
source deactivate

source activate NeilGAN
srun python multistyle_main_slurm.py --coare=1 --img_to_load=-1 --load_previous=1 --style_iteration=$iteration --content_weight=$c --style_weight=$s
source deactivate

source activate NeilGAN
srun python multistyle_main_slurm.py --coare=1 --img_to_load=-1 --load_previous=1 --style_iteration=$iteration --content_weight=$c --style_weight=$s
source deactivate

source activate NeilGAN
srun python multistyle_main_slurm.py --coare=1 --img_to_load=-1 --load_previous=1 --style_iteration=$iteration --content_weight=$c --style_weight=$s
source deactivate

source activate NeilGAN
srun python multistyle_main_slurm.py --coare=1 --img_to_load=-1 --load_previous=1 --style_iteration=$iteration --content_weight=$c --style_weight=$s
source deactivate

source activate NeilGAN
srun python multistyle_main_slurm.py --coare=1 --img_to_load=-1 --load_previous=1 --style_iteration=$iteration --content_weight=$c --style_weight=$s
source deactivate

source activate NeilGAN
srun python multistyle_main_slurm.py --coare=1 --img_to_load=-1 --load_previous=1 --style_iteration=$iteration --content_weight=$c --style_weight=$s
source deactivate

source activate NeilGAN
srun python multistyle_main_slurm.py --coare=1 --img_to_load=-1 --load_previous=1 --style_iteration=$iteration --content_weight=$c --style_weight=$s
source deactivate

source activate NeilGAN
srun python multistyle_main_slurm.py --coare=1 --img_to_load=-1 --load_previous=1 --style_iteration=$iteration --content_weight=$c --style_weight=$s
source deactivate

source activate NeilGAN
srun python multistyle_main_slurm.py --coare=1 --img_to_load=-1 --load_previous=1 --style_iteration=$iteration --content_weight=$c --style_weight=$s
source deactivate

source activate NeilGAN
srun python multistyle_main_slurm.py --coare=1 --img_to_load=-1 --load_previous=1 --style_iteration=$iteration --content_weight=$c --style_weight=$s
source deactivate

source activate NeilGAN
srun python multistyle_main_slurm.py --coare=1 --img_to_load=-1 --load_previous=1 --style_iteration=$iteration --content_weight=$c --style_weight=$s
source deactivate