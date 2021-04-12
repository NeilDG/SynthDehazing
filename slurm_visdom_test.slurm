#!/bin/bash
#SBATCH -J VISDOM_TEST
#SBATCH --partition=serial
#SBATCH --qos=84c-1d_serial
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --output=visdom_test.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph
#SBATCH --mail-type=END

#About this script:
# Installation of necessary libraries

module load anaconda
module load cuda/10.1_cudnn-7.6.5

source activate NeilGAN_V2

srun python visdom_plot_test_main.py --coare=1
conda deactivate