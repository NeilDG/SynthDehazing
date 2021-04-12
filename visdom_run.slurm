#!/bin/bash
#SBATCH -J NeilVis
#SBATCH --partition=serial
#SBATCH --qos=84c-1d_serial
#SBATCH --nodes=1
#SBATCH --output=visdom_run.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph 
#SBATCH --mail-type=END 

# prepare working environment
module load anaconda

# activate your python environment
source activate NeilGAN_V2
visdom -env_path "/scratch1/scratch2/neil.delgallego/Visdom/"
source deactivate

module unload anaconda