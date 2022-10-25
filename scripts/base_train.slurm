#!/bin/bash

#SBATCH --job-name=base_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=28:00:00
#SBATCH --gres=gpu
#SBATCH --partition=v100

# config path
config_path=$1

# Singularity path
ext3_path=/scratch/$USER/py39/overlay-25GB-500K.ext3
sif_path=/scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

# start running
singularity exec --nv --overlay ${ext3_path}:ro ${sif_path} \
/bin/bash -c "
source /ext3/env.sh
cd /scratch/$USER/Deformable-DETR-Test
python -m main --config ${config_path}
"