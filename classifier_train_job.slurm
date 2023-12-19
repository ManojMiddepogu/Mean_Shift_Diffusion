#!/bin/bash
#SBATCH --job-name=llvm_classifier_train_job

#SBATCH --account=csci_ga_3033_102-2023fa
#SBATCH --partition=n1s16-v100-2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8

#SBATCH --time=24:00:00

#SBATCH --output=./classifier_train_job.out
#SBATCH --error=./classifier_train_job.err
#SBATCH --export=ALL

singularity exec --bind /scratch --nv --overlay /scratch/crg9968/llvm/overlay-25GB-500K.ext3:ro /scratch/crg9968/llvm/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash  -c "
source /ext3/env.sh
cd /scratch/crg9968/llvm/Clustered_Diffusion
conda activate diffusion2

OPENAI_LOGDIR=/scratch/crg9968/llvm/classifier_train_cifar GPUS_PER_NODE=2 \
python3 scripts/classifier_train.py --data_dir /scratch/crg9968/datasets/cifar10/train --val_data_dir /scratch/crg9968/datasets/cifar10/test \
 --noised True --image_size 32 --classifier_width 64 \
 --lr 3e-4 --anneal_lr True --batch_size 512 --iterations 150000 \
 --save_interval 5000 --diffusion_steps 1000 --noise_schedule linear
"
