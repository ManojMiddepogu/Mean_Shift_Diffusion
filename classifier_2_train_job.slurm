#!/bin/bash
#SBATCH --job-name=classifier_train_job_2

#SBATCH --gres=gpu:rtx8000:2
#SBATCH --cpus-per-task=12

#SBATCH --time=24:00:00

#SBATCH --output=./classifier_train_job_2.out
#SBATCH --error=./classifier_train_job_2.err
#SBATCH --export=ALL

singularity exec --bind /scratch --nv --overlay /scratch/crg9968/llvm/overlay-50G-10M.ext3:ro /scratch/crg9968/llvm/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash  -c "
source /ext3/env.sh
cd /scratch/crg9968/llvm/Clustered_Diffusion
conda activate diffusion2

OPENAI_LOGDIR=/scratch/crg9968/llvm/classifier_train_cifar_2 GPUS_PER_NODE=2 \
python3 scripts/classifier_train.py --data_dir /scratch/crg9968/datasets/cifar10_2/train --val_data_dir /scratch/crg9968/datasets/cifar10_2/test \
 --noised True --image_size 32 --classifier_width 64 \
 --lr 3e-4 --anneal_lr True --batch_size 1024 --iterations 150000 --weight_decay 0.05 \
 --save_interval 3000 --diffusion_steps 1000 --noise_schedule linear
"
