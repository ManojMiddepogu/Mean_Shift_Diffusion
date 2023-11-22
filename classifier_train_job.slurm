#!/bin/bash
#SBATCH --job-name=llvm_classifier_train_job

#SBATCH --account=csci_ga_3033_102-2023fa
#SBATCH --partition=n1s8-v100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

#SBATCH --time=1:00:00

#SBATCH --output=./classifier_train_job.out
#SBATCH --error=./classifier_train_job.err
#SBATCH --export=ALL

singularity exec --bind /scratch --nv --overlay /scratch/crg9968/llvm/overlay-25GB-500K.ext3:ro /scratch/crg9968/llvm/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/env.sh
cd /scratch/crg9968/llvm/Clustered_Diffusion
conda activate diffusion

OPENAI_LOGDIR=/scratch/crg9968/llvm/classifier_train_bs5112_anneal_lr GPUS_PER_NODE=1 \
python3 scripts/classifier_train.py --data_dir /scratch/crg9968/datasets/mnist/train --val_data_dir /scratch/crg9968/datasets/mnist/test \
 --noised True --image_size 32 --classifier_width 64 \
 --lr 1e-4 --anneal_lr True --batch_size 512 --iterations 10000 \
 --save_interval 10000 --diffusion_steps 1000 --noise_schedule linear
"
