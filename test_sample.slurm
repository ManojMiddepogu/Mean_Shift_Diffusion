#!/bin/bash
#SBATCH --job-name=test_llvm_job

#SBATCH --account=csci_ga_3033_102-2023fa
#SBATCH --partition=n1s8-v100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

#SBATCH --time=1:00:00

#SBATCH --output=./test_llvm_sample.out
#SBATCH --error=./test_llvm_sample.err
#SBATCH --export=ALL

singularity exec --bind /scratch --nv --overlay /scratch/crg9968/llvm/overlay-25GB-500K.ext3:ro /scratch/crg9968/llvm/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/env.sh
cd /scratch/crg9968/llvm/Clustered_Diffusion
conda activate diffusion

OPENAI_LOGDIR=/scratch/crg9968/llvm/logs_test_run_mnist_first_run GPUS_PER_NODE=1 \
python3 scripts/image_sample.py --num_samples 100 --plot_samples 100 --batch_size 128 --use_ddim False \
 --model_path /scratch/crg9968/llvm/logs_test_run_mnist_first_run/ema_0.9999_030000.pt \
 --image_size 32 --num_channels 64 \
 --num_res_blocks 2 --attention_resolutions "16,8" \
 --diffusion_steps 1000 --noise_schedule linear \
 --save_images True
"
