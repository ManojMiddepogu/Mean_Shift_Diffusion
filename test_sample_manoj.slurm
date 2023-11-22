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

singularity exec --bind /scratch --nv --overlay /scratch/mm12799/overlay-25GB-500K.ext3:ro /scratch/mm12799/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/env.sh
cd /scratch/mm12799/Clustered_Diffusion
conda activate diffusion

OPENAI_LOGDIR=/scratch/mm12799/Clustered_Diffusion/guided_diffusion/logs/logs_test_run GPUS_PER_NODE=1 \
python3 scripts/image_sample.py --num_samples 32 --batch_size 32 --use_ddim False \
 --model_path /scratch/mm12799/Clustered_Diffusion/guided_diffusion/logs/logs_test_run/model030000.pt \
 --image_size 32 --num_channels 64 \
 --num_res_blocks 2 --attention_resolutions "16,8"\
 --diffusion_steps 1000 --noise_schedule linear
"
