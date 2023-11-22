#!/bin/bash
#SBATCH --job-name=test_llvm_job

#SBATCH --account=csci_ga_3033_102-2023fa
#SBATCH --partition=n1s8-v100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

#SBATCH --time=1:00:00

#SBATCH --output=./test_llvm_metrics.out
#SBATCH --error=./test_llvm_metrics.err
#SBATCH --export=ALL

singularity exec --bind /scratch --nv --overlay /scratch/mm12799/overlay-25GB-500K.ext3:ro /scratch/mm12799/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/env.sh
cd /scratch/mm12799/Clustered_Diffusion
conda activate diffusion

OPENAI_LOGDIR=/scratch/mm12799/Clustered_Diffusion/guided_diffusion/logs/logs_test_run GPUS_PER_NODE=1 \
python3 /scratch/mm12799/Clustered_Diffusion/evaluations/evaluator.py  \
 --ref_batch /scratch/mm12799/Clustered_Diffusion/guided_diffusion/logs/logs_test_run/samples_32x32x32x3.npz \
 --sample_batch /scratch/mm12799/Clustered_Diffusion/guided_diffusion/logs/logs_test_run/samples_32x32x32x3.npz
"