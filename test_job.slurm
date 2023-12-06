#!/bin/bash
#SBATCH --job-name=test_llvm_job

#SBATCH --account=csci_ga_3033_102-2023fa
#SBATCH --partition=n1s8-v100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

#SBATCH --time=8:00:00

#SBATCH --output=./test_llvm_job.out
#SBATCH --error=./test_llvm_job.err
#SBATCH --export=ALL

singularity exec --bind /scratch --nv --overlay /scratch/crg9968/llvm/overlay-25GB-500K.ext3:ro /scratch/crg9968/llvm/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash  -c "
source /ext3/env.sh
cd /scratch/crg9968/llvm/Clustered_Diffusion
conda activate diffusion2

OPENAI_LOGDIR=/scratch/crg9968/llvm/logs_test_run_mnist_uniform_mu0sigma1 GPUS_PER_NODE=1 \
python3 scripts/clustered_image_train.py --data_dir /scratch/crg9968/datasets/mnist/train \
 --image_size 32 --num_channels 64 --num_res_blocks 2 --attention_resolutions "16,8" --dropout 0.1 --class_cond True --schedule_sampler uniform \
 --sigma_small True --diffusion_steps 1000 --noise_schedule linear --guidance_loss_type JS --denoise_loss_type MSE \
 --lr 1e-4 --weight_decay 0.0 --batch_size 128 --lr_anneal_steps 10001 --no_guidance_step 2000 --save_interval 1000 \
 --clip_denoised True --num_samples_visualize 100 --use_ddim False \
 --mu0sigma1 True
"
