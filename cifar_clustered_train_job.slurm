#!/bin/bash
#SBATCH --job-name=cifar_clustered_train_job

#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16GB

#SBATCH --time=24:00:00

#SBATCH --output=./cifar_clustered_train_job.out
#SBATCH --error=./cifar_clustered_train_job.err
#SBATCH --export=ALL

openai_logdir=$1
data_dir=$2
training_data_inception_mu_sigma_path=$3
distance=$4
schedule_sampler=$5
no_guidance_step=$6
freeze_guidance_after_no_guidance_step=$7
lr_anneal_steps=$8
save_interval=$9
fid_interval=${10}
num_samples=${11}
num_samples_batch_size=${12}
wandb_run_name=${13}
use_scale_shift_norm=${14}
batch_size=${15}


singularity exec --bind /scratch --nv --overlay /scratch/crg9968/llvm/overlay-25GB-500K.ext3:ro /scratch/crg9968/llvm/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash  -c "
source /ext3/env.sh
cd /scratch/crg9968/llvm/Clustered_Diffusion
conda activate diffusion2

OPENAI_LOGDIR=${openai_logdir} GPUS_PER_NODE=1 \
python3 scripts/clustered_image_train.py \
 --data_dir ${data_dir} --random_flip True \
 --training_data_inception_mu_sigma_path ${training_data_inception_mu_sigma_path} \
 --image_size 32 --num_channels 64 --num_res_blocks 2 --attention_resolutions "16,8" --dropout 0.1 \
 --diffusion_steps 1000 --noise_schedule linear --class_cond True --sigma_small True --use_scale_shift_norm ${use_scale_shift_norm} \
 --guidance_loss_type JS --denoise_loss_type MSE --distance ${distance} --scale_distance True \
 --schedule_sampler ${schedule_sampler} --no_guidance_step ${no_guidance_step} --freeze_guidance_after_no_guidance_step ${freeze_guidance_after_no_guidance_step} \
 --lr 1e-4 --weight_decay 0.0 --batch_size ${batch_size} --lr_anneal_steps ${lr_anneal_steps} --save_interval ${save_interval} --fid_interval ${fid_interval} \
 --clip_denoised True --num_samples ${num_samples} --num_samples_batch_size ${num_samples_batch_size} --num_samples_visualize 100 --use_ddim False \
 --wandb_run_name ${wandb_run_name}
"
