#!/bin/bash

data_dir="/scratch/crg9968/datasets/cifar10"
training_data_inception_mu_sigma_path="/scratch/crg9968/datasets/cifar10/train/cifar_train_mu_sigma.npz"
distance=0.1
no_guidance_step=4000
lr_anneal_steps=50001
save_interval=1000
fid_interval=2000
num_samples=1000
num_samples_batch_size=500
use_scale_shift_norm=True

same_t_sampler="same-t"
uniform_scheduler="uniform"
alternate_scheduler="alternate"
gantype_scheduler="gantype"

batch_size_128=128

# Same-T Sampler
    # Step < no_guidance_step
        # Both models
        # Both Losses
    # Step >= no_guidance_step
        # No guidance loss
        # Freeze
            # No guidance model

# Alternate Sampler
    # Step < no_guidance_step
        # Even - Same T
            # Both models
            # Both Losses
        # Odd - Uniform
            # No guidance model
            # No guidance loss
    # Step >= no_guidance_step
        # No guidance loss
        # Freeze
            # No guidance model

# GAN Type Sampler
    # Step < no_guidance_step
        # Even - Same T
            # No denoise model
            # Both Losses
        # Odd - Uniform
            # No guidance model
            # No guidance loss
    # Step >= no_guidance_step
        # No guidance loss
        # Freeze
            # No guidance model

# Baseline
sbatch ./baseline_cifar_job.slurm "/scratch/crg9968/llvm/logs_baseline_cifar" $data_dir $training_data_inception_mu_sigma_path \
    $uniform_scheduler $lr_anneal_steps $save_interval $fid_interval $num_samples $num_samples_batch_size "baseline_cifar_50K_uniform" \
    $use_scale_shift_norm $batch_size_128

# Clustered, Alternate, Freeze
sbatch ./cifar_clustered_train_job.slurm "/scratch/crg9968/llvm/logs_clustered_cifar_alternate_freeze" $data_dir $training_data_inception_mu_sigma_path $distance \
    $alternate_scheduler $no_guidance_step "True" $lr_anneal_steps $save_interval $fid_interval $num_samples $num_samples_batch_size \
    "clustered_cifar_50K_alternate_freeze" $use_scale_shift_norm $batch_size_128

# Clustered, Alternate, No Freeze
sbatch ./cifar_clustered_train_job.slurm "/scratch/crg9968/llvm/logs_clustered_cifar_alternate_nofreeze" $data_dir $training_data_inception_mu_sigma_path $distance \
    $alternate_scheduler $no_guidance_step "False" $lr_anneal_steps $save_interval $fid_interval $num_samples $num_samples_batch_size \
    "clustered_cifar_50K_alternate_nofreeze" $use_scale_shift_norm $batch_size_128

# Clustered, GANType, Freeze
sbatch ./cifar_clustered_train_job.slurm "/scratch/crg9968/llvm/logs_clustered_cifar_gantype_freeze" $data_dir $training_data_inception_mu_sigma_path $distance \
    $gantype_scheduler $no_guidance_step "True" $lr_anneal_steps $save_interval $fid_interval $num_samples $num_samples_batch_size \
    "clustered_cifar_50K_gantype_freeze" $use_scale_shift_norm $batch_size_128

# Clustered, GANType, No Freeze
sbatch ./cifar_clustered_train_job.slurm "/scratch/crg9968/llvm/logs_clustered_cifar_gantype_nofreeze" $data_dir $training_data_inception_mu_sigma_path $distance \
    $gantype_scheduler $no_guidance_step "False" $lr_anneal_steps $save_interval $fid_interval $num_samples $num_samples_batch_size \
    "clustered_cifar_50K_gantype_nofreeze" $use_scale_shift_norm $batch_size_128
