#!/bin/bash

data_dir="/scratch/crg9968/datasets/cifar10/train"
training_data_inception_mu_sigma_path="/scratch/crg9968/datasets/cifar10/cifar_train_mu_sigma.npz"
distance=0.1
no_guidance_step=4000
lr_anneal_steps=50001
save_interval=1000
fid_interval=2000
num_samples=1000
num_samples_batch_size=500

uniform_scheduler="uniform"
alternate_scheduler="alternate"
gantype_scheduler="gantype"

# Baseline
sbatch ./baseline_cifar_job.slurm "/scratch/crg9968/llvm/logs_0" $data_dir $training_data_inception_mu_sigma_path \
    $uniform_scheduler $lr_anneal_steps $save_interval $fid_interval $num_samples $num_samples_batch_size "baseline_cifar_50K_uniform"

# Clustered, Alternate, Freeze
sbatch ./cifar_clustered_train_job.slurm "/scratch/crg9968/llvm/logs_1" $data_dir $training_data_inception_mu_sigma_path $distance \
    $alternate_scheduler $no_guidance_step "True" $lr_anneal_steps $save_interval $fid_interval $num_samples $num_samples_batch_size \
    "clustered_cifar_50K_alternate_freeze"

# Clustered, Alternate, No Freeze
sbatch ./cifar_clustered_train_job.slurm "/scratch/crg9968/llvm/logs_2" $data_dir $training_data_inception_mu_sigma_path $distance \
    $alternate_scheduler $no_guidance_step "False" $lr_anneal_steps $save_interval $fid_interval $num_samples $num_samples_batch_size \
    "clustered_cifar_50K_alternate_nofreeze"

# Clustered, GANType, Freeze
sbatch ./cifar_clustered_train_job.slurm "/scratch/crg9968/llvm/logs_3" $data_dir $training_data_inception_mu_sigma_path $distance \
    $gantype_scheduler $no_guidance_step "True" $lr_anneal_steps $save_interval $fid_interval $num_samples $num_samples_batch_size \
    "clustered_cifar_50K_gantype_freeze"

# Clustered, GANType, No Freeze
sbatch ./cifar_clustered_train_job.slurm "/scratch/crg9968/llvm/logs_4" $data_dir $training_data_inception_mu_sigma_path $distance \
    $gantype_scheduler $no_guidance_step "False" $lr_anneal_steps $save_interval $fid_interval $num_samples $num_samples_batch_size \
    "clustered_cifar_50K_gantype_nofreeze"
