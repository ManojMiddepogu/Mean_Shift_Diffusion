# Mean Shit Diffusion

This repository is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion), from the paper - [Diffusion Models Beat GANS on Image Synthesis](http://arxiv.org/abs/2105.05233). We made modifications to the diffusion process, as described in out report.

## Environment Setup
We recommend to set up a conda environment using the providd `environment.yml` file.
```
conda env create -f environment.yml
```

After setting up the conda environment, run
```
python setup.py develop
```

## Dataset Setup
Update the appropriate paths to download the datasets and run:
```
python datasets/cifar-10.py
python datasets/cifar-2.py
python datasets/mnist.py
```

## Mu and Sigma for FID Computation
To comptue FID of generated samples with the dataset, we compute the mu and sigma of the whole dateset, store it and load these values during training. Run the following commands to get the corresponding values:
```
python3 scripts/compute_fid_score.py --batch-size 512 --save-stats datasets/cifar10/cifar_train_reference.npz datasets/cifar10/cifar_train_mu_sigma.npz
python3 scripts/compute_fid_score.py --batch-size 512 --save-stats datasets/cifar2/cifar_train_reference.npz datasets/cifar2/cifar_train_mu_sigma.npz
python3 scripts/compute_fid_score.py --batch-size 512 --save-stats datasets/mnist/mnist_train_reference.npz datasets/mnist/mnist_train_mu_sigma.npz
```

## Run Experiments
Refer to the scripts `run_experiments.sh` and `run_experiments_2.sh` that run the baseline, adversarial, alternate, adversarial (with guidance freeze), alternate (with guidance freeze) for cifar-10 and cifar-2 respectively. Modify the log directory, dataset directories accordingly.
