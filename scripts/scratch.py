import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion.nn import mean_flat, mean_flat_2
from guided_diffusion.losses import normal_js
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    clustered_model_and_diffusion_defaults,
    create_clustered_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse

th.set_printoptions(precision=10)

os['OPENAI_LOGDIR'] = "/scratch/crg9968/llvm/logs_test_run_newsample"
os["GPUS_PER_NODE"] = 1

defaults = dict()
defaults.update(clustered_model_and_diffusion_defaults())
defaults.update(dict(
    model_path="/scratch/crg9968/llvm/logs_test_run_newsample/model010000.pt",
    class_cond=True,
    image_size=32,
    num_channels=64,
    num_res_blocks=2,
    attention_resolutions="16, 8",
    diffusion_steps=1000,
    noise_schedule="linear",
    guidance_loss_type="JS",
    denoise_loss_type="MSE"
))
parser = argparse.ArgumentParser()
add_dict_to_argparser(parser, defaults)
args = parser.parse_args()

dist_util.setup_dist()
model, diffusion = create_clustered_model_and_diffusion(**args_to_dict(args, clustered_model_and_diffusion_defaults().keys()))
model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
model.to(dist_util.dev())
model.eval()

t = th.arange(args.diffusion_steps, device=dist_util.dev())
y = th.tensor(0, device=dist_util.dev()).expand(t.shape[0])

with th.no_grad():
    mu_bar, sigma_bar = model.guidance_model(t, diffusion.sqrt_one_minus_alphas_cumprod, y)

original_sigma_bar = th.tensor(diffusion.sqrt_one_minus_alphas_cumprod, dtype=sigma_bar[0].dtype, device=dist_util.dev())

assert th.equal(sigma_bar, original_sigma_bar), "sigma bar are not equal"
assert th.allclose(sigma_bar, original_sigma_bar, atol=1e-6), "variance outputs are not equal"
