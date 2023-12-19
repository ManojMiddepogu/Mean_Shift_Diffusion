import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from guided_diffusion.nn import mean_flat, mean_flat_2, timestep_embedding
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
    # model_path="/scratch/crg9968/llvm/logs_4/model033000.pt", # Same t
    # model_path="/scratch/crg9968/llvm/logs_4/model043000.pt", # Gantype NoFreeze Noscaleshift
        # use_scale_shift_norm=False,
    model_path="/scratch/crg9968/llvm/logs_2/model041000.pt", # Alternate NoFreeze 2classes
    class_cond=True,
    image_size=32,
    num_channels=64,
    num_res_blocks=2,
    attention_resolutions="16, 8",
    diffusion_steps=1000,
    noise_schedule="linear",
    guidance_loss_type="JS",
    denoise_loss_type="MSE",
))
parser = argparse.ArgumentParser()
add_dict_to_argparser(parser, defaults)
args = parser.parse_args()

dist_util.setup_dist()
model, diffusion = create_clustered_model_and_diffusion(**args_to_dict(args, clustered_model_and_diffusion_defaults().keys()))
model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
model.to(dist_util.dev())
model.eval()

# t = th.arange(args.diffusion_steps, device=dist_util.dev())
# y = th.tensor(0, device=dist_util.dev()).expand(t.shape[0])
# with th.no_grad():
#     mu_bar, sigma_bar = model.guidance_model(t, diffusion.sqrt_one_minus_alphas_cumprod, y)
# original_sigma_bar = th.tensor(diffusion.sqrt_one_minus_alphas_cumprod, dtype=sigma_bar[0].dtype, device=dist_util.dev())
# assert th.equal(sigma_bar, original_sigma_bar), "sigma bar are not equal"
# assert th.allclose(sigma_bar, original_sigma_bar, atol=1e-6), "variance outputs are not equal"

num_classes = 2
t = th.arange(0, 1000, 100, device=dist_util.dev())
y = th.tensor(0, device=dist_util.dev()).expand(t.shape[0])
y_bar = th.arange(0, num_classes, device=dist_util.dev())
t_bar = th.tensor(999, device=dist_util.dev()).expand(y_bar.shape[0])

with th.no_grad():
    time_embed_check = model.guidance_model.time_embed(timestep_embedding(t, model.guidance_model.model_channels))
    label_embed_check = model.guidance_model.label_emb(y_bar)
    mu_bar_same_y, sigma_bar_same_y = model.guidance_model(t, diffusion.sqrt_one_minus_alphas_cumprod, y)
    mu_bar_same_t, sigma_bar_same_t = model.guidance_model(t_bar, diffusion.sqrt_one_minus_alphas_cumprod, y_bar)
    mu_bar_same_y = mu_bar_same_y.view(t.shape[0], -1)
    mu_bar_same_t = mu_bar_same_t.view(t_bar.shape[0], -1)
    mu_bar_same_y = th.cat((th.zeros((1, 3072), device=mu_bar_same_y.device), mu_bar_same_y), dim=0)
    mu_bar_same_t = th.cat((th.zeros((1, 3072), device=mu_bar_same_t.device), mu_bar_same_t), dim=0)

print("Time Embed Shape", time_embed_check.shape)
time_embed_dist_matrix = th.norm(time_embed_check[1:] - time_embed_check[:-1], dim=1, p=2)
print("Time Embed Matrix", time_embed_dist_matrix, "\n")

print("Label Embed Shape", label_embed_check.shape)
norm_tensor = F.normalize(label_embed_check, p=2, dim=1)
label_embed_cosine_similarity_matrix = th.mm(norm_tensor, norm_tensor.t())
print("Label Embed Matrix", label_embed_cosine_similarity_matrix, "\n")

print("Mu Bar Same Y Shape", mu_bar_same_y.shape)
mu_bar_same_y_dist_matrix = th.norm(mu_bar_same_y[1:] - mu_bar_same_y[:-1], dim=1, p=2)
print("Mu Bar Same Y Distance Matrix", mu_bar_same_y_dist_matrix)

print("Mu Bar Same Y Shape", mu_bar_same_y.shape)
norm_tensor = F.normalize(mu_bar_same_y, p=2, dim=1)
mu_bar_same_y_cosine_similarity_matrix = th.mm(norm_tensor, norm_tensor.t())
print("Mu Bar Same Y Same Class Similarity Matrix", mu_bar_same_y_cosine_similarity_matrix)

print("Mu Bar Same T Shape", mu_bar_same_t.shape)
norm_tensor = F.normalize(mu_bar_same_t, p=2, dim=1)
mu_bar_same_t_cosine_similarity_matrix = th.mm(norm_tensor, norm_tensor.t())
print("Mu Bar Same T Same Class Similarity Matrix", mu_bar_same_t_cosine_similarity_matrix)

print("Mu Bar Same T Shape", mu_bar_same_t.shape)
mu_bar_same_t_distance = th.cdist(mu_bar_same_t, mu_bar_same_t, p=2)
print("Mu Bar Same T Distance", mu_bar_same_t_distance)
