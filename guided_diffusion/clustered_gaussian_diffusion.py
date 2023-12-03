import enum
import math
import random

import numpy as np
import torch as th

from .nn import mean_flat, mean_flat_2
from .losses import normal_kl, discretized_gaussian_log_likelihood, normal_js, normal_wd
# from .test_model import TestModel
from .clustered_model import ClusteredModel


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ClusteredModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}, NOT IMPLEMENTED SHOULD BE EASY TO IMPLEMENT
    START_X = enum.auto()  # the model predicts x_0, NOT IMPLEMENTED SHOULD BE EASY TO IMPLEMENT
    EPSILON = enum.auto()  # the model predicts epsilon


class ClusteredModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto() # NOT IMPLEMENTED
    FIXED_SMALL = enum.auto() # SIGMA_TILDE^2(Y, T)
    FIXED_LARGE = enum.auto() # BETA_T SIGMA^2(Y, T)
    LEARNED_RANGE = enum.auto() # NOT IMPLEMENTED


class ClusteredDenoiseLossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class ClusteredGuidanceLossType(enum.Enum):
    JS = enum.auto()  # use Jensen Shannon
    WD = enum.auto()  # use Wasserstein Distance


class ClusteredGaussianDiffusion:

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        guidance_loss_type,
        denoise_loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        if self.model_mean_type not in [ClusteredModelMeanType.EPSILON]:
            raise NotImplementedError(f"Model Mean Type {self.model_mean_type} not implemented!")
        self.model_var_type = model_var_type
        if self.model_var_type not in [ClusteredModelVarType.FIXED_SMALL, ClusteredModelVarType.FIXED_LARGE]:
            raise NotImplementedError(f"Model Var Type {self.model_var_type} not implemented!")
        self.guidance_loss_type = guidance_loss_type
        self.denoise_loss_type = denoise_loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        self.alphas = 1.0 - betas
        self.alphas_next = np.append(self.alphas[1:], 0.0)
        self.sqrt_alphas = np.sqrt(self.alphas)
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(self.alphas_cumprod_prev)
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod_prev = np.sqrt(1.0 - np.append(1.0, self.alphas_cumprod[:-1]))
        self.sqrt_one_minus_alphas_cumprod_next = np.sqrt(1.0 - np.append(self.alphas_cumprod[1:], 0.0))
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.test_print = True

    def q_mean_variance(self, x_start, t, mu_bar_y_t, sigma_bar_y_t):
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        ) + mu_bar_y_t
        # CHECK - IS THIS VARIANCE CORRECT? SHOULDN'T THIS BE DIAGONAL?
        variance = _broadcast_tensor(sigma_bar_y_t ** 2, x_start.shape)
        log_variance = th.log(variance)
        return mean, variance, log_variance
    
    def q_sample(self, x_start, t, mu_bar_y_t, sigma_bar_y_t, noise=None):
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + mu_bar_y_t
            + _broadcast_tensor(sigma_bar_y_t, x_start.shape) * noise
        )
    
    def posterior_variance(self, x_t, t, sigma_bar_y_t, sigma_bar_y_tm1):
        return _broadcast_tensor(sigma_bar_y_tm1 ** 2, x_t.shape) * (1 - _extract_into_tensor(self.alphas, t, x_t.shape) * _broadcast_tensor((sigma_bar_y_tm1 / sigma_bar_y_t) ** 2, x_t.shape))
    
    def q_posterior_variance(self, x_t, t, sigma_bar_y_t, sigma_bar_y_tm1, sigma_bar_y_tp1 = None):
        posterior_variance = self.posterior_variance(x_t, t, sigma_bar_y_t, sigma_bar_y_tm1)

        t_is_zero = (t == 0)
        # Make sure that `sigma_bar_y_tp1` is provided when it's needed
        if t_is_zero.any():
            assert sigma_bar_y_tp1 is not None, "sigma_bar_y_tp1 is required when any t == 0"

        # Compute the log variance for t+1 where t is 0
        if t_is_zero.any():
            posterior_log_variance_t_plus_1 = th.log(
                self.posterior_variance(x_t[t_is_zero], t[t_is_zero] + 1, sigma_bar_y_tp1[t_is_zero], sigma_bar_y_t[t_is_zero])
            )
        
        # Prepare a tensor for the log variance clipped with the same shape as posterior_variance
        posterior_log_variance_clipped = th.empty_like(posterior_variance)
        
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain.
        # Assign the computed log variance for t+1 to the corresponding elements
        if t_is_zero.any():
            posterior_log_variance_clipped[t_is_zero] = posterior_log_variance_t_plus_1
        
        # Compute and assign the log variance for the rest of the batch
        posterior_log_variance_clipped[~t_is_zero] = th.log(posterior_variance[~t_is_zero])

        assert (
            posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_t.shape[0]
        )
        return posterior_variance, posterior_log_variance_clipped
    
    def q_posterior_mean(self, x_start, x_t, t, mu_bar_y_t, mu_bar_y_tm1, sigma_bar_y_t, sigma_bar_y_tm1):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod_prev, t, x_t.shape) * x_start
            + mu_bar_y_tm1
            + (
                _extract_into_tensor(self.sqrt_alphas, t, x_t.shape)
                * _broadcast_tensor((sigma_bar_y_tm1 / sigma_bar_y_t) ** 2, x_t.shape)
                * (x_t - _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_start - mu_bar_y_t)
            )
        )
        return posterior_mean
    
    def q_posterior_mean_variance(self, x_start, x_t, t, mu_bar_y_t, mu_bar_y_tm1, sigma_bar_y_t, sigma_bar_y_tm1, sigma_bar_y_tp1 = None):
        assert x_start.shape == x_t.shape
        posterior_mean = self.q_posterior_mean(x_start, x_t, t, mu_bar_y_t, mu_bar_y_tm1, sigma_bar_y_t, sigma_bar_y_tm1)
        posterior_variance, posterior_log_variance_clipped = self.q_posterior_variance(x_t, t, sigma_bar_y_t, sigma_bar_y_tm1, sigma_bar_y_tp1)
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, mu_bar_y_t, mu_bar_y_tm1, sigma_bar_y_t, sigma_bar_y_tm1, sigma_bar_y_tp1 = None, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model.denoise_model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ClusteredModelVarType.FIXED_SMALL, ClusteredModelVarType.FIXED_LARGE]:
            model_variance, model_log_variance = {
                # CHECK - FOR FIXED_LARGE, AND FIXED_SMALL; THERE IS A SMALL DIFFERENCE AT T==0, IN TERMS OF VARIANCE NOT LOG VARIANCE VALUE. IS THIS CORRECT?
                # CHECK - WRITE THIS EFFICIENTLY?
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                # CHECK - THIS IS INCORRECT, t IS FOR A BATCH NOT A SINGLE ELEMENT. MAKE CHANGES ACCORDING TO q_posterior_variance()
                # ClusteredModelVarType.FIXED_LARGE: (self.q_posterior_variance(x_t, t+1, sigma_bar_y_tp1, sigma_bar_y_t, None)) if (t == 0) else \
                #      (sigma_bar_y_t ** 2 - _extract_into_tensor(self.alphas, t, x_t) * sigma_bar_y_tm1 ** 2, np.log(sigma_bar_y_t ** 2 - _extract_into_tensor(self.alphas, t, x_t) * sigma_bar_y_tm1 ** 2)),
                # CHECK - HANDLE t==0 CASE PROPERLY THIS IS COMPLETELY WRONG FOR FIXED_LARGE. I CAN TORALLY IGNORE THIS AND SAY NOT IMPLEMENTED
                ClusteredModelVarType.FIXED_LARGE: (_broadcast_tensor(sigma_bar_y_t ** 2, x.shape) - _extract_into_tensor(self.alphas, t, x.shape) * _broadcast_tensor(sigma_bar_y_tm1 ** 2, x.shape), th.log(_broadcast_tensor(sigma_bar_y_t ** 2, x.shape) - _extract_into_tensor(self.alphas, t, x.shape) * _broadcast_tensor(sigma_bar_y_tm1 ** 2, x.shape))),
                ClusteredModelVarType.FIXED_SMALL: (self.q_posterior_variance(x, t, sigma_bar_y_t, sigma_bar_y_tm1, sigma_bar_y_tp1)),
            }[self.model_var_type]
        else:
            raise NotImplementedError(f"Model Var Type {self.model_var_type} not implemented!")

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x
        
        if self.model_mean_type == ClusteredModelMeanType.EPSILON:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output, mu_bar_y_t=mu_bar_y_t, sigma_bar_y_t=sigma_bar_y_t)
            )
            model_mean = self.q_posterior_mean(pred_xstart, x, t, mu_bar_y_t, mu_bar_y_tm1, sigma_bar_y_t, sigma_bar_y_tm1)
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart, mu_bar_y_t, sigma_bar_y_t):
        return (
            x_t
            - _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * pred_xstart
            - mu_bar_y_t
        ) / _broadcast_tensor(sigma_bar_y_t, x_t.shape)

    def _predict_xstart_from_eps(self, x_t, t, eps, mu_bar_y_t, sigma_bar_y_t):
        assert x_t.shape == eps.shape
        return (
            (x_t - mu_bar_y_t - _broadcast_tensor(sigma_bar_y_t, x_t.shape) * eps) * _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        )

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t
    
    def p_sample(
        self,
        model,
        x,
        t,
        mu_bar_y_t,
        mu_bar_y_tm1,
        sigma_bar_y_t,
        sigma_bar_y_tm1,
        sigma_bar_y_tp1=None,
        clip_denoised=True,
        denoised_fn=None,
        # CHECK - REMOVED THE CONDITION FUNCTION HERE, SINCE WE WANT TO COMPARE WITHOUT THIS GUIDANCE?
        model_kwargs=None,
    ):
        out = self.p_mean_variance(
            model,
            x,
            t,
            mu_bar_y_t,
            mu_bar_y_tm1,
            sigma_bar_y_t,
            sigma_bar_y_tm1,
            sigma_bar_y_tp1,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]
    
    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        
        indices = list(range(self.num_timesteps))[::-1]
        y = model_kwargs['y']

        # if isinstance(model, TestModel): # FOR SAMPLING SCRIPT
        if isinstance(model, ClusteredModel): # FOR SAMPLING SCRIPT
            guidance_model = model.guidance_model
        else:
            guidance_model = model.module.guidance_model # CHECK - IF model.module WORKS IN THE SAMPLING SCRIPT

        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
            t = th.tensor([indices[0]] * shape[0], device=device)
            with th.no_grad():
                mu_bar_y_t, sigma_bar_y_t = guidance_model(t, self.sqrt_one_minus_alphas_cumprod, y)
                img = mu_bar_y_t + _broadcast_tensor(sigma_bar_y_t, img.shape) * img

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                mu_bar_y_t, sigma_bar_y_t = guidance_model(t, self.sqrt_one_minus_alphas_cumprod, y)
                mu_bar_y_tm1, sigma_bar_y_tm1 = guidance_model(t-1, self.sqrt_one_minus_alphas_cumprod, y)
                mu_bar_y_tp1, sigma_bar_y_tp1 = (None, None)
                t_is_zero = (t == 0)
                if t_is_zero.any():
                    t_incremented = th.where(t_is_zero, t+1, t)
                    # CHECK - INCREMENTING t ONLY FOR t==0, NOTE THIS IS WRONG FOR CASES WHERE T is NOT 0, BUT THIS IS NOT AN ISSUE AS WE ONLY USE THE INDICES WHERE t==0
                    # CHECK - ALSO I AM DOING THIS BECAUSE USING T+1 FOR THE WHOLE THING IS CAUSING OUT OF BOUNDS ISSUE IN RESPACE.PY FILE FOR THE MAP[ts]
                    mu_bar_y_tp1, sigma_bar_y_tp1 = guidance_model(t_incremented, self.sqrt_one_minus_alphas_cumprod, y)
                if self.test_print:
                    print(sigma_bar_y_t)
                    self.test_print = False
                    # print(sigma_bar_y_tm1)
                    # print(sigma_bar_y_tp1)
                out = self.p_sample(
                    model, # PASSING WHOLE MODEL HERE, AND WILL PICK DENOISE MODEL INSIDE
                    img,
                    t,
                    mu_bar_y_t,
                    mu_bar_y_tm1,
                    sigma_bar_y_t,
                    sigma_bar_y_tm1,
                    sigma_bar_y_tp1,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]
    
    def _vb_terms_bpd(
        self, model, x_start, x_t, t, mu_bar_y_t, mu_bar_y_tm1, sigma_bar_y_t, sigma_bar_y_tm1, sigma_bar_y_tp1, clip_denoised=True, model_kwargs=None
    ):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start, x_t, t, mu_bar_y_t, mu_bar_y_tm1, sigma_bar_y_t, sigma_bar_y_tm1, sigma_bar_y_tp1
        )
        out = self.p_mean_variance(
            model, x_t, t, mu_bar_y_t, mu_bar_y_tm1, sigma_bar_y_t, sigma_bar_y_tm1, sigma_bar_y_tp1, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}
    
    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, no_guidance=False):
        # CHECK - LOSS IS NOT IMPLEMENTED YET.
        y = model_kwargs['y']
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        
        terms = {}
        
        guidance_model = model.guidance_model
        denoise_model = model.denoise_model

        # CHECK - _broadcast_tensor(), BETTER TO BROADCAST SIGMA HERE ITSELF? RATHER THAN IN FUNCTIONS?
        if no_guidance:
            with th.no_grad():
                mu_bar_y_t, sigma_bar_y_t = guidance_model(t, self.sqrt_one_minus_alphas_cumprod, y)
        else:
            mu_bar_y_t, sigma_bar_y_t = guidance_model(t, self.sqrt_one_minus_alphas_cumprod, y)
        # CHECK - for t==0, i.e., t-1 < 0; mu_bar and sigma_bar should be ideally 0.
            # CHECK - IN THIS CASE ARE THE MODEL GIVEN VALUES USED ANYWHERE? THEY SHOULD DEFINITELY NOT BE USED IN GRADIENT COMPUTATION.....
        # THIS SHOULD BE sqrt_one_minus_alphas_cumprod AND NOT sqrt_one_minus_alphas_cumprod_prev, SINCE I AM PASSING t-1 AS INPUT. SAME GOES FOR OTHER PLACES AS WELL.
        if no_guidance:
            with th.no_grad():
                mu_bar_y_tm1, sigma_bar_y_tm1 = guidance_model(t-1, self.sqrt_one_minus_alphas_cumprod, y)
        else:
            mu_bar_y_tm1, sigma_bar_y_tm1 = guidance_model(t-1, self.sqrt_one_minus_alphas_cumprod, y)
        mu_bar_y_tp1, sigma_bar_y_tp1 = (None, None)
        t_is_zero = (t == 0)
        if t_is_zero.any():
            t_incremented = th.where(t_is_zero, t+1, t)
            # CHECK - INCREMENTING t ONLY FOR t==0, NOTE THIS IS WRONG FOR CASES WHERE T is NOT 0, BUT THIS IS NOT AN ISSUE AS WE ONLY USE THE INDICES WHERE t==0 IN q_posterior_variance
            # CHECK - ALSO I AM DOING THIS BECAUSE USING T+1 FOR THE WHOLE THING IS CAUSING OUT OF BOUNDS ISSUE IN RESPACE.PY FILE FOR THE MAP[ts]
            if no_guidance:
                with th.no_grad():
                    mu_bar_y_tp1, sigma_bar_y_tp1 = guidance_model(t_incremented, self.sqrt_one_minus_alphas_cumprod, y)
            else:
                mu_bar_y_tp1, sigma_bar_y_tp1 = guidance_model(t_incremented, self.sqrt_one_minus_alphas_cumprod, y)

        q_mean, q_variance, q_log_variance = self.q_mean_variance(x_start, t, mu_bar_y_t, sigma_bar_y_t)
        # CHECK - ADD GUIDANCE TRIPLET LOSS USING THE ABOVE VALUES
        terms["guidance_loss"] = 0.0
        # if self.guidance_loss_type == ClusteredGuidanceLossType.JS or self.guidance_loss_type == ClusteredGuidanceLossType.WD:
        if not no_guidance:
            if self.guidance_loss_type == ClusteredGuidanceLossType.JS or self.guidance_loss_type == ClusteredGuidanceLossType.WD:
                b, *shape = q_mean.shape

                # Expand and repeat mean and sigma for vectorized computation
                q_mean_1 = q_mean.unsqueeze(1).expand(b, b, *shape)
                q_log_variance_1 = q_log_variance.unsqueeze(1).expand(b, b, *shape)
                q_mean_2 = q_mean.unsqueeze(0).expand(b, b, *shape)
                q_log_variance_2 = q_log_variance.unsqueeze(0).expand(b, b, *shape)

                # Compute distance -> bxb
                distance_matrix = mean_flat_2(normal_js(q_mean_1, q_log_variance_1, q_mean_2, q_log_variance_2)) \
                            if self.guidance_loss_type == ClusteredGuidanceLossType.JS else \
                                    mean_flat_2(normal_wd(q_mean_1, q_log_variance_1, q_mean_2, q_log_variance_2))
                distance_matrix_diff = distance_matrix.clone()

                # Expand labels for comparison
                labels_expanded = y.view(b, 1).expand(b, b)
                same_class_mask = labels_expanded.eq(labels_expanded.t())
                diff_class_mask = ~same_class_mask

                # Masks to avoid self-comparison
                eye_mask = th.eye(b).bool()
                same_class_mask[eye_mask] = False
                distance_matrix[eye_mask] = th.tensor(-float('inf')) # update same batch element for min
                distance_matrix_diff[eye_mask] = th.tensor(float('inf')) # update same batch element for max

                distance_matrix[diff_class_mask] = th.tensor(-float('inf')) # Update diff class elements for min
                distance_matrix_diff[same_class_mask] = th.tensor(float('inf')) # Update same class elements for max

                distance_same = distance_matrix.max(dim=1).values
                distance_same[distance_same == -float("inf")] = 0.0 # => No same class
                distance_diff = distance_matrix_diff.min(dim=1).values
                distance_diff[distance_diff == float("inf")] = 0.0 # => No diff class

                # terms["guidance_loss"] += th.mean(th.nn.functional.relu(distance_same - distance_diff + 1))
                # terms["guidance_loss"] += th.mean(th.nn.functional.relu(distance_same - distance_diff + 3))
                terms["guidance_loss"] += th.mean(th.nn.functional.relu(distance_same - distance_diff + 6))

                # labels = y['y']
                # b = labels.shape[0]
                
                # # Expand labels for comparison
                # labels_expanded = labels.view(b, 1).expand(b, b)
                # same_class_mask = labels_expanded.eq(labels_expanded.t())
                # diff_class_mask = ~same_class_mask

                # # CHECK - DO WE MASK OUT SELF EXAMPLES?

                # # Randomly select indices for same and different classes
                # same_class_indices = [th.masked_select(th.arange(b), same_class_mask[i]) for i in range(b)]
                # diff_class_indices = [th.masked_select(th.arange(b), diff_class_mask[i]) for i in range(b)]

                # same_class_indices = [random.choice(indices) for indices in same_class_indices]
                # diff_class_indices = [random.choice(indices) for indices in diff_class_indices]

                # same_class_mean = q_mean[same_class_indices]
                # same_class_log_variance = q_log_variance[same_class_indices]
                # diff_class_mean = q_mean[diff_class_indices]
                # diff_class_log_variance = q_log_variance[diff_class_indices]

                # js_same = mean_flat(normal_js(q_mean, q_log_variance, same_class_mean, same_class_log_variance))
                # js_diff = mean_flat(normal_js(q_mean, q_log_variance, diff_class_mean, diff_class_log_variance))

                # # CHECK - TRIPLET LOSS MIN DISTANCE?? 1??
                # terms["guidance_loss"] += th.mean(th.nn.functional.relu(js_same - js_diff + 1))
            else:
                raise NotImplementedError(self.guidance_loss_type)

        x_t = self.q_sample(x_start, t, mu_bar_y_t, sigma_bar_y_t, noise)
        if self.denoise_loss_type == ClusteredDenoiseLossType.KL or self.denoise_loss_type == ClusteredDenoiseLossType.RESCALED_KL:
            terms["denoise_loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                mu_bar_y_t=mu_bar_y_t,
                mu_bar_y_tm1=mu_bar_y_tm1,
                sigma_bar_y_t=sigma_bar_y_t,
                sigma_bar_y_tm1=sigma_bar_y_tm1,
                sigma_bar_y_tp1=sigma_bar_y_tp1,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.denoise_loss_type == ClusteredDenoiseLossType.RESCALED_KL:
                terms["denoise_loss"] *= self.num_timesteps
        elif self.denoise_loss_type == ClusteredDenoiseLossType.MSE or self.denoise_loss_type == ClusteredDenoiseLossType.RESCALED_MSE:
            model_output = denoise_model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type not in [ClusteredModelVarType.FIXED_SMALL, ClusteredModelVarType.FIXED_LARGE]:
                raise NotImplementedError(f"Model Var Type {self.model_var_type} not implemented!")
            
            if self.model_mean_type not in [ClusteredModelMeanType.EPSILON]:
                raise NotImplementedError(f"Model Mean Type {self.model_mean_type} not implemented!")

            target = {
                ClusteredModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["denoise_mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms: # CHECK - THIS TERM DOESN'T EXIST UNLESS WE LEARN SIGMA CASE? HOW IS THIS APPLICABLE IN OUR CASE? WE ARE LEARNING SIGMS IN SOME SENSE?
                terms["denoise_loss"] = terms["denoise_mse"] + terms["vb"]
            else:
                terms["denoise_loss"] = terms["denoise_mse"]
        else:
            raise NotImplementedError(self.denoise_loss_type)

        # CHECK - ADD WEIGHTS HERE FOR LOSSES?
        terms["loss"] = terms["guidance_loss"] + terms["denoise_loss"]
        return terms


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def _broadcast_tensor(t, broadcast_shape):
    while len(t.shape) < len(broadcast_shape):
        t = t[..., None]
    return t.expand(broadcast_shape)


def _broadcast_tensor_test_baseline(t, broadcast_shape, arr, timesteps):
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    t = t * res.unsqueeze(-1)
    while len(t.shape) < len(broadcast_shape):
        t = t[..., None]
    return t.expand(broadcast_shape)
