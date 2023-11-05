import enum
import math

import numpy as np
import torch as th

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood


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
        self.model_var_type = model_Var_type
        if self.model_var_type not in [ClusteredModelVarType.FIXED_SMALL, ClusteredModelVarType.FIXED_LARGE]:
            raise NotImplementedError(f"Model Var Type {self.model_Var_type} not implemented!")
        self.guidance_loss_type = gudiance_loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.sqrt_alphas = np.sqrt(self.alphas)
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(self.alphas_cumprod_prev)
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t, mu_bar_y_t, sigma_bar_y_t):
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        ) + mu_bar_y_t
        variance = sigma_bar_y_t ** 2
        log_variance = np.log(variance)
        return mean, variance, log_variance
    
    def q_sample(self, x_start, t, mu_bar_y_t, sigma_bar_y_t, noise=None):
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + mu_bar_y_t
            + sigma_bar_y_t * noise
        )
    
    def posterior_variance(self, x_t, t, sigma_bar_y_t, sigma_bar_y_tm1):
        return (sigma_bar_y_tm1 ** 2) * (1 - _extract_into_tensor(self.alphas, t, x_t.shape) * ((sigma_bar_y_tm1 / sigma_bar_y_t) ** 2))
    
    def q_posterior_variance(self, x_t, t, sigma_bar_y_t, sigma_bar_y_tm1, sigma_bar_y_tp1 = None):
        posterior_variance = self.posterior_variance(x_t, t, sigma_bar_y_t, sigma_bar_y_tm1)
        if (t == 0):
            # log calculation clipped because the posterior variance is 0 at the
            # beginning of the diffusion chain.
            assert sigma_bar_y_tp1 is not None # Need t+1 to compute log variance in this case
        posterior_log_variance_clipped = np.log(self.posterior_variance(x_t, t+1, sigma_bar_y_tp1, sigma_bar_y_t)) if t == 0 else np.log(posterior_variance)
        assert (
            posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_variance, posterior_log_variance_clipped
    
    def q_posterior_mean_variance(self, x_start, x_t, t, mu_bar_y_t, mu_bar_y_tm1, sigma_bar_y_t, sigma_bar_y_tm1, sigma_bar_y_tp1 = None):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod_prev, t, x_t.shape) * x_start
            + mu_bar_y_tm1
            + (
                _extract_into_tensor(self.sqrt_alphas, t, x_t.shape)
                * (sigma_bar_y_tm1 / sigma_bar_y_t)
                * (x_t - _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_start - mu_bar_y_t)
            )
        )
        posterior_variance, posterior_log_variance_clipped = self.q_posterior_variance(x_t, t, sigma_bar_y_t, sigma_bar_y_tm1, sigma_bar_y_tp1)
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, denoise_model, x, t, mu_bar_y_t, mu_bar_y_tm1, sigma_bar_y_t, sigma_bar_y_tm1, sigma_bar_y_tp1 = None, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        model_kwargs = None # CHECK - NO NEED OF CLASS CONDITIONING, SO SETTING THIS TO NONE
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = denoise_model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.FIXED_SMALL, ModelVarType.FIXED_LARGE]:
            model_variance, model_log_variance = {
                # CHECK - FOR FIXED_LARGE, AND FIXED_SMALL; THERE IS A SMALL DIFFERENCE AT T==0, IN TERMS OF VARIANCE NOT LOG VARIANCE VALUE. IS THIS CORRECT?
                # CHECK - WRITE THIS EFFICIENTLY?
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (self.q_posterior_variance(x_t, t+1, sigma_bar_y_tp1, sigma_bar_y_t, None)) if (t == 0) else \
                     (sigma_bar_y_t ** 2 - _extract_into_tensor(self.alphas, t, x_t) * sigma_bar_y_tm1 ** 2, np.log(sigma_bar_y_t ** 2 - _extract_into_tensor(self.alphas, t, x_t) * sigma_bar_y_tm1 ** 2)),
                ModelVarType.FIXED_SMALL: (self.q_posterior_variance(x_t, t, sigma_bar_y_t, sigma_bar_y_tm1, sigma_bar_y_tp1)),
            }[self.model_var_type]
        else:
            raise NotImplementedError(f"Model Var Type {self.model_Var_type} not implemented!")

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x
        
        if self.model_mean_type == ClusteredModelMeanType.EPSILON:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_outpu,t, mu_bar_y_t=mu_bar_y_t, sigma_bar_y_t=sigma_bar_y_t)
            )
            model_mean, _, _ = self.q_posterior_mean_variance(
                pred_xstart, x, t, mu_bar_y_t, mu_bar_y_tm1, sigma_bar_y_t, sigma_bar_y_tm1, sigma_bar_y_tp1
            )
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
        ) / sigma_bar_y_t

    def _predict_xstart_from_eps(self, x_t, t, eps, mu_bar_y_t, sigma_bar_y_t):
        assert x_t.shape == eps.shape
        return (
            (x_t - mu_bar_y_t - sigma_bar_y_t * eps) / _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
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
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                # CHECK - WRITE CODE TO GET GUIDANCE VALUES. HERE MODEL IS THE COMPLETE MODEL?
                out = self.p_sample(
                    model,
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
    
    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        # CHECK - LOSS IS NOT IMPLEMENTED YET.
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

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
