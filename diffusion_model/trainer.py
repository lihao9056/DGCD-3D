#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Core diffusion model and trainer for DGCD-3D.

Implements a conditional 3D diffusion model for DWI image enhancement,
with support for anatomical mask guidance and perceptual losses.
"""

import math
import copy
import time
import logging
import warnings
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from einops import rearrange
from .vae.unetloss import UNETPerceptualLoss

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# Utility Functions
# ============================================================================

def exists(x):
    """Check if a value exists (is not None)."""
    return x is not None


def default(val, d):
    """Return value if exists, otherwise call/function d."""
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    """Infinitely cycle through a dataloader."""
    while True:
        for data in dl:
            yield data


def extract(a, t, x_shape):
    """Extract values from array at indices t and reshape."""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def extract_into_tensor(arr, timesteps, broadcast_shape):
    """Extract values from 1-D numpy array for batch indices."""
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def noise_like(shape, device, repeat=False):
    """Generate noise tensor."""
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def loss_backwards(fp16, loss, optimizer, **kwargs):
    """Backward pass with optional mixed precision."""
    if fp16:
        try:
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward(**kwargs)
        except ImportError:
            warnings.warn("APEX not available, falling back to normal precision")
            loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)


# ============================================================================
# Exponential Moving Average (EMA)
# ============================================================================

class EMA:
    """Exponential Moving Average for model weights."""
    
    def __init__(self, beta: float):
        super().__init__()
        self.beta = beta
    
    def update_model_average(self, ma_model: nn.Module, current_model: nn.Module):
        """Update moving average model weights."""
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)
    
    def update_average(self, old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        """Compute weighted average."""
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


# ============================================================================
# Noise Schedule
# ============================================================================

def get_named_eta_schedule(
    schedule_name: str = 'exponential',
    num_diffusion_timesteps: int = 1000,
    min_noise_level: float = 0.001,
    etas_end: float = 0.99,
    kappa: float = 2.0,  
    kwargs: Optional[Dict] = None
) -> np.ndarray:
    """
    Generate noise schedule for diffusion process.
    
    Args:
        schedule_name: Type of schedule ('exponential').
        num_diffusion_timesteps: Number of diffusion steps.
        min_noise_level: Minimum noise level.
        etas_end: Maximum eta value.
        kappa: Scaling factor.
        kwargs: Additional schedule parameters.
        
    Returns:
        Array of sqrt(eta) values.
    """
    if schedule_name == 'exponential':
        power = kwargs.get('power', 0.2) if kwargs else 0.2
        etas_start = min(min_noise_level / kappa, min_noise_level, math.sqrt(0.001))
        increaser = math.exp(1 / (num_diffusion_timesteps - 1) * math.log(etas_end / etas_start))
        base = np.ones([num_diffusion_timesteps, ]) * increaser
        power_timestep = np.linspace(0, 1, num_diffusion_timesteps, endpoint=True) ** power
        power_timestep *= (num_diffusion_timesteps - 1)
        sqrt_etas = np.power(base, power_timestep) * etas_start
        return sqrt_etas
    else:
        raise ValueError(f"Unknown schedule_name: {schedule_name}")


# ============================================================================
# Gaussian Diffusion Model
# ============================================================================

class GaussianDiffusion(nn.Module):
    """
    3D Conditional Gaussian Diffusion Model.
    
    Implements a diffusion model with:
    - Conditional guidance from low-field images
    - Anatomical mask constraints
    - Uncertainty-aware noise scheduling
    - Perceptual loss supervision
    """
    
    def __init__(
        self,
        denoise_fn: nn.Module,
        image_size: int,
        depth_size: int,
        channels: int = 1,
        timesteps: int = 500,
        loss_type: str = 'l2',
        with_condition: bool = False,
        kappa: float = 0.8,
        schedule_kwargs: Optional[Dict] = None,
        perceptual_weight: float = 0.5
    ):
        super().__init__()
        
        self.channels = channels
        self.image_size = image_size
        self.depth_size = depth_size
        self.denoise_fn = denoise_fn
        self.with_condition = with_condition
        self.kappa = kappa
        self.perceptual_weight = perceptual_weight
        
        # Noise schedule
        schedule_kwargs = schedule_kwargs or {"power": 0.2}
        self.sqrt_etas = get_named_eta_schedule(
            schedule_name='exponential',
            num_diffusion_timesteps=timesteps,
            min_noise_level=0.2,
            etas_end=0.99,
            kwargs=schedule_kwargs
        )
        
        self.etas = self.sqrt_etas ** 2
        self.num_timesteps = int(self.etas.shape[0])
        
        # Posterior parameters
        self.etas_prev = np.append(0.0, self.etas[:-1])
        self.alpha = self.etas - self.etas_prev
        self.f = np.ones(self.num_timesteps)
        
        self.posterior_variance = kappa ** 2 * self.etas_prev / self.etas * self.alpha
        self.posterior_variance_clipped = np.append(
            self.posterior_variance[1], self.posterior_variance[1:]
        )
        self.posterior_log_variance_clipped = np.log(self.posterior_variance_clipped)
        self.posterior_mean_coef1 = self.etas_prev / self.etas
        self.posterior_mean_coef2 = self.alpha / self.etas
        
        # Perceptual loss
        self.perceptual_loss_fn = UNETPerceptualLoss(spatial_dims=3)
        
        # Weight for loss computation
        self.weight_loss_mse = 0.5 / self.posterior_variance_clipped * (self.alpha / self.etas) ** 2
    
    def cal_mse(self, condition_tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Calculate uncertainty map from condition and mask.
        
        Args:
            condition_tensor: Low-field image tensor.
            mask: Anatomical mask tensor.
            
        Returns:
            Uncertainty map tensor.
        """
        return torch.abs(condition_tensor * (1 - mask))
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        un: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample from q(x_t | x_0, y, y_hat).
        
        Forward diffusion process with conditional guidance.
        
        Args:
            x_start: Original image (3T).
            y: Condition image (0.23T).
            y_hat: Mask.
            un: Uncertainty map.
            t: Timestep.
            noise: Optional noise tensor.
            
        Returns:
            Noisy image at timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        y_tilde = y_hat + extract_into_tensor(self.f, t, x_start.shape) * (y - y_hat)
        # x_target = (1 - un) * x_start + un * y_hat
        x_target = x_start
        return (
            x_target 
            + extract_into_tensor(self.etas, t, x_target.shape) * (y_tilde - x_target)
            + extract_into_tensor(self.sqrt_etas * self.kappa, t, x_target.shape) * un * noise
        )
    
    def p_mean_variance(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        un: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute mean and variance of p(x_{t-1} | x_t).
        
        Args:
            model: Denoising network.
            x_t: Noisy image at timestep t.
            y: Condition image.
            y_hat: Mask.
            un: Uncertainty map.
            t: Timestep.
            clip_denoised: Whether to clip denoised prediction.
            
        Returns:
            Dictionary with mean, variance, log_variance, pred_xstart.
        """
        # Predict velocity
        v_pred = model(x_t, t, y, y_hat)
        
        beta_t = extract_into_tensor(self.etas, t, x_t.shape)
        alpha_t = 1.0 - beta_t
        sigma_t = extract_into_tensor(self.sqrt_etas * self.kappa, t, x_t.shape) * un
        y_tilde = y_hat + extract_into_tensor(self.f, t, x_t.shape) * (y - y_hat)
        
        x_t_center = x_t - beta_t * y_tilde
        denom = alpha_t.pow(2) + sigma_t.pow(2) + 1e-8
        x0_pred = (alpha_t * x_t_center - sigma_t * v_pred) / denom
        
        if clip_denoised:
            x0_pred = x0_pred.clamp(0, 1)
        
        # Compute posterior mean and variance
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_t
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x0_pred
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape) * un ** 2
        posterior_log_variance_clipped = (
            extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape) 
            + 2 * torch.log(un)
        )
        
        return {
            "mean": posterior_mean,
            "variance": posterior_variance,
            "log_variance": posterior_log_variance_clipped,
            "pred_xstart": x0_pred,
        }
    
    def ddim_sample(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        un: torch.Tensor,
        t: torch.Tensor,
        ddim_eta: float = 0.0,
        clip_denoised: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        DDIM sampling step.
        
        Args:
            model: Denoising network.
            x: Current noisy image.
            y: Condition image.
            y_hat: Mask.
            un: Uncertainty map.
            t: Current timestep.
            ddim_eta: DDIM noise parameter.
            clip_denoised: Whether to clip predictions.
            
        Returns:
            Dictionary with sampled image and predicted x_0.
        """
        out = self.p_mean_variance(
            model=model, x_t=x, y=y, y_hat=y_hat, un=un, t=t,
            clip_denoised=clip_denoised
        )
        pred_xstart = out["pred_xstart"]
        
        etas = extract_into_tensor(self.etas, t, x.shape)
        etas_prev = extract_into_tensor(self.etas_prev, t, x.shape)
        alpha = extract_into_tensor(self.alpha, t, x.shape)
        sigma = ddim_eta * self.kappa * torch.sqrt(etas_prev / etas) * torch.sqrt(alpha)
        
        m_t = torch.sqrt(etas_prev / etas)
        k_t = (1 - etas_prev - (1 - etas) * m_t)
        y_t = (etas_prev - torch.sqrt(etas * etas_prev)) * y
        
        noise = torch.randn_like(x)
        mean_pred = pred_xstart * k_t + x * m_t + y_t
        
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = mean_pred + nonzero_mask * sigma * noise
        
        return {"sample": sample, "pred_xstart": pred_xstart}
    
    def prior_sample(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        un: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample from prior p(x_T | y, y_hat).
        
        Args:
            y: Condition image.
            y_hat: Mask.
            un: Uncertainty map.
            noise: Optional noise tensor.
            
        Returns:
            Initial noisy sample.
        """
        if noise is None:
            noise = torch.randn_like(y)
        
        t = torch.tensor([self.num_timesteps - 1] * y.shape[0], device=y.device).long()
        return y + extract_into_tensor(self.sqrt_etas * self.kappa, t, y.shape) * un * noise
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        condition_tensors: Optional[torch.Tensor] = None,
        mask_tensors: Optional[torch.Tensor] = None,
        use_ddim: bool = True,
        target_step: Optional[int] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Generate samples using DDIM or DDPM.
        
        Args:
            batch_size: Number of samples.
            condition_tensors: Conditional input (low-field images).
            mask_tensors: Anatomical masks.
            use_ddim: Use DDIM sampling (faster).
            target_step: Return intermediate result at this step.
            
        Returns:
            Dictionary with final sample and predictions.
        """
        if use_ddim:
            un = self.cal_mse(mask_tensors, condition_tensors)
            x = torch.randn_like(condition_tensors)
            x_sample = self.prior_sample(condition_tensors, mask_tensors, un)
            
            indices = list(range(self.num_timesteps))[::-1]
            
            with tqdm(indices, desc="DDIM Sampling", unit="step", 
                     ncols=100, mininterval=0.1, dynamic_ncols=True,
                     leave=False) as pbar:
                for i in pbar:
                    t = torch.tensor([i] * condition_tensors.shape[0], device=x.device)
                    out = self.ddim_sample(
                        model=self.denoise_fn,
                        x=x_sample,
                        y=condition_tensors,
                        y_hat=mask_tensors,
                        un=un,
                        t=t
                    )
                    x_sample = out["sample"]
                    
                    if target_step is not None and i == target_step:
                        print(f"Reached step {i}")
                        return out
            
            return out
        else:
            # DDPM sampling
            image_size = self.image_size
            depth_size = self.depth_size
            channels = self.channels
            return self.p_sample_loop(
                (batch_size, channels, depth_size, image_size, image_size),
                condition_tensors=condition_tensors,
                mask_tensors=mask_tensors
            )
    
    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        condition_tensors: torch.Tensor,
        mask: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute training loss.
        
        Args:
            x_start: Target image (3T).
            t: Timestep.
            condition_tensors: Condition image (0.23T).
            mask: Anatomical mask.
            noise: Optional noise tensor.
            
        Returns:
            Tuple of (total_loss, noisy_image, predicted_x0, lpips_loss).
        """
        un = self.cal_mse(condition_tensors, mask)
        
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Forward diffusion
        x_noisy = self.q_sample(
            x_start=x_start, y=condition_tensors, y_hat=mask, 
            un=un, t=t, noise=noise
        )
        
        # Compute targets for v-prediction
        y_tilde = mask + extract_into_tensor(self.f, t, x_start.shape) * (condition_tensors - mask)
        beta_t = extract_into_tensor(self.etas, t, x_start.shape)
        alpha_t = 1.0 - beta_t
        sigma_t = extract_into_tensor(self.sqrt_etas * self.kappa, t, x_start.shape) * un
        
        x_t_center = x_noisy - beta_t * y_tilde
        v_target = alpha_t * noise - sigma_t * x_start
        
        # Predict velocity
        v_pred = self.denoise_fn(x_noisy, t, condition_tensors, mask)
        
        # V-prediction loss
        loss_v = F.mse_loss(v_pred, v_target)
        
        # Recover x0 prediction
        denom = alpha_t.pow(2) + sigma_t.pow(2) + 1e-8
        x0_pred = (alpha_t * x_t_center - sigma_t * v_pred) / denom
        x0_pred = x0_pred.clamp(0, 1)
        
        # Time-dependent weight (higher weight for smaller t)
        with torch.no_grad():
            w_small_t = 1.0 - (t.float() / (self.num_timesteps - 1)).view(-1, *([1] * (x_start.ndim - 1)))
            w_small_t = w_small_t.detach()
        
        # X0 prediction loss
        loss_x0 = F.mse_loss(x0_pred, x_start)
        
        # Perceptual loss
        lpips_loss = self.perceptual_loss_fn(x0_pred, x_start)
        
        # Combine losses
        lambda_x0 = 0.2
        lambda_lpips = 0.2
        loss = loss_v + (lambda_x0 * w_small_t * loss_x0).mean() + (lambda_lpips * w_small_t * lpips_loss).mean()
        
        return loss, x_noisy, x0_pred, lpips_loss
    
    def forward(
        self,
        x: torch.Tensor,
        condition_tensors: torch.Tensor,
        mask: torch.Tensor,
        *args,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        
        Forward pass - compute loss and predictions.
        
        Args:
            x: Target image (3T).
            condition_tensors: Condition image (0.23T).
            mask: Anatomical mask.
            
        Returns:
            Tuple of (loss, noisy_image, predicted_x0, lpips_loss).
        """
        

        b, c, d, h, w = x.shape
        device = x.device
        
        assert h == self.image_size and w == self.image_size, \
            f"Image size must be {self.image_size}, got h={h}, w={w}"
        
        t = torch.randint(0, self.num_timesteps, (b,), device=x.device).long()
        return self.p_losses(x, t, condition_tensors=condition_tensors, mask=mask, *args, **kwargs)


# ============================================================================
# Trainer
# ============================================================================

class Trainer:
    """
    Trainer for DGCD-3D diffusion model.
    
    Handles training loop, checkpointing, EMA updates, and distributed training.
    """
    
    def __init__(
        self,
        diffusion_model: GaussianDiffusion,
        dataset: Dataset,
        dataloader: DataLoader,
        image_size: int = 256,
        depth_size: int = 13,
        train_batch_size: int = 1,
        train_lr: float = 1e-4,
        train_num_steps: int = 1000,
        gradient_accumulate_every: int = 2,
        ema_decay: float = 0.995,
        fp16: bool = False,
        step_start_ema: int = 2000,
        update_ema_every: int = 10,
        save_and_sample_every: int = 1000,
        results_folder: str = './results',
        with_condition: bool = False,
        distributed: bool = False,
        rank: int = 0,
        resume_path: Optional[str] = None
    ):
        """
        Initialize trainer.
        
        Args:
            diffusion_model: GaussianDiffusion model.
            dataset: Training dataset.
            dataloader: Data loader.
            image_size: Image spatial size.
            depth_size: Depth size.
            train_batch_size: Batch size.
            train_lr: Learning rate.
            train_num_steps: Number of training epochs.
            gradient_accumulate_every: Gradient accumulation steps.
            ema_decay: EMA decay rate.
            fp16: Use mixed precision.
            step_start_ema: Start EMA after this many steps.
            update_ema_every: Update EMA every N steps.
            save_and_sample_every: Save checkpoint every N epochs.
            results_folder: Folder to save results.
            with_condition: Use conditional generation.
            distributed: Use distributed training.
            rank: Process rank.
            resume_path: Path to resume from.
        """
        super().__init__()
        
        device = torch.device(f'cuda:{rank}')
        self.model = diffusion_model.to(device)
        
        # EMA model
        self.ema = EMA(ema_decay)
        self.ema_model = diffusion_model.to(device)
        self.ema_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        
        if distributed:
            self.ema_model = DDP(self.ema_model, device_ids=[rank])
        
        # Training parameters
        self.update_ema_every = update_ema_every
        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.image_size = image_size
        self.depth_size = depth_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.num_epochs = train_num_steps
        
        # Data
        self.ds = dataset
        self.dataloader = dataloader
        self.dl = cycle(self.dataloader)
        
        # Optimizer
        self.opt = AdamW(self.model.parameters(), lr=train_lr)
        self.train_lr = train_lr
        self.with_condition = with_condition
        
        # Training state
        self.step = 0
        self.fp16 = fp16
        self.distributed = distributed
        self.rank = rank
        self.current_epoch = 0
        
        # Checkpointing
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        self.resume_path = resume_path
        
        if resume_path is not None:
            self.load_checkpoint(resume_path)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=f'cuda:{self.rank}')
        
        model = self.model.module if isinstance(self.model, DDP) else self.model
        ema_model = self.ema_model.module if isinstance(self.ema_model, DDP) else self.ema_model
        
        model.load_state_dict(checkpoint['model'])
        ema_model.load_state_dict(checkpoint['ema'])
        self.opt.load_state_dict(checkpoint['optimizer'])
        
        self.step = checkpoint.get('step', 0)
        self.current_epoch = checkpoint.get('epoch', 0)
        
        print(f"✅ Successfully loaded checkpoint: epoch={self.current_epoch}, step={self.step}")
    
    def reset_parameters(self):
        """Reset EMA model parameters to match main model."""
        model = self.model.module if isinstance(self.model, DDP) else self.model
        ema_model = self.ema_model.module if isinstance(self.ema_model, DDP) else self.ema_model
        
        ema_model.load_state_dict(model.state_dict(), strict=False)
    
    def step_ema(self):
        """Update EMA model."""
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        
        self.ema.update_model_average(self.ema_model, self.model)
        
        model = self.model.module if isinstance(self.model, DDP) else self.model
        ema_model = self.ema_model.module if isinstance(self.ema_model, DDP) else self.ema_model
        self.ema.update_model_average(ema_model, model)
        
        if self.distributed:
            for param in self.ema_model.parameters():
                torch.distributed.broadcast(param.data, src=0)
    
    def save(self, milestone: int):
        """Save training checkpoint."""
        if self.rank != 0:
            return
        
        model = self.model.module if isinstance(self.model, DDP) else self.model
        ema_model = self.ema_model.module if isinstance(self.ema_model, DDP) else self.ema_model
        
        data = {
            'step': self.step,
            'epoch': self.current_epoch,
            'model': model.state_dict(),
            'ema': ema_model.state_dict(),
            'optimizer': self.opt.state_dict(),
        }
        
        os.makedirs(self.results_folder / 'model', exist_ok=True)
        torch.save(data, str(self.results_folder / f'model/model-{milestone}.pt'))
        print(f"Saved checkpoint: model-{milestone}.pt")
    
    def train(self):
        """Main training loop."""
        if self.rank == 0:
            logging.basicConfig(
                filename='training_loss.log',
                level=logging.INFO,
                format='%(asctime)s - %(message)s',
                filemode='a'
            )
        
        start_epoch = self.current_epoch
        
        for epoch in range(start_epoch, self.num_epochs):
            if self.distributed and hasattr(self.dataloader.sampler, 'set_epoch'):
                self.dataloader.sampler.set_epoch(epoch)
            
            # Save checkpoint periodically
            if (epoch + 1) % self.save_and_sample_every == 0 and self.rank == 0:
                self.current_epoch = epoch + 1
                self.save((epoch + 1) // self.save_and_sample_every)
            
            if self.rank == 0:
                print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            epoch_loss = 0.0
            batch_count = 0
            
            for i, data in enumerate(self.dataloader):
                self.step += 1
                
                if self.with_condition:
                    input_tensors = data['input'].float().to(f'cuda:{self.rank}')
                    target_tensors = data['target'].float().to(f'cuda:{self.rank}')
                    mask_tensors = data['mask'].float().to(f'cuda:{self.rank}')
                    
                    # Forward pass
                    loss, _, _, lpips_loss = self.model(
                        target_tensors, input_tensors, mask_tensors
                    )
                    
                    # Backward pass
                    loss.backward()
                    
                    if self.rank == 0:
                        print(f"  Batch {i}: loss={loss.item():.4f}")
                        logging.info(
                            f"Epoch {epoch+1}/{self.num_epochs}, "
                            f"Batch {i}/{len(self.dataloader)}, "
                            f"Loss: {loss.item():.6f}"
                        )
                    
                    epoch_loss += loss.item()
                    batch_count += 1
                
                # Gradient accumulation and optimizer step
                if (i + 1) % self.gradient_accumulate_every == 0:
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad /= self.gradient_accumulate_every
                    
                    self.opt.step()
                    self.opt.zero_grad()
                    
                    # EMA update
                    if self.step >= self.step_start_ema and self.step % self.update_ema_every == 0:
                        self.step_ema()
            
            # Log epoch statistics
            if batch_count > 0 and self.rank == 0:
                avg_loss = epoch_loss / batch_count
                print(f"  Average loss: {avg_loss:.6f}")
        
        if self.distributed:
            torch.distributed.barrier()
        
        if self.rank == 0:
            print("Training completed!")