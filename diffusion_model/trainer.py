#-*- coding:utf-8 -*-
#


import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
from torch.utils import data
from pathlib import Path
from torch.optim import Adam,AdamW
from torchvision import transforms, utils
from PIL import Image
import nibabel as nib
import numpy as np
from tqdm import tqdm
from einops import rearrange
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
import os
import warnings
from PIL import Image
import random
from .DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D
from torchvision.utils import save_image
import logging
import sys
import os
from .vae import loss_vae
from .vae.AutoencoderKL import AutoencoderKL
from .vae.loss_vae import VAEPerceptualLoss

from diffusion_model.adversarial_loss import PatchAdversarialLoss
from diffusion_model.patchgan_discriminator import PatchDiscriminator



pyiqa_parent_dir = "/home/ttyy-user03/daima/un/diffusion_model"
sys.path.append(pyiqa_parent_dir)



try:
    import pyiqa2 as pyiqa
    print("pyiqa2 导入成功！")
except ImportError as e:
    print("导入失败:", e)
    print("当前 sys.path:", sys.path)


dwt = DWT_3D('haar')
idwt = IDWT_3D('haar')

from torch.nn.parallel import DistributedDataParallel as DDP



warnings.filterwarnings("ignore", category=UserWarning)


try:
    from apex import amp
    APEX_AVAILABLE = True
    print("APEX: ON")
except:
    APEX_AVAILABLE = False
    print("APEX: OFF")

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    while True:
        for data in dl:

            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        print("no use fp16")
        loss.backward(**kwargs)

# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

# gaussian diffusion trainer class
def _extract_into_tensor(arr, timesteps, broadcast_shape): #new

    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
    
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def get_named_eta_schedule( #new
        schedule_name = 'exponential',
        num_diffusion_timesteps=1000,
        min_noise_level = 0.2,
        etas_end=0.99,
        kappa=1.0,
        kwargs=None):
 
    if schedule_name == 'exponential':
        power = kwargs.get('power', 0.2)
        etas_start = min(min_noise_level / kappa, min_noise_level, math.sqrt(0.001))
        increaser = math.exp(1/(num_diffusion_timesteps-1)*math.log(etas_end/etas_start))
        base = np.ones([num_diffusion_timesteps, ]) * increaser
        power_timestep = np.linspace(0, 1, num_diffusion_timesteps, endpoint=True)**power
        power_timestep *= (num_diffusion_timesteps-1)
        sqrt_etas = np.power(base, power_timestep) * etas_start
        # print('sqrt_etas: ', sqrt_etas)
    
    else:
        raise ValueError(f"Unknow schedule_name {schedule_name}")

    return sqrt_etas


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        depth_size,
        channels = 1,
        timesteps = 250,
        loss_type = 'l2',
        betas = None,
        with_condition = False,
        with_pairwised = False,
        apply_bce = False,
        lambda_bce = 0.0,
        
        kappa = 2.5,
        
        schedule_kwargs = {"power": 0.2},
        
    ):
        super().__init__()
        
        sqrt_etas = get_named_eta_schedule(schedule_name = 'exponential',
        num_diffusion_timesteps=1000,
        min_noise_level = 0.2,
        etas_end=0.99,
        kwargs=schedule_kwargs,)
                                           
        self.channels = channels
        self.image_size = image_size
        self.depth_size = depth_size
        self.denoise_fn = denoise_fn
        self.with_condition = with_condition
        self.kappa = kappa
        self.perceptual_weight = 1
        # self.apply_bce = apply_bce
        # self.lambda_bce = lambda_bce
        
        self.sqrt_etas = sqrt_etas
        self.etas = sqrt_etas**2
        self.sf = 1
        assert len(self.etas.shape) == 1, "etas must be 1-D"
        assert (self.etas > 0).all() and (self.etas <= 1).all()

        self.num_timesteps = int(self.etas.shape[0])
        self.etas_prev = np.append(0.0, self.etas[:-1])
        self.alpha = self.etas - self.etas_prev
        self.f = np.ones(self.num_timesteps)
        self.f_prev = np.ones(self.num_timesteps)
        # self.loss_lpips = pyiqa.create_metric('lpips-vgg', as_loss=True)
        
        self.posterior_variance = kappa**2 * self.etas_prev / self.etas * self.alpha
        self.posterior_variance_clipped = np.append(
                self.posterior_variance[1], self.posterior_variance[1:]
                )
        self.posterior_log_variance_clipped = np.log(self.posterior_variance_clipped)
        self.posterior_mean_coef1 = self.etas_prev / self.etas
        self.posterior_mean_coef2 = self.alpha / self.etas

        self.vae = AutoencoderKL(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            num_channels=(32, 64, 128),
            latent_channels=16,
            num_res_blocks=1,
            norm_num_groups=16,
            attention_levels=(False, False, True),
            use_checkpointing=False
        )
        state_dict = torch.load('./checkpoint/vae/autoencoder_epoch_300.pth')
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.vae.load_state_dict(state_dict)
        self.vae.eval()
        self.vae.requires_grad_(False)
        
        self.perceptual_loss_fn = VAEPerceptualLoss(self.vae)

        

        # weight_loss_mse = 0.5 / self.posterior_variance_clipped * (self.alpha / self.etas)**2

        # self.weight_loss_mse = weight_loss_mse


    def q_posterior(self, x_start, x_t, t, c=None):
        x_hat = 0.
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t +
            extract(self.posterior_mean_coef3, t, x_t.shape) * x_hat
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    

    def set_seed(self, seed=42):
        torch.manual_seed(seed)            
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)    
            torch.cuda.manual_seed_all(seed) 
        np.random.seed(seed)                
        random.seed(seed)                    
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False
        
    @torch.no_grad()
    def q_posterior_mean_variance(self, x_start, x_t, y, y_hat, un, t):
        """
        Compute the mean and variance of the diffusion posterior:

        q(x_{t-1} | x_t, x_0, y, y_hat)
        = N(x_{t-1}; eta_{t-1}/eta_t * x_t + alpha_t/eta_t * x_0 + eta_{t-1} * (y_tilde_{t-1} - y_tilde_t), un^2 * kappa^2 * eta_{t-1}/eta_t * alpha_t * I)

        """
        assert x_start.shape == x_t.shape
        x_target = x_start
        # x_target = (1 - un) * x_start + un * y_hat
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_t
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_target
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape) * un ** 2
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        ) + 2 * torch.log(un)

        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
        
    def p_mean_variance(
        self, model, x_t, y, y_hat, un, t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None
    ):
      
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x_t.shape[:2]
        assert t.shape == (B,)

        model_output = model(x_t,t,y,y_hat)
        # model_output = self.denoise_fn(x_noisy,t,condition_tensors,mask)

        model_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape) * (un ** 2)
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape) + 2 * torch.log(un)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        pred_xstart = process_xstart(model_output)
            
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x_t, y=y, y_hat=y_hat, un=un, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x_t.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }
    def prior_sample(self, y, y_hat, un, noise=None):

        if noise is None:
            noise = torch.randn_like(y)

        t = torch.tensor([self.num_timesteps-1,] * y.shape[0], device=y.device).long()
        return (y + _extract_into_tensor(self.sqrt_etas * self.kappa, t, y.shape) * un * noise)
        
    @torch.no_grad()
    def sample(self, batch_size = 1, condition_tensors = None, mask_tensors = None, use_ddim = True, target_step=None):
        if use_ddim == True:
            un = self.cal_mse(condition_tensors, mask_tensors)
            x = torch.randn_like(condition_tensors)
            
            x_sample = self.prior_sample(condition_tensors,mask_tensors,un)
            indices = list(range(self.num_timesteps))[::-1]
            
   
            for i in indices:
                t = torch.tensor([i] * condition_tensors.shape[0], device=x.device)
                with torch.no_grad():
                    out = self.ddim_sample(
                        model=self.denoise_fn,
                        x=x_sample,
                        y=condition_tensors, y_hat=mask_tensors, un=un,
                        t=t
                    )
                    x_sample = out["sample"]
                    if target_step is not None and i == target_step:
                        print(f"{i}")
                        return out
                    # yield out
                    # x = out["sample"]

        else:
            image_size = self.image_size
            depth_size = self.depth_size
            channels = self.channels
            return self.p_sample_loop((batch_size, channels, depth_size, image_size, image_size), condition_tensors = condition_tensors, mask_tensors=mask_tensors)
        
    def ddim_sample(
        self,
        model,
        x, y, y_hat, un, t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        ddim_eta=0.0,
    ):
  
        out = self.p_mean_variance(
            model=model,
            x_t=x,
            y=y,
            y_hat=y_hat,
            un=un, 
            t=t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        pred_xstart = out["pred_xstart"]
        etas = _extract_into_tensor(self.etas, t, x.shape)
        etas_prev = _extract_into_tensor(self.etas_prev, t, x.shape)
        alpha = _extract_into_tensor(self.alpha, t, x.shape)
        sigma = ddim_eta * self.kappa * torch.sqrt(etas_prev / etas) * torch.sqrt(alpha)

        m_t = torch.sqrt(etas_prev / etas)

        k_t = (1 - etas_prev - (1 - etas) * m_t)

        y_t = (etas_prev - torch.sqrt(etas * etas_prev)) * y

        noise = torch.randn_like(x)
        mean_pred = (
            pred_xstart * k_t + x * m_t + y_t
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        print(f"---------------")
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}


    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.1):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img

    def q_sample(self, x_start, y, y_hat, un, t, noise=None): #new
        """
        # y:0.23T, x_start:3T 
        # un:the uncertainty map of y
        #return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape == un.shape
        y_tilde = y_hat + _extract_into_tensor(self.f, t, x_start.shape) * (y - y_hat)
        x_target = x_start
        # x_target = (1 - un) * x_start + un * y_hat
        return (
            x_target + _extract_into_tensor(self.etas, t, x_target.shape) * (y_tilde - x_target)
            + _extract_into_tensor(self.sqrt_etas * self.kappa, t, x_target.shape) * un * noise
        )

    def disp_loss(self, Z, tau=0.5):
        Z = F.normalize(Z, p=2, dim=1) 
        D = torch.pdist(Z, p=2)**2
        dispersion_term = torch.mean(torch.exp(-D / tau))
        loss = torch.log(dispersion_term)
        # print(f"{loss.item()}")


        return loss
    def cal_mse(self, condition_tensor=None,mask=None):
        
        diff = (mask - condition_tensor) / 2
        
        un_max = 0.1
        b_un =  0.4
        micro_uncertainty = torch.abs(diff).clamp_(0., un_max) / un_max
        micro_uncertainty = b_un + (1 - b_un) * micro_uncertainty
        
        # print(f"un:{micro_uncertainty.min()}he{micro_uncertainty.max()}")
            
        return micro_uncertainty

    def _scale_input(self, inputs,  t):
        var_un = 0.3
        std = torch.sqrt(_extract_into_tensor(self.etas, t, inputs.shape) * self.kappa**2 * var_un + 0.5**2)
        inputs_norm = inputs / std

        return inputs_norm
    
    def mean_flat(self, tensor):
 
        # return tensor.mean(dim=list(range(1, len(tensor.shape))))
        return tensor.abs().mean(dim=list(range(1, len(tensor.shape))))

    
    def p_losses(self, x_start, t, condition_tensors = None, mask=None,noise = None):        

        un = self.cal_mse(condition_tensors, mask)
        noise = torch.randn_like(x_start)
        terms = {}
        if self.with_condition:
            x_noisy = self.q_sample(x_start=x_start,y = condition_tensors, y_hat=mask, un = un, t=t, noise=noise)
            model_output = self.denoise_fn(self._scale_input(x_noisy,t),t,condition_tensors,mask)
            terms['loss'] = self.mean_flat(x_start - model_output)
            loss_mse = terms['loss']
            B, C, D, H, W = model_output.shape

            model_output = model_output.permute(0, 1, 3, 4, 2)
            x_start = x_start.permute(0, 1, 3, 4, 2)

            

            print(f"model_:{model_output.min()}he{model_output.max()}")
            
            model_output = torch.clamp(model_output, 0.0, 1.0)
            
            l_lpips = self.perceptual_loss_fn(model_output, x_start)
            print("output.requires_grad:", model_output.requires_grad)


            print(f"l_lpips:{l_lpips.item()}")
            print(f"l_mse:{loss_mse.item()}")
                        
            if torch.any(torch.isnan(l_lpips)):
                l_lpips = torch.nan_to_num(l_lpips, nan=0.0)
            l_lpips = l_lpips * self.perceptual_weight
            # l_lpips = l_lpips
            
            
            loss = 0.4*loss_mse+l_lpips
            # loss = l_lpips
        
        return loss,x_noisy,model_output,l_lpips

    

    def forward(self, x, condition_tensors, mask, *args, **kwargs):
        b, c, d, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}, but now h={h} and w={w}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, condition_tensors=condition_tensors,mask=mask, *args, **kwargs)


# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        dataloader,
        ema_decay = 0.995,
        image_size = 51,
        depth_size = 13,
        train_batch_size = 1,
        # train_lr = 2e-6,
        train_lr = 1e-4,
        train_num_steps = 1,
        gradient_accumulate_every = 2,
        fp16 = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder = './results',
        with_condition = False,
        with_pairwised = False,
        distributed=False,
        resume_path = None,
        rank=0):
        super().__init__()
        
        device = torch.device(f'cuda:{rank}')
        self.model = diffusion_model.to(device)
        
        print(f"dataloader:{len(dataloader)}")
        
        self.ema = EMA(ema_decay)
        # self.ema_model = copy.deepcopy(self.model)
        self.ema_model = diffusion_model.to(device)
        self.ema_model.load_state_dict(copy.deepcopy(self.model.state_dict()))

        if distributed:
            self.ema_model = DDP(self.ema_model, device_ids=[rank])
            
        self.update_ema_every = update_ema_every
        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.depth_size = depth_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.num_epochs = train_num_steps

        self.ds = dataset
        # self.sampler = DistributedSampler(self.ds, shuffle=True)
        
        self.dataloader = dataloader
        self.dl = cycle(self.dataloader)

        self.opt = AdamW(self.model.parameters(), lr=train_lr)
        self.train_lr = train_lr
        self.train_batch_size = train_batch_size
        self.with_condition = with_condition
        self.loss_history = []
        self.step = 0
        self.fp16 = fp16
        if fp16:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt, opt_level='O1')
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        self.distributed = distributed
        self.rank = rank

        self.resume_path = resume_path
        
        if resume_path is not None:
            self.load_checkpoint(resume_path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=f'cuda:{self.rank}')
        
        model = self.model.module if isinstance(self.model, DDP) else self.model
        ema_model = self.ema_model.module if isinstance(self.ema_model, DDP) else self.ema_model
    
        model.load_state_dict(checkpoint['model'])
        ema_model.load_state_dict(checkpoint['ema'])
        self.opt.load_state_dict(checkpoint['optimizer'])
        
        self.step = checkpoint['step']
        self.current_epoch = checkpoint['epoch']
        self.loss_history = checkpoint['loss_history']
        
        print(f"✅ 成功加载断点: epoch={self.current_epoch}, step={self.step}")


    def reset_parameters(self):
        model = self.model.module if isinstance(self.model, DDP) else self.model
        ema_model = self.ema_model.module if isinstance(self.ema_model, DDP) else self.ema_model
    
        model_state_dict = model.state_dict()
    
        ema_model.load_state_dict(model_state_dict, strict=False)


    def step_ema(self):
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

    def save(self, milestone):
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
            'loss_history': self.loss_history 
        }
        
        os.makedirs(self.results_folder/'model', exist_ok=True)
        torch.save(data, str(self.results_folder/f'model/model-{milestone}.pt'))
        print('保存权重')

    
    def train(self):
        backwards = partial(loss_backwards, self.fp16)
        start_time = time.time()
        
        if self.rank == 0:
            logging.basicConfig(
                filename='training_loss.log',
                level=logging.INFO,
                format='%(asctime)s - %(message)s',
                filemode='a'
            )

        start_epoch = getattr(self, 'current_epoch', 0)
        for epoch in range(start_epoch, self.num_epochs):
            if self.distributed and hasattr(self.dataloader.sampler,'set_epoch'):
                self.dataloader.sampler.set_epoch(epoch)

            if (epoch + 1) % self.save_and_sample_every == 0 and self.rank == 0:
                self.current_epoch = epoch + 1  
                self.save((epoch + 1) // self.save_and_sample_every)

            print(f"Epoch{epoch+1}/{self.num_epochs}")
            
            print(f"dataloader:{len(self.dataloader)}")
            
            accumulate_steps = self.gradient_accumulate_every
            epoch_loss = 0.0
            batch_count = 0.0

            Z_buffer = []
            loss_buffer = []
            disp_interval = 4  
            disp_weight = 0.05
        
            for i,data in enumerate(self.dataloader):
                self.step+=1
                if self.with_condition:
                    input_tensors = data['input'].float().to(f'cuda:{self.rank}')
                    target_tensors = data['target'].float().to(f'cuda:{self.rank}')
                    mask_tensors = data['mask'].float().to(f'cuda:{self.rank}')
                    loss,_,_,l_lpips = self.model(target_tensors, input_tensors, mask_tensors)
                
                    
                    print(f"loss:{i}:{loss.item()}")
                    if self.rank == 0:
                        logging.info( 
                            f"Epoch {epoch+1}/{self.num_epochs}, "
                            f"Batch {i}/{len(self.dataloader)}, "
                            f"Loss: {loss.item():.6f} l_pips:{l_lpips.item():.6f}"
                        )
                    epoch_loss += loss.item()
                    batch_count +=1
                    
                else:
                    data = data.cuda()
                    loss = self.model(data)
    
                loss.backward()
                # for name, param in self.model.named_parameters():
                #         if param.requires_grad:
                #             if param.grad is not None:
                #                 print(f"{name}: grad mean = {param.grad.abs().mean():.6f}")
                #             else:
                #                 print(f"{name}: grad is None")
               
                if (i+1) % accumulate_steps == 0:
                    
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad /= self.gradient_accumulate_every
                        
                    self.opt.step()
                    self.opt.zero_grad()
                   
                    if self.step >= self.step_start_ema and self.step % self.update_ema_every == 0:
                        self.step_ema()

                   
                        
            avg_loss = epoch_loss / batch_count
            # self.loss_history.append(avg_loss)
            # print(f"....{self.loss_history}")
            self.plot_loss(avg_loss)
                        
        if self.distributed:
            torch.distributed.barrier()        

    def plot_loss(self,avg_loss):
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, 'b-', linewidth=1, label='Training Loss')
        plt.title(f"Training Loss (Epoch{self.num_epochs})")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig('./training_loss interim.png', dpi=100, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
   
