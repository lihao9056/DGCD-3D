
    

#-*- coding:utf-8 -*-
# +
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
from diffusion_model.trainer_brats import GaussianDiffusion, Trainer
from diffusion_model.unet_brats import create_model
from dataset import  NiftiPairImageGenerator
import argparse
import torch
from torch.utils import data
from diffusion_model.adversarial_loss import PatchAdversarialLoss
from diffusion_model.patchgan_discriminator import PatchDiscriminator

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import os 
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
# -

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_folder', type=str, default="./dataset/data/0.23T_DWI")
parser.add_argument('-t', '--target_folder', type=str, default="./dataset/data/3T_DWI")
parser.add_argument('-m', '--mask_folder', type=str, default="./dataset/data/mask_train_0.23T")
parser.add_argument('--input_size', type=int, default=256)
parser.add_argument('--depth_size', type=int, default=13)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=2)
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--epochs', type=int, default=800) # epochs parameter specifies the number of training iterations
parser.add_argument('--save_pt_dir', type=str, default="./checkpoint/model-un-mask")

parser.add_argument('--timesteps', type=int, default=250)
parser.add_argument('--save_and_sample_every', type=int, default=10)
parser.add_argument('--with_condition', type=str, default='True')
# parser.add_argument('-r', '--resume_path', type=str, default= './checkpoint/model-un-mask/model/model-40.pt')
parser.add_argument('-r', '--resume_path', type=str, default= None)
parser.add_argument('--local_rank', type=int, default=0)
# 添加此行

args = parser.parse_args()



input_folder = args.input_folder
target_folder = args.target_folder
mask_folder = args.mask_folder
input_size = args.input_size
depth_size = args.depth_size
num_channels = args.num_channels
num_res_blocks = args.num_res_blocks
save_and_sample_every = args.save_and_sample_every
with_condition = args.with_condition
save_pt_dir = args.save_pt_dir
resume_path = args.resume_path

num = [file for file in os.listdir(target_folder)]
print(f"------len:{len(num)}")

# dist.init_process_group(backend='nccl', init_method='env://')
# torch.cuda.set_device(args.local_rank)
# device = torch.device('cuda', args.local_rank)
args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))

torch.cuda.set_device(args.local_rank)
device = torch.device('cuda', args.local_rank)

dist.init_process_group(
    backend='nccl',
    init_method='env://'
)

# input tensor: (B, 1, H, W, D)  value range: [-1, 1]
transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: t.permute(2, 0, 1)),
    Lambda(lambda t: t.unsqueeze(0)),
    # Lambda(lambda t: t.transpose(3, 1)),
])

input_transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: t.permute(2, 0, 1)),
    Lambda(lambda t: t.unsqueeze(0)),
])

if with_condition:
    dataset = NiftiPairImageGenerator(
        input_folder,
        target_folder,
        mask_folder,
        input_size=input_size,
        depth_size=depth_size,
        transform=input_transform if with_condition else transform,
        target_transform=transform,
        # full_channel_mask=True
    )
else:
    print("Please modify your code to unconditional generation")


sampler = DistributedSampler(
    dataset,
    num_replicas=int(os.environ.get('WORLD_SIZE', 1)),
    rank=args.local_rank,
    shuffle=True
)


dataloader = data.DataLoader(
    dataset,
    batch_size=args.batchsize,
    sampler=sampler,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    shuffle=False
)


in_channels = 1 if with_condition or with_pairwised else 1
out_channels = 1
print(f"in_channel:{in_channels}")

model = create_model(
        args.input_size,
        args.num_channels,
        args.num_res_blocks,
        in_channels=in_channels,
        out_channels=1
    ).to(device)

model = DDP(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True
    )

# discriminator = PatchDiscriminator(spatial_dims=3, num_layers_d=3, num_channels=32, in_channels=1, out_channels=1).to(device)
# discriminator = DDP(
#     discriminator,
#     device_ids=[args.local_rank],
#     output_device=args.local_rank
# )

diffusion = GaussianDiffusion(
    model,
    image_size = input_size,
    depth_size = depth_size,
    timesteps = args.timesteps,   # number of steps
    loss_type = 'l2',    # L1 or L2
    with_condition=with_condition,
    channels=out_channels
).cuda()

trainer = Trainer(
    diffusion,
    dataset,
    dataloader,
    depth_size = depth_size,
    train_batch_size = args.batchsize,
    train_lr = 1e-4,
    train_num_steps = args.epochs,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False,#True,                       # turn on mixed precision training with apex
    save_and_sample_every = save_and_sample_every,
    results_folder = save_pt_dir,
    with_condition=with_condition,
    distributed=True,
    rank=args.local_rank,
    resume_path=args.resume_path

)

trainer.train()