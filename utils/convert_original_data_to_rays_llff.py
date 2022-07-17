import os
import time
import sys

sys.path.insert(0, './')

import numpy as np
import configargparse
import torch
import imageio
import json
import cv2

from dataset.load_llff import load_llff_data


def to_tensor(x):
    return x.to('cpu') if isinstance(
        x, torch.Tensor) else torch.Tensor(x).to('cpu')


def to_array(x):
    return x if isinstance(x, np.ndarray) else x.data.cpu().numpy()


def get_rays(H, W, focal, c2w, trans_origin='', focal_scale=1):
    focal *= focal_scale
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                          torch.linspace(
                              0, H - 1,
                              H))  # pytorch's meshgrid has indexing='ij'
    i, j = i.t(), j.t()
    dirs = torch.stack(
        [(i - W * .5) / focal, -(j - H * .5) / focal, -torch.ones_like(i)],
        -1)  # TODO-@mst: check if this H/W or W/H is in the right order
    # Rotate ray directions from camera frame to the world frame
    # rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs.unsqueeze(dim=-2) * c2w[:3, :3],
                       -1)  # shape: [H, W, 3]

    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


r"""Usage:
        python <this_file> --splits <train_val_splits> --datadir <dir_path_to_original_data>
Example: 
        python utils/convert_original_data_to_rays_llff.py --splits train --datadir data/nerf_llff_data/flower
"""

# Input Args
half_res = True  # default setting, corresponding to 400x400 images in the synthetic dataset in NeRF
white_bkgd = True  # default setting for the synthetic dataset in NeRF
split_size = 4096  # manually set
llffhold = 8
##############################################

parser = configargparse.ArgumentParser()
parser.add_argument("--splits", type=str, default='train')
parser.add_argument("--datadir", type=str, default='')
parser.add_argument("--suffix", type=str, default='')
parser.add_argument("--ignore",
                    type=str,
                    default='',
                    help='ignore some samples')
args = parser.parse_args()

# Set up save folders
splits = args.splits.split(',')
datadir = args.datadir
prefix = ''.join(splits)
suffix = sys.argv[3] if len(sys.argv) == 4 else ''
savedir = f'{os.path.normpath(datadir)}_real_{prefix}{args.suffix}'
os.makedirs(savedir, exist_ok=True)

# Load all images

images, poses, *_ = load_llff_data(args.datadir,
                                   factor=8,
                                   recenter=True,
                                   bd_factor=.75,
                                   spherify=False,
                                   path_zflat=False,
                                   n_pose_video=120)
images, poses = images.to('cpu'), poses.to('cpu')
H, W, focal = poses[0, :3, -1]
H, W = int(H), int(W)
poses = poses[:, :3, :4]
if llffhold > 0:
    print('Auto LLFF holdout,', llffhold)
    i_test = np.arange(images.shape[0])[::llffhold]
i_val = i_test
i_train = np.array([
    i for i in np.arange(int(images.shape[0]))
    if (i not in i_test and i not in i_val)
])

all_imgs, all_poses = [], []
if 'train' in splits:
    all_imgs += [images[i_train]]
    all_poses += [poses[i_train]]
if 'val' in splits or 'test' in splits:
    all_imgs += [images[i_val]]
    all_poses += [poses[i_val]]
all_imgs = torch.cat(all_imgs, dim=0)
all_poses = torch.cat(all_poses, dim=0)
print(
    f'Read all images and poses, done. all_imgs shape {all_imgs.shape}, all_poses shape {all_poses.shape}'
)

# Get rays together
all_data = []  # All rays_o, rays_d, rgb will be saved here
for im, po in zip(all_imgs, all_poses):
    rays_o, rays_d = get_rays(H, W, focal, po[:3, :4])  # [H, W, 3]
    data = torch.cat([rays_o, rays_d, im], dim=-1)  # [H, W, 9]
    all_data += [data.view(H * W, 9)]  # [H*W, 9]
all_data = torch.cat(all_data, dim=0)
print(f'Collect all rays, done. all_data shape {all_data.shape}')

# Shuffle rays
rand_ix1 = np.random.permutation(all_data.shape[0])
rand_ix2 = np.random.permutation(all_data.shape[0])
all_data = all_data[rand_ix1][rand_ix2]
all_data = to_array(all_data)

# Save
split = 0
num = all_data.shape[0] // split_size * split_size
for ix in range(0, num, split_size):
    split += 1
    save_path = f'{savedir}/{prefix}_{split}.npy'
    d = all_data[ix:ix + split_size]
    np.save(save_path, d)
    print(f'[{split}/{num//split_size}] save_path: {save_path}')
print(f'All data saved at "{savedir}"')
