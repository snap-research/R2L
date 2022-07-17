import os
import time
import numpy as np
import sys
import configargparse
import torch
import imageio
import json
import cv2


def to_tensor(x):
    return x.to('cpu') if isinstance(
        x, torch.Tensor) else torch.Tensor(x).to('cpu')


def to_array(x):
    return x if isinstance(x, np.ndarray) else x.data.cpu().numpy()


def tile(a, dim, n_tile):
    r"""Copied from DONERF code 'util/helper.py'.
    """
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*repeat_idx)
    order_index = torch.LongTensor(
        np.concatenate([
            init_dim * np.arange(n_tile) + i for i in range(init_dim)
        ])).to(a.device)
    return torch.index_select(a, dim, order_index)


def generate_ray_directions(w, h, fov, focal):
    r"""Copied from DONERF code 'util/raygeneration.py'.
    """
    x_dist = np.tan(fov / 2) * focal
    y_dist = x_dist * (h / w)
    x_dist_pp = x_dist / (w / 2)
    y_dist_pp = y_dist / (h / 2)

    start = np.array(
        [-(x_dist - x_dist_pp / 2), -(y_dist - y_dist_pp / 2), focal])
    ray_d = np.repeat(start[None], repeats=w * h, axis=0).reshape((h, w, -1))
    w_range = np.repeat(np.arange(0, w)[None], repeats=h, axis=0)
    h_range = np.repeat(np.arange(0, h)[None], repeats=w, axis=0).T
    ray_d[:, :, 0] += x_dist_pp * w_range
    ray_d[:, :, 1] += y_dist_pp * h_range

    dirs = ray_d / np.tile(
        np.linalg.norm(ray_d, axis=2)[:, :, None], (1, 1, 3))
    dirs[:, :, 1] *= -1.
    dirs[:, :, 2] *= -1.
    return dirs


def nerf_get_ray_dirs(rotations, directions) -> torch.tensor:
    r"""Copied from DONERF code 'nerf_raymarch_common.py'.
    """
    # multiply them with the camera transformation matrix
    ray_directions = torch.bmm(rotations, torch.transpose(directions, 1, 2))
    ray_directions = torch.transpose(ray_directions, 1, 2).reshape(
        directions.shape[0] * directions.shape[1], -1)

    return ray_directions


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
        python convert_original_data_to_rays_blender.py --splits train --datadir data/bulldozer
        python convert_original_data_to_rays_blender.py --splits train --datadir data/nerf_synthetic/lego
"""

# Input Args
white_bkgd = True  # default setting for the synthetic dataset in NeRF
split_size = 4096  # manually set
##############################################

parser = configargparse.ArgumentParser()
parser.add_argument("--splits", type=str, default='')
parser.add_argument("--datadir", type=str, default='')
parser.add_argument("--suffix", type=str, default='')
parser.add_argument("--ignore",
                    type=str,
                    default='',
                    help='ignore some samples')
parser.add_argument("--donerf", action='store_true')
parser.add_argument("--full_res", action='store_true')
args = parser.parse_args()

# Hand-designed rule
if 'ficus' in args.datadir:
    args.ignore = '10,13,14,24,26,30,31,37,39,40,41,47,48,49,52,54,55,57,58,66,67,74,75,76,77,79,81,82,87,88,89,94,97,99'  # images of phi >= 0

# Set up save folders
splits = args.splits.split(',')
datadir = args.datadir
prefix = ''.join(splits)
savedir = f'{os.path.normpath(datadir)}_real_{prefix}{args.suffix}'
os.makedirs(savedir, exist_ok=True)

# Load all train/val images
all_imgs, all_poses, metas = [], [], {}
ignored_img_indices = args.ignore.split(',')
for s in splits:
    with open(os.path.join(datadir, 'transforms_{}.json'.format(s)),
              'r') as fp:
        metas[s] = json.load(fp)
for s in splits:
    meta = metas[s]
    imgs = []
    poses = []
    for frame in meta['frames']:
        fname = os.path.join(datadir, frame['file_path'] + '.png')
        img_index = frame['file_path'].split('_')[
            -1]  # 'file_path' example: "./train/r_3"
        if img_index not in ignored_img_indices:
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
    imgs = (np.array(imgs) / 255.).astype(
        np.float32)  # keep all 4 channels (RGBA)
    num_channels = imgs[-1].shape[
        2]  # @mst: for donerf data, some of them do not have A channel
    poses = np.array(poses).astype(np.float32)
    all_imgs.append(imgs)
    all_poses.append(poses)
all_imgs = np.concatenate(all_imgs, 0)
all_poses = np.concatenate(all_poses, 0)
print(
    f'Read all images and poses, done. all_imgs shape {all_imgs.shape}, all_poses shape {all_poses.shape}'
)

# Resize if necessary
H, W = all_imgs[0].shape[:2]
# -- @mst: DONERF data includes 'camera_angle_x' in 'dataset_info.json'. Here expand to its case.
if 'camera_angle_x' in meta:
    camera_angle_x = float(meta['camera_angle_x'])
else:  # to train with DONERF data
    with open(os.path.join(datadir, 'dataset_info.json'), 'r') as fp:
        camera_angle_x = float(json.load(fp)['camera_angle_x'])
# --

focal = .5 * W / np.tan(.5 * camera_angle_x)
half_res = not args.full_res
if half_res:
    H = H // 2
    W = W // 2
    focal = focal / 2.
    imgs_half_res = np.zeros((all_imgs.shape[0], H, W, num_channels))
    for i, img in enumerate(all_imgs):
        imgs_half_res[i] = cv2.resize(img, (H, W),
                                      interpolation=cv2.INTER_AREA)
    all_imgs = imgs_half_res
all_imgs = to_tensor(all_imgs)
all_poses = to_tensor(all_poses)
if (
        num_channels == 4
) and white_bkgd:  # on DONERF dataset, classroom, forest, pavillon, they do not have alpha channel
    all_imgs = all_imgs[..., :3] * all_imgs[..., -1:] + (1. -
                                                         all_imgs[..., -1:])
print(
    f'Resize, done. all_imgs shape {all_imgs.shape}, all_poses shape {all_poses.shape}, num_channels of the images {num_channels}'
)

# Get rays together
all_data = []  # All rays_o, rays_d, rgb will be saved here
for im, po in zip(all_imgs, all_poses):
    if args.donerf:
        # -- Refers to DONERF code:
        # https://github.com/MingSun-Tse/DONERF/blob/1986abe52e1fe26d8192d53452c54ff54a463761/src/datasets.py#L258
        # https://github.com/MingSun-Tse/DONERF/blob/1986abe52e1fe26d8192d53452c54ff54a463761/src/features.py#L377
        po = po.unsqueeze(dim=0)  # 'nerf_get_ray_dirs' later needs a batch dim
        rotations = po[:, :3, :3]
        npdirs = generate_ray_directions(W, H, camera_angle_x, focal)
        directions = torch.from_numpy(npdirs.flatten().reshape(
            -1, 3)).unsqueeze(
                dim=0).float()  # 'nerf_get_ray_dirs' later needs a batch dim
        rays_d = nerf_get_ray_dirs(rotations, directions).view(H, W,
                                                               3)  # [H, W, 3]

        # n_images, n_samples = 1, H * W
        # rays_o1 = tile(po, dim=0, n_tile=n_samples).view(H, W, 3)
        rays_o = po[0, :3, -1].expand(rays_d.shape)  # [H, W, 3]
        # --

        # -- For debug, use NeRF's ray generating code, which is different from DONERF's: (rays_d2 - rays_d) is not zero
        # rays_o2, rays_d2 = get_rays(H, W, focal, po[0, :3, :4]) # [H, W, 3]
        # import pdb; pdb.set_trace()
        # --
    else:
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
