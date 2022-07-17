import os, time, numpy as np
import torch
import imageio
import json
import torch.nn.functional as F
import cv2
from torch.utils.data import Dataset
from utils.run_nerf_raybased_helpers import to_tensor, to8b, to_array, visualize_3d

trans_t = lambda t: torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t],
                                  [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor(
    [[1, 0, 0, 0], [0, np.cos(phi), -np.sin(phi), 0],
     [0, np.sin(phi), np.cos(phi), 0], [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor(
    [[np.cos(th), 0, -np.sin(th), 0], [0, 1, 0, 0],
     [np.sin(th), 0, np.cos(th), 0], [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0],
                        [0, 0, 0, 1]]) @ c2w
    return c2w


def load_blender_data(basedir,
                      half_res=False,
                      testskip=1,
                      n_pose=40,
                      perturb=False):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)),
                  'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            # print(frame.keys()) # frame keys: file_path, rotation, transform_matrix
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(
            np.float32)  # keep all 4 channels (RGBA)
        num_channels = imgs[-1].shape[
            2]  # @mst: for donerf data, some of them do not have A channel
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    # -- @mst: DONERF data includes 'camera_angle_x' in 'dataset_info.json'. Here expand to its case.
    if 'camera_angle_x' in meta:
        camera_angle_x = float(meta['camera_angle_x'])
    else:  # to train with DONERF data
        with open(os.path.join(basedir, 'dataset_info.json'), 'r') as fp:
            camera_angle_x = float(json.load(fp)['camera_angle_x'])
    # --
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    thetas = np.linspace(-180, 180, n_pose + 1)[:-1]
    render_poses = to_array(
        torch.stack([pose_spherical(t, -30, 4) for t in thetas], 0))

    # -- @mst: plot origins and dirs for understanding
    render_poses = np.array([to_array(get_rand_pose())
                             for _ in range(200)])  # test rand pose
    cmaps = ['Greens', 'Reds']

    savepath = 'ray_origin_scatters_dataposes_vs_videoposes_blender.pdf'
    xyzs = [(poses[:, 0, 3], poses[:, 1, 3], poses[:, 2, 3]),
            (render_poses[:, 0, 3], render_poses[:, 1, 3], render_poses[:, 2,
                                                                        3])]
    visualize_3d(xyzs=xyzs, savepath=savepath, cmaps=cmaps)

    savepath = 'ray_dir_scatters_dataposes_vs_videoposes_blender.pdf'
    xyzs = [(poses[:, 0, 2], poses[:, 1, 2], poses[:, 2, 2]),
            (render_poses[:, 0, 2], render_poses[:, 1, 2], render_poses[:, 2,
                                                                        2])]
    visualize_3d(xyzs=xyzs, savepath=savepath, cmaps=cmaps)
    # --

    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, num_channels))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (H, W),
                                          interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return to_tensor(imgs), to_tensor(poses), to_tensor(render_poses), [
        H, W, focal
    ], i_split


def setup_blender_datadir(datadir_old, datadir_new):
    '''Use existing data as a softlink. Deprecated now'''
    import shutil
    if os.path.exists(datadir_new):
        if os.path.isfile(datadir_new):
            os.remove(datadir_new)
        else:
            shutil.rmtree(datadir_new)
        os.makedirs(datadir_new)

    # copy json file
    shutil.copy(datadir_old + '/transforms_train.json', datadir_new)

    # create softlink for images
    imgs = [
        x for x in os.listdir(datadir_old + '/train') if x.endswith('.png')
    ]
    os.makedirs(datadir_new + '/train')
    cwd = os.getcwd()
    os.chdir(datadir_new + '/train')
    dirname_old = datadir_old.split('/')[-1]
    for img in imgs:
        # print(os.getcwd())
        # print(f'../../{dirname_old}/train/{img} -> f{img}')
        os.symlink(f'../../{dirname_old}/train/{img}', f'{img}')
    os.chdir(cwd)  # change back working directory


def setup_blender_datadir_v2(datadir_old,
                             datadir_new,
                             half_res=False,
                             white_bkgd=True):
    '''Set up datadir and save data as .npy.
    '''
    import shutil
    if os.path.exists(datadir_new):
        if os.path.isfile(datadir_new):
            os.remove(datadir_new)
        else:
            shutil.rmtree(datadir_new)
    os.makedirs(datadir_new)

    # copy json file
    shutil.copy(f'{datadir_old}/transforms_train.json', datadir_new)

    # save .png images to .npy
    imgs = [
        x for x in os.listdir(f'{datadir_old}/train') if x.endswith('.png')
    ]
    os.makedirs(f'{datadir_new}/train')
    for img in imgs:
        rgb = imageio.imread(f'{datadir_old}/train/{img}')
        rgb = np.array(rgb) / 255.
        if half_res:
            H, W = rgb.shape[:2]
            rgb = cv2.resize(rgb, (H // 2, W // 2),
                             interpolation=cv2.INTER_AREA)
        rgb = rgb[..., :3] * rgb[..., -1:] + (
            1. - rgb[..., -1:]) if white_bkgd else rgb[..., :3]
        np.save(f"{datadir_new}/train/{img.replace('.png', '.npy')}", rgb)


def save_blender_data(datadir, poses, images, split='train'):
    '''Save pseudo data created by a trained nerf.'''
    import json, os
    json_file = '%s/transforms_%s.json' % (datadir, split)
    with open(json_file) as f:
        data = json.load(f)

    frames = data['frames']
    n_img = len(frames)
    folder = os.path.split(json_file)[
        0]  # example: 'data/nerf_synthetic/lego/'
    for pose, img in zip(poses, images):
        n_img += 1
        img_path = './%s/r_%d_pseudo' % (
            split, n_img - 1
        )  # add the 'pseudo' suffix to differentiate it from real-world data

        # add new frame data to json
        new_frame = {k: v for k, v in frames[0].items()}
        new_frame['file_path'] = img_path
        new_frame['transform_matrix'] = pose.data.cpu().numpy().tolist()
        frames += [new_frame]

        # save image
        img_path = '%s/%s.npy' % (folder, img_path)
        # imageio.imwrite(img_path, to8b(img.data.cpu().numpy()))
        np.save(img_path, img.data.cpu().numpy())

    with open(json_file, 'w') as f:
        data['frames'] = frames
        json.dump(data, f, indent=4)


# Use dataloader to load data
def is_img(x):
    _, ext = os.path.splitext(x)
    return ext.lower() in ['.png', '.jpeg', '.jpg', '.bmp', '.npy']


class BlenderDataset(Dataset):
    '''Load data in .npy (they are images stored in .npy).
    '''

    def __init__(self,
                 datadir,
                 pseudo_ratio=0.5,
                 n_original=100,
                 split='train'):
        self.datadir = datadir
        with open(os.path.join(datadir, 'transforms_{}.json'.format(split)),
                  'r') as fp:
            frames = json.load(fp)['frames']
            n_pseudo = int(n_original / (1 - pseudo_ratio) - n_original)
            pseudo_indices = np.random.permutation(
                len(frames) - n_original)[:n_pseudo] + n_original
            self.frames = frames[:n_original]
            for ix in pseudo_indices:
                self.frames.append(frames[ix])

    def __getitem__(self, index):
        index = index % (len(self.frames))
        frame = self.frames[index]
        pose = torch.Tensor(frame['transform_matrix'])
        fname = os.path.join(self.datadir, frame['file_path'] +
                             '.npy')  # 'file_path' includes file extension
        img = torch.Tensor(np.load(fname))
        return img, pose, index

    def __len__(self):
        return len(self.frames)


class BlenderDataset_v2(Dataset):
    r"""Load data of ray origins and directions. This is the most straightforward way.
    """

    def __init__(self,
                 datadir,
                 dim_dir=3,
                 dim_rgb=3,
                 rand_crop_size=-1,
                 img_H=0,
                 img_W=0,
                 hold_ratio=0,
                 pseudo_ratio=1.):
        self.datadir = datadir
        pseudo = [
            f'{datadir}/{x}' for x in os.listdir(datadir)
            if x.endswith('.npy') and not x.startswith('train_')
        ]
        original = [
            f'{datadir}/{x}' for x in os.listdir(datadir)
            if x.endswith('.npy') and x.startswith('train_')
        ]

        # Pick a subset of pseudo data, merge it with original data
        assert 0 <= pseudo_ratio <= 1 or pseudo_ratio == -1
        if pseudo_ratio == -1:  # use all the data
            all_splits = pseudo + original
        else:
            original_ratio = 1. - pseudo_ratio
            num_pseudo = int(len(original) / original_ratio) - len(original)
            pseudo = np.random.choice(pseudo, num_pseudo).tolist()
            all_splits = pseudo + original

        # Hold some data, not used for training (for ablation study)
        assert 0 <= hold_ratio < 1
        if hold_ratio > 0:
            left = int(len(all_splits) * (1 - hold_ratio))
            all_splits = np.random.choice(all_splits, left)

        self.all_splits = all_splits
        self.dim_dir = dim_dir
        self.dim_rgb = dim_rgb
        self.rand_crop_size = rand_crop_size
        self.img_H = img_H
        self.img_W = img_W
        print(
            f'Load data done. #All files: {len(self.all_splits)} #Original: {len(original)} #Pseudo: {len(pseudo)}'
        )

    def _square_rand_bbox(self):
        bbx1 = np.random.randint(0, self.img_W - self.rand_crop_size + 1)
        bby1 = np.random.randint(0, self.img_H - self.rand_crop_size + 1)
        return bbx1, bby1, bbx1 + self.rand_crop_size, bby1 + self.rand_crop_size

    def __getitem__(self, index):
        d = np.load(self.all_splits[index])  # [H, W, 9] or [n_ray, 9]
        d = torch.Tensor(d)  # [H, W, 9] or [n_ray, 9]

        if self.rand_crop_size > 0:
            bbx1, bby1, bbx2, bby2 = self._square_rand_bbox()
            d = d[bby1:bby2, bbx1:bbx2, :]

        return d[..., :3], d[..., 3:3 +
                             self.dim_dir], d[..., 3 + self.dim_dir:3 +
                                              self.dim_dir + self.dim_rgb]

    def __len__(self):
        return len(self.all_splits)


def get_novel_poses(args, n_pose, theta1=-180, theta2=180, phi1=-90, phi2=0):
    '''Even-spaced sampling'''
    near, far = 2, 6
    if isinstance(n_pose, int):
        thetas = np.linspace(theta1, theta2, n_pose + 1)[:-1]
        phis = [-30]
        radiuses = [4]
    else:  # n_pose is a list
        if ':' not in n_pose[0]:
            n_pose = [int(x) for x in n_pose]
            thetas = np.linspace(theta1, theta2, n_pose[0] + 1)[:-1]
            phis = np.linspace(phi1, phi2, n_pose[1] + 2)[1:-1]
            radiuses = np.linspace(near, far, n_pose[2] + 2)[1:-1]
        else:
            mode, value = n_pose[0].split(':')
            thetas = np.linspace(
                theta1, theta2,
                int(value) + 1)[:-1] if mode == 'sample' else [float(value)]
            mode, value = n_pose[1].split(':')
            phis = np.linspace(
                phi1, phi2,
                int(value) + 2)[1:-1] if mode == 'sample' else [float(value)]
            mode, value = n_pose[2].split(':')
            radiuses = np.linspace(
                near, far,
                int(value) + 2)[1:-1] if mode == 'sample' else [float(value)]
    novel_poses = torch.stack([
        pose_spherical(t, p, r) for r in radiuses for p in phis for t in thetas
    ], 0)
    return to_tensor(novel_poses)


def get_rand_pose():
    '''Random sampling. Random origins and directions.
    '''
    theta1 = -180
    theta2 = 180
    phi1 = -90
    phi2 = 0
    theta = theta1 + np.random.rand() * (theta2 - theta1)
    phi = phi1 + np.random.rand() * (phi2 - phi1)
    return to_tensor(pose_spherical(theta, phi, 4))
