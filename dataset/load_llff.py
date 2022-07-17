import numpy as np
import os, imageio
from torch.utils.data import Dataset
from utils.run_nerf_raybased_helpers import to_tensor, to_array, to8b, visualize_3d

########## Slightly modified version of LLFF data loading code
##########  see https://github.com/Fyusion/LLFF for original


def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from shutil import copy
    from subprocess import check_output

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [
        f for f in imgs
        if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])
    ]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100. / r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join([
            'mogrify', '-resize', resizearg, '-format', 'png',
            '*.{}'.format(ext)
        ])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')


def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):

    poses_arr = np.load(os.path.join(basedir,
                                     'poses_bounds.npy'))  # shape [20, 17]
    poses = poses_arr[:, :-2].reshape([-1, 3,
                                       5]).transpose([1, 2,
                                                      0])  # shape [3, 5, 20]
    bds = poses_arr[:, -2:].transpose([1, 0])  # roughly, bounds = 1.7~7.6

    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape

    sfx = ''

    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1

    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print(imgdir, 'does not exist, returning')
        return

    imgfiles = [
        os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))
        if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')
    ]
    if poses.shape[-1] != len(imgfiles):
        print('Mismatch between imgs {} and poses {} !!!!'.format(
            len(imgfiles), poses.shape[-1]))
        return

    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor

    if not load_imgs:
        return poses, bds

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    imgs = imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
    imgs = np.stack(imgs, -1)

    print('Loaded image data', imgs.shape,
          poses[:, -1, 0])  # imgs.shape (378, 504, 3, 20)
    return poses, bds, imgs


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)  # @mst:
    # vec1_avg = up
    # vec0 = normalize(np.cross(vec1_avg, vec2))
    vec0 = normalize(np.cross(
        up, vec2))  # @mst: replace above with this neat line; shape (3,)
    vec1 = normalize(np.cross(vec2, vec0))  # @mst: shape (3,)
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses):
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    return c2w


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    '''@mst: zrate = 0.5, rots = 2'''
    render_poses = []
    rads = np.array(list(rads) + [1.])  #
    hwf = c2w[:, 4:5]
    for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array(
                [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) *
            rads)
        # @mst: above is equivalent to matrix mul: [3, 4] @ [4, 1]
        z = normalize(c - np.dot(c2w[:3, :4], np.array([
            0, 0, -focal, 1.
        ])))  # @mst: why use extra focal instead of the focal in poses?
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))

    # @mst: try our random poses
    # render_poses = [get_rand_pose_v2().data.cpu().numpy() for _ in range(N)]
    return render_poses


# @mst
def get_rand_pose_v2():
    # -- pass local variables
    c2w, up, focal, poses = GLOBALS['c2w'], GLOBALS['up'], GLOBALS[
        'focal'], GLOBALS['poses']
    rads = np.array(list(np.max(np.abs(poses[:, :3, 3]), axis=0)) + [1])
    hwf = c2w[:, 4:5]
    # --
    mins_o, maxs_o = get_bbox(poses[:, :3, 3])  # origins
    mins_d, maxs_d = get_bbox(poses[:, :3, 2])  # directions

    # scheme 1
    # c = np.dot(c2w[:3,:4], np.array([np.random.rand() * 2 - 1, np.random.rand() * 2 - 1, np.random.rand() * 2 - 1, 1.]) * rads)
    # z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-0.85*focal, 1.]))) # this 0.85 is manually tuned to cover all training poses

    # scheme 2: this is more random and targeted
    c = np.dot(
        c2w[:3, :4],
        np.array([
            rand_uniform(mins_o[0], maxs_o[0], scale=1.1),
            rand_uniform(mins_o[1], maxs_o[1], scale=1.1),
            rand_uniform(mins_o[2], maxs_o[2], scale=1.1), 1
        ]))
    z = np.dot(
        c2w[:3, :4],
        np.array([
            rand_uniform(mins_d[0], maxs_d[0], scale=1.1),
            rand_uniform(mins_d[1], maxs_d[1], scale=1.1),
            rand_uniform(mins_d[2], maxs_d[2], scale=1.1), 1
        ]))
    z = normalize(z)
    pose = np.concatenate([viewmatrix(z, up, c), hwf], 1)
    return to_tensor(pose)


# @mst
def get_bbox(array):
    '''get the bounding box of a bunch of points in 3d space'''
    array = np.array(array)
    assert len(
        array.shape) == 2 and array.shape[1] == 3  # shape should be [N, 3]
    mins, maxs = np.min(array, axis=0), np.max(array, axis=0)
    return mins, maxs


def rand_uniform(left, right, scale=1):
    assert right > left
    middle = (left + right) * 0.5
    left = middle - (right - left) * scale * 0.5
    right = 2 * middle - left
    return np.random.rand() * (right - left) + left


def recenter_poses(poses):
    # @mst: poses shape: [n_img, 3, 5]
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])

    # get the average c2w
    c2w = poses_avg(poses)  # @mst: shape [3, 5]
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)  # @mst: make it to 4x4

    # @mst: update real-world poses
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]),
                     [poses.shape[0], 1, 1])  # @mst: [n_img, 1, 4]
    poses = np.concatenate([poses[:, :3, :4], bottom],
                           -2)  # @mst: [n_img, 4, 4]

    # @mst: tranform using average pose
    poses = np.linalg.inv(c2w) @ poses

    poses_[:, :3, :
           4] = poses[:, :3, :
                      4]  # @mst: do not change hwf, update the c2w matrix
    poses = poses_
    return poses


#####################
def spherify_poses(poses, bds):

    p34_to_44 = lambda p: np.concatenate([
        p,
        np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])
    ], 1)

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv(
            (np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1, .2, .3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(
        poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1. / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2 - zh**2)
    new_poses = []

    for th in np.linspace(0., 2. * np.pi, 120):

        camorigin = np.array(
            [radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate([
        new_poses,
        np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)
    ], -1)
    poses_reset = np.concatenate([
        poses_reset[:, :3, :4],
        np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)
    ], -1)

    return poses_reset, new_poses, bds


def load_llff_data(basedir,
                   factor=8,
                   recenter=True,
                   bd_factor=.75,
                   spherify=False,
                   path_zflat=False,
                   n_pose_video=120):
    poses, bds, imgs = _load_data(
        basedir, factor=factor)  # factor=8 downsamples original imgs by 8x
    print('Loaded', basedir, bds.min(),
          bds.max())  # @mst, room: 10.706691140704915 91.66782174279389

    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate(
        [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1. / (bds.min() * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc
    # -- @mst: For scene 'room'
    # import pdb; pdb.set_trace()
    # rescale to make the min of bds to 1/0.75, far of bds is around 5+~11+.
    # poses: [41, 3, 5], bds: [41, 2]
    # --

    if recenter:
        poses = recenter_poses(poses)

    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:

        c2w = poses_avg(poses)  # @mst: poses: [20, 3, 5]
        print('recentered', c2w.shape)
        print(c2w[:3, :4])

        ## Get spiral
        # Get average pose
        # up = normalize(poses[:, :3, 1].sum(0)) # @mst: move to below, more locally used

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
        dt = .75
        mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(
            np.abs(tt), 90,
            0)  # @mst: sort postions in ascending order, pick the 90%
        c2w_path = c2w
        N_views = n_pose_video
        N_rots = 2
        if path_zflat:
            #             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
            rads[2] = 0.
            N_rots = 1
            N_views /= 2

        # Generate poses for spiral path
        up = normalize(poses[:, :3, 1].sum(0))
        # -- @mst: set globals for later use. I know it's ugly...
        global GLOBALS
        GLOBALS = {}
        GLOBALS['c2w'] = c2w
        GLOBALS['up'] = up
        GLOBALS['rads'] = rads
        GLOBALS['focal'] = focal
        GLOBALS['poses'] = poses  # @mst: will be used later
        # --
        render_poses = render_path_spiral(c2w_path,
                                          up,
                                          rads,
                                          focal,
                                          zdelta,
                                          zrate=.5,
                                          rots=N_rots,
                                          N=N_views)

    render_poses = np.array(render_poses).astype(np.float32)

    # -- @mst: plot origins and dirs for understanding
    cmaps = ['Greens', 'Reds']
    xyzs = [(poses[:, 0, 3], poses[:, 1, 3], poses[:, 2, 3]),
            (render_poses[:, 0, 3], render_poses[:, 1, 3], render_poses[:, 2,
                                                                        3])]
    savepath = f'ray_origin_scatters_dataposes_vs_videoposes_llff.pdf'
    visualize_3d(xyzs, savepath=savepath, cmaps=cmaps)

    xyzs = [(poses[:, 0, 2], poses[:, 1, 2], poses[:, 2, 2]),
            (1.2 * render_poses[:, 0, 2], 1.2 * render_poses[:, 1, 2],
             1.2 * render_poses[:, 2, 2])]
    savepath = f'ray_dir_scatters_dataposes_vs_videoposes_llff.pdf'
    visualize_3d(xyzs, savepath=savepath, cmaps=cmaps)
    # --

    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape,
          bds.shape)  # (20, 3, 5) (20, 378, 504, 3) (20, 2)

    dists = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)

    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return to_tensor(images), to_tensor(poses), to_tensor(bds), to_tensor(
        render_poses), i_test
