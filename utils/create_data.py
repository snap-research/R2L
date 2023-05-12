import os, sys, copy

sys.path.insert(0, './')
import numpy as np
import imageio, shutil
import time
import torch
import torch.nn.functional as F

from model.nerf_raybased import NeRF
from utils.run_nerf_raybased_helpers import to_tensor, to_array, mse2psnr, to8b, img2mse, load_weights, sample_pdf, ndc_rays, get_rays, get_embedder

from dataset.load_llff import load_llff_data
from dataset.load_deepvoxels import load_dv_data
from dataset.load_blender import load_blender_data, setup_blender_datadir_v2 as setup_blender_datadir, save_blender_data, get_novel_poses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

# Set up logging directories -------
from smilelogging import Logger
from smilelogging.utils import Timer
from option import args

logger = Logger(args)
accprint = logger.log_printer.accprint
netprint = logger.log_printer.netprint
ExpID = logger.ExpID
# ---------------------------------

# redefine get_rays
from functools import partial

get_rays1 = get_rays
get_rays = partial(get_rays,
                   trans_origin=args.trans_origin,
                   focal_scale=args.focal_scale)


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([
            fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)
        ], 0)

    return ret


def run_network(inputs,
                viewdirs,
                fn,
                embed_fn,
                embeddirs_fn,
                netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(
        inputs, [-1, inputs.shape[-1]]
    )  # @mst: shape: torch.Size([65536, 3]), 65536=1024*64 (n_rays * n_sample_per_ray)
    embedded = embed_fn(inputs_flat)  # shape: [n_rays*n_sample_per_ray, 63]

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat,
                            list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(
            rays_flat[i:i + chunk],
            **kwargs)  # @mst: train, rays_flat.shape(0) = 1024, chunk = 32768
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H,
           W,
           focal,
           chunk=1024 * 32,
           rays=None,
           c2w=None,
           ndc=True,
           near=0.,
           far=1.,
           use_viewdirs=False,
           c2w_staticcam=None,
           **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(
            viewdirs, dim=-1, keepdim=True
        )  # @mst: 'rays_d' is real-world data, needs normalization.
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(
        rays_o,
        [-1, 3]).float()  # @mst: test: [160000, 3], 400*400; train: [1024, 3]
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(
        rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)

    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses,
                hwf,
                chunk,
                render_kwargs,
                gt_imgs=None,
                savedir=None,
                render_factor=0,
                new_render_func=False):
    H, W, focal = hwf
    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs, disps = [], []
    for i, c2w in enumerate(render_poses):
        if new_render_func:  # our new rendering func
            model = render_kwargs['network_fn']
            perturb = render_kwargs['perturb']
            rays_o, rays_d = get_rays(H, W, focal,
                                      c2w[:3, :4])  # rays_o shape: # [H, W, 3]
            rays_o, rays_d = rays_o.view(-1, 3), rays_d.view(-1, 3)

            # batchify
            rgb, disp = [], []
            for ix in range(0, rays_o.shape[0], chunk):
                rgb_, disp_, *_ = model(rays_o[ix:ix + chunk],
                                        rays_d[ix:ix + chunk],
                                        perturb=perturb)
                rgb += [rgb_]
                disp += [disp_]
            rgb, disp = torch.cat(rgb, dim=0), torch.cat(disp, dim=0)
            rgb, disp = rgb.view(H, W, -1), disp.view(H, W, -1)

        else:  # original implementation
            rgb, disp, acc, _ = render(H,
                                       W,
                                       focal,
                                       chunk=chunk,
                                       c2w=c2w[:3, :4],
                                       **render_kwargs)

        rgbs.append(rgb)
        disps.append(disp)

        if savedir is not None:
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, to8b(rgbs[-1]))

    rgbs = torch.stack(rgbs, dim=0)
    disps = torch.stack(disps, dim=0)

    if gt_imgs is not None:
        test_loss = img2mse(rgbs, gt_imgs)
        test_psnr = mse2psnr(test_loss)
    else:
        test_loss, test_psnr = None, None

    return rgbs, disps, test_loss, test_psnr


def create_nerf(args, near, far):
    """Instantiate NeRF's MLP model.
    """
    # set up model
    model, model_fine, network_query_fn, grad_vars, optimizer = [
        None
    ] * 5  # we do not need model here; we use teacher, which is set up below

    # in KD, there is a pretrained teacher
    if args.teacher_ckpt:
        teacher_fn = NeRF(D=8,
                          W=256,
                          input_ch=63,
                          output_ch=4,
                          skips=[4],
                          input_ch_views=27,
                          use_viewdirs=args.use_viewdirs).to(device)
        teacher_fine = NeRF(D=8,
                            W=256,
                            input_ch=63,
                            output_ch=4,
                            skips=[4],
                            input_ch_views=27,
                            use_viewdirs=args.use_viewdirs).to(
                                device)  # TODO: not use fixed arguments

        # set to eval
        teacher_fn.eval()
        teacher_fine.eval()
        for param in teacher_fn.parameters():
            param.requires_grad = False
        for param in teacher_fine.parameters():
            param.requires_grad = False

        # load weights
        ckpt_path, ckpt = load_weights(teacher_fn, args.teacher_ckpt,
                                       'network_fn_state_dict')
        ckpt_path, ckpt = load_weights(teacher_fine, args.teacher_ckpt,
                                       'network_fine_state_dict')
        print(f'Load teacher ckpt successfully: "{ckpt_path}"')

        # get network_query_fn
        embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
        if args.use_viewdirs:
            embeddirs_fn, input_ch_views = get_embedder(
                args.multires_views, args.i_embed)
        network_query_fn = lambda inputs, viewdirs, network_fn: run_network(
            inputs,
            viewdirs,
            network_fn,
            embed_fn=embed_fn,
            embeddirs_fn=embeddirs_fn,
            netchunk=args.netchunk)
    # start iteration
    start = 0

    # use DataParallel
    teacher_fn = torch.nn.DataParallel(teacher_fn)
    teacher_fine = torch.nn.DataParallel(teacher_fine)

    # set up training args
    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    # set up testing args
    render_kwargs_test = {
        k: render_kwargs_train[k]
        for k in render_kwargs_train
    }
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    if args.teacher_ckpt:
        render_kwargs_train['teacher_fn'] = teacher_fn
        render_kwargs_train['teacher_fine'] = teacher_fine

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw,
                z_vals,
                rays_d,
                raw_noise_std=0,
                white_bkgd=False,
                pytest=False,
                verbose=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(
        -act_fn(raw) * dists)  # @mst: opacity

    dists = z_vals[..., 1:] - z_vals[..., :-1]  # dists for 'distances'
    dists = torch.cat(
        [dists, to_tensor([1e10]).expand(dists[..., :1].shape)],
        -1)  # [N_rays, N_samples]
    # @mst: 1e10 for infinite distance

    dists = dists * torch.norm(
        rays_d[..., None, :],
        dim=-1)  # @mst: direction vector needs normalization. why this * ?

    rgb = torch.sigmoid(
        raw[..., :3])  # [N_rays, N_samples, 3], RGB for each sampled point
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape).to(device) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = to_tensor(noise)

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]

    # print to check alpha
    if verbose and global_step % args.i_print == 0:
        for i_ray in range(0, alpha.shape[0], 100):
            logtmp = ['%.4f' % x for x in alpha[i_ray]]
            netprint('%4d: ' % i_ray + ' '.join(logtmp))

    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(
        torch.cat(
            [torch.ones((alpha.shape[0], 1)).to(device), 1. - alpha + 1e-10],
            -1), -1)[:, :-1]  # @mst: [N_rays, N_samples]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map).to(device),
                              depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[
        0]  # N_rays = 32768 (1024*32) for test, 1024 for train
    # @mst: ray_batch.shape, train: [1024, 11]
    rays_o, rays_d = ray_batch[:, 0:
                               3], ray_batch[:, 3:
                                             6]  # [N_rays, 3] each, o for 'origin', d for 'direction'
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # @mst: near=2, far=6, in batch

    t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
    if not lindisp:
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    z_vals = z_vals.expand(
        [N_rays, N_samples]
    )  # @mst: shape: torch.Size([1024, 64]) for train, torch.Size([32768, 64]) for test

    # @mst: perturbation of depth z, with each depth value at the middle point
    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape).to(device)  # uniform dist [0, 1)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = to_tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[
        ..., :, None]  # [N_rays, N_samples, 3]
    # when training: [1024, 1, 3] + [1024, 1, 3] * [1024, 64, 1]
    # rays_d range: [-1, 1]

    #     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)

    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw,
        z_vals,
        rays_d,
        raw_noise_std,
        white_bkgd,
        pytest=pytest,
        verbose=verbose)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid.cpu(),
                               weights[..., 1:-1].cpu(),
                               N_importance,
                               det=(perturb == 0.),
                               pytest=pytest)
        z_samples = z_samples.detach().to(device)

        z_vals, _ = torch.sort(
            torch.cat([z_vals, z_samples], -1),
            -1)  # @mst: sort to merge the fine samples with the coarse samples
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[
            ..., :, None]  # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        #         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {
        'rgb_map': rgb_map,
        'disp_map': disp_map,
        'acc_map': acc_map,
        'depth_map': depth_map
    }
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def get_teacher_target(poses, H, W, focal, render_kwargs_train, args,
                       n_pseudo_img):
    render_kwargs_ = {x: v for x, v in render_kwargs_train.items()}
    render_kwargs_['network_fn'] = render_kwargs_train[
        'teacher_fn']  # temporarily change the network_fn
    render_kwargs_['network_fine'] = render_kwargs_train[
        'teacher_fine']  # temporarily change the network_fine
    render_kwargs_.pop('teacher_fn')
    render_kwargs_.pop('teacher_fine')
    teacher_target = []
    t_ = time.time()
    for ix, pose in enumerate(poses):
        print(
            f'[{ix}/{len(poses)}] Using teacher to render more images... elapsed time: {(time.time() - t_):.2f}s'
        )
        rays_o, rays_d = get_rays(H, W, focal, pose)
        batch_rays = torch.stack([rays_o, rays_d], 0)
        rgb, *_ = render(H,
                         W,
                         focal,
                         chunk=args.chunk,
                         rays=batch_rays,
                         verbose=False,
                         retraw=False,
                         **render_kwargs_)
        teacher_target.append(rgb)
        n_pseudo_img[0] += 1

        # check pseudo images
        if n_pseudo_img[0] <= 5:
            filename = f'{args.datadir_kd_new}/pseudo_sample_{n_pseudo_img[0]}.png'
            imageio.imwrite(filename, to8b(rgb))
    print(
        f'Teacher rendering done ({len(poses)} views). Time: {(time.time() - t_):.2f}s'
    )
    return teacher_target


def get_teacher_target_for_rays(rays, render_kwargs_train):
    '''Directly get outputs for rays'''
    render_kwargs_ = {x: v for x, v in render_kwargs_train.items()}
    render_kwargs_['network_fn'] = render_kwargs_train[
        'teacher_fn']  # temporarily change the network_fn
    render_kwargs_['network_fine'] = render_kwargs_train[
        'teacher_fine']  # temporarily change the network_fine
    render_kwargs_.pop('teacher_fn')
    render_kwargs_.pop('teacher_fine')
    H, W, focal = [0] * 3  # placeholder
    rgbs, *_ = render(H,
                      W,
                      focal,
                      rays=rays,
                      verbose=False,
                      retraw=False,
                      **render_kwargs_)
    print(f'rgbs.shape: {rgbs.shape}')
    return rgbs


def train():
    # Load data
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(
            args.datadir,
            args.factor,
            recenter=True,
            bd_factor=.75,
            spherify=args.spherify)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape, render_poses.shape, hwf,
              args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([
            i for i in np.arange(int(images.shape[0]))
            if (i not in i_test and i not in i_val)
        ])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

        from load_llff import get_rand_pose_v2 as get_rand_pose

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(
            args.datadir,
            args.half_res,
            args.testskip,
            n_pose=args.n_pose_video)
        print('Loaded blender', images.shape, poses.shape, render_poses.shape,
              hwf, args.datadir)
        # Loaded blender (138, 400, 400, 4) (138, 4, 4) torch.Size([40, 4, 4]) [400, 400, 555.5555155968841] ./data/nerf_synthetic/lego
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. -
                                                           images[..., -1:])
        else:
            images = images[..., :3]

        from dataset.load_blender import get_rand_pose

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(
            scene=args.shape, basedir=args.datadir, testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf,
              args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R - 1.
        far = hemi_R + 1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    train_images, train_poses = images[i_train], poses[i_train]
    test_poses, test_images = poses[i_test], images[i_test]

    # data sketch
    print(
        f'{len(i_train)} original train views are [{" ".join([str(x) for x in i_train])}]'
    )
    print(
        f'{len(i_test)} test views are [{" ".join([str(x) for x in i_test])}]')
    print(f'{len(i_val)} val views are [{" ".join([str(x) for x in i_val])}]')
    print(
        f'train_images shape {train_images.shape} train_poses shape {train_images.shape}'
    )

    # Create log dir and copy the config file
    f = f'{logger.log_path}/args.txt'
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = f'{logger.log_path}/config.txt'
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, *_ = create_nerf(args, near, far)

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    if args.test_teacher:
        assert args.teacher_ckpt
        print('Testing teacher...')
        render_kwargs_ = {x: v for x, v in render_kwargs_test.items()}
        render_kwargs_['network_fn'] = render_kwargs_train[
            'teacher_fn']  # temporarily change the network_fn
        render_kwargs_['network_fine'] = render_kwargs_train[
            'teacher_fine']  # temporarily change the network_fine
        with torch.no_grad():
            *_, test_loss, test_psnr = render_path(
                test_poses,
                hwf,
                4096,
                render_kwargs_,
                gt_imgs=test_images,
                render_factor=args.render_factor,
                new_render_func=False)
        print(
            f'Teacher test: Loss {test_loss.item():.4f} PSNR {test_psnr.item():.4f}'
        )

    # --- generate new data using trained NeRF
    datadir_kd_old, datadir_kd_new = args.datadir_kd.split(':')

    # render pseduo data
    chunk = args.create_data_chunk
    if args.create_data in ['spiral_evenly_spaced']:
        # set up data directory
        setup_blender_datadir(datadir_kd_old, datadir_kd_new, args.half_res,
                              args.white_bkgd)
        print('Set up new data directory, done!')

        # get poses of psuedo data
        kd_poses = get_novel_poses(args, n_pose=args.n_pose_kd).to(device)
        n_new_pose = len(kd_poses)
        rand_ix = np.random.permutation(n_new_pose)
        kd_poses = kd_poses[rand_ix]  # shuffle
        print(f'Get new poses, done! Total num of new poses: {n_new_pose}')

        n_img = len(train_images)
        n_pseudo_img = [0]
        args.datadir_kd_new = datadir_kd_new  # will be used later
        for i in range(0, n_new_pose, chunk):
            poses = kd_poses[i:i + chunk]
            n_img += len(poses)
            teacher_target = get_teacher_target(poses, H, W, focal,
                                                render_kwargs_train, args,
                                                n_pseudo_img)
            if args.dataset_type == 'blender':
                save_blender_data(datadir_kd_new, poses, teacher_target)
            print(
                f'Create new data. Save to "{datadir_kd_new}". Now total #train samples: {n_img}'
            )

    elif args.create_data in ['rand']:
        split = 0  # index for the generated .npy files
        # set up data directory
        if os.path.exists(datadir_kd_new):
            if args.rm_existing_data:
                rm_func = os.remove if os.path.isfile(
                    datadir_kd_new) else shutil.rmtree
                rm_func(datadir_kd_new)
                os.makedirs(datadir_kd_new)
                print(
                    f'Remove existing data dir. Set up new data directory, done!'
                )
            else:
                npys = [
                    x for x in os.listdir(datadir_kd_new) if x.endswith('.npy')
                ]
                split = len(npys)
                print(
                    f'Found existing data dir. Keep it. #Existing npy files: {split}'
                )
        else:
            os.makedirs(datadir_kd_new)
            print(f'Set up new data directory, done!')

        # set up model
        render_kwargs_ = {x: v for x, v in render_kwargs_train.items()}
        render_kwargs_['network_fn'] = render_kwargs_train[
            'teacher_fn']  # temporarily change the network_fn
        render_kwargs_['network_fine'] = render_kwargs_train[
            'teacher_fine']  # temporarily change the network_fine
        render_kwargs_.pop('teacher_fn')
        render_kwargs_.pop('teacher_fine')

        # run
        i_save, split_size = 100, 4096  # every 4096 rays will make up a .npy file
        data, t0 = [], time.time()
        timer = Timer(args.n_pose_kd)
        for i in range(1, args.n_pose_kd + 1):
            pose = get_rand_pose()
            focal_ = focal * (
                np.random.rand() +
                1) if args.use_rand_focal else focal  # scale focal by [1, 2)
            rays_o, rays_d = get_rays1(
                H, W, focal_, pose[:3, :4])  # rays_o, rays_d shape: [H, W, 3]
            # @mst: note, here it MUST be 'pose[:3,:4]', using 'pose' will cause white rgb output.

            batch_rays = torch.stack([rays_o, rays_d], dim=0)
            rgb, disp, *_, ret_dict = render(H,
                                       W,
                                       focal,
                                       chunk=args.chunk,
                                       rays=batch_rays,
                                       verbose=False,
                                       retraw=False,
                                       **render_kwargs_)
            depth = ret_dict['depth_map']  # [H, W]
            depth = depth[..., None]  # [H, W, 1]
            disp = disp[..., None]  # [H, W, 1]
            if args.does_terminate:
                disp[disp > 0] = 1
            disp = torch.nan_to_num(disp, nan=0)
            mask = (rgb[:, :, 0] > 0.9) & (rgb[:, :, 1] > 0.9) & (rgb[:, :, 2] > 0.9)

            # replace values in tensor1 with new value (e.g. 0) where mask is True
            disp[mask.unsqueeze(2).repeat(1, 1, 1)] = 0
            if args.learn_depth in ['surface']:
                depth = rays_o + rays_d * depth.expand_as(rays_d)  # [H, W, 3]
            if args.learn_depth:
                data_ = torch.cat([rays_o, rays_d, disp],
                                  dim=-1)  # [H, W, 7]
            else:
                data_ = torch.cat([rays_o, rays_d, rgb], dim=-1)  # [H, W, 9]
            data += [data_.view(rays_o.shape[0] * rays_o.shape[1], -1)]
            print(
                f'[{i}/{args.n_pose_kd}] Using teacher to render more images... elapsed time: {(time.time() - t0):.2f}s'
            )
            print(f'Predicted finish time: {timer()}')

            # check pseudo images
            if i <= 5:
                filename = f'{datadir_kd_new}/pseudo_sample_{i}.png'
                if args.save_depth:
                    imageio.imwrite(filename, to8b(disp))
                    print(disp.abs().mean())
                else:
                    imageio.imwrite(filename, to8b(rgb))
                    print(rgb.abs().mean())

            # save to avoid out of memory
            if i % i_save == 0:
                data = torch.cat(data, dim=0)

                # shuffle rays
                rand_ix1 = np.random.permutation(data.shape[0])
                rand_ix2 = np.random.permutation(data.shape[0])
                data = data[rand_ix1][rand_ix2]
                data = to_array(data)

                # save
                num = data.shape[0] // split_size * split_size
                for ix in range(0, num, split_size):
                    split += 1
                    save_path = f'{datadir_kd_new}/data_{split}.npy'
                    d = data[ix:ix + split_size]
                    np.save(save_path, d)
                print(
                    f'[{i}/{args.n_pose_kd}] Saved data at "{datadir_kd_new}"')
                data = []  # reset

    elif args.create_data in ['rand_tworays']:  # for nerf_v4
        # set up data directory
        if os.path.exists(datadir_kd_new):
            if os.path.isfile(datadir_kd_new):
                os.remove(datadir_kd_new)
            else:
                shutil.rmtree(datadir_kd_new)
        os.makedirs(datadir_kd_new)
        print('Set up new data directory, done!')

        # set up model
        render_kwargs_ = {x: v for x, v in render_kwargs_train.items()}
        render_kwargs_['network_fn'] = render_kwargs_train[
            'teacher_fn']  # temporarily change the network_fn
        render_kwargs_['network_fine'] = render_kwargs_train[
            'teacher_fine']  # temporarily change the network_fine
        render_kwargs_.pop('teacher_fn')
        render_kwargs_.pop('teacher_fine')

        # run
        i_save, split_size = 100, 4096  # every 4096 rays will make up a .npy file
        data, t0, split = [], time.time(), 0
        timer = Timer(args.n_pose_kd)
        for i in range(1, args.n_pose_kd + 1):
            pose = get_rand_pose()
            focal_ = focal * (np.random.rand() + 1)  # scale focal by [1, 2)
            rays_o, rays_d = get_rays1(H, W, focal_,
                                       pose)  # rays_o, rays_d shape: [H, W, 3]
            batch_rays = torch.stack([rays_o, rays_d], dim=0)  # [2, H, W, 3]
            rgb, *_ = render(
                H,
                W,
                focal,
                chunk=args.chunk,
                rays=
                batch_rays,  # when batch_rays are given, it will not create rays inside 'render'
                verbose=False,
                retraw=False,
                **render_kwargs_)

            # for each pixel, get its neighbor pixel, add it to the data
            offset = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1],
                      [1, 0], [1, 1]]
            rays_d2 = copy.deepcopy(rays_d)
            rgb2 = copy.deepcopy(rgb)
            H, W = rays_o.shape[0], rays_o.shape[1]
            for h in range(H):
                for w in range(W):
                    while True:
                        offset_h, offset_w = offset[np.random.permutation(
                            len(offset))[0]]
                        neighbor_h = h + offset_h
                        neighbor_w = w + offset_w
                        if 0 <= neighbor_h < H and 0 <= neighbor_w < W:
                            break
                    rays_d2[h, w] = rays_d[neighbor_h, neighbor_w]
                    rgb2[h, w] = rgb[neighbor_h, neighbor_w]

            data_ = torch.cat([rays_o, rays_d, rays_d2, rgb, rgb2],
                              dim=-1)  # [H, W, 15]
            data_ = data_.view(-1, 15)  # [H*W, 15]

            data += [data_]
            print(
                f'[{i}/{args.n_pose_kd}] Using teacher to render more images... elapsed time: {(time.time() - t0):.2f}s'
            )
            print(f'Predicted finish time: {timer()}')

            # check pseudo images
            if i <= 5:
                filename = f'{datadir_kd_new}/pseudo_sample_{i}.png'
                imageio.imwrite(filename, to8b(rgb))

            # save to avoid out of memory
            if i % i_save == 0:
                data = torch.cat(data, dim=0)  # [i_save*H*W, 15]

                # shuffle rays
                rand_ix1 = np.random.permutation(data.shape[0])
                rand_ix2 = np.random.permutation(data.shape[0])
                data = data[rand_ix1][rand_ix2]
                data = to_array(data)

                # save
                num = data.shape[0] // split_size * split_size
                for ix in range(0, num, split_size):
                    split += 1
                    save_path = f'{datadir_kd_new}/data_{split}.npy'
                    d = data[ix:ix + split_size]
                    np.save(save_path, d)
                print(
                    f'[{i}/{args.n_pose_kd}] Saved data at "{datadir_kd_new}"')
                data = []  # reset

    elif args.create_data in ['rand_images']:  # for nerf_v6
        # set up data directory
        if os.path.exists(datadir_kd_new):
            if os.path.isfile(datadir_kd_new):
                os.remove(datadir_kd_new)
            else:
                shutil.rmtree(datadir_kd_new)
        os.makedirs(datadir_kd_new)
        print('Set up new data directory, done!')

        # set up model
        render_kwargs_ = {x: v for x, v in render_kwargs_train.items()}
        render_kwargs_['network_fn'] = render_kwargs_train[
            'teacher_fn']  # temporarily change the network_fn
        render_kwargs_['network_fine'] = render_kwargs_train[
            'teacher_fine']  # temporarily change the network_fine
        render_kwargs_.pop('teacher_fn')
        render_kwargs_.pop('teacher_fine')

        # run
        t0 = time.time()
        timer = Timer(args.n_pose_kd // 10)
        for i in range(1, args.n_pose_kd + 1):
            pose = get_rand_pose()
            focal_ = focal * (np.random.rand() + 1)  # scale focal by [1, 2)
            rays_o, rays_d = get_rays1(H, W, focal_,
                                       pose)  # rays_o, rays_d shape: [H, W, 3]
            batch_rays = torch.stack([rays_o, rays_d], 0)
            rgb, *_ = render(H,
                             W,
                             focal,
                             chunk=args.chunk,
                             rays=batch_rays,
                             verbose=False,
                             retraw=False,
                             **render_kwargs_)
            data_ = torch.cat([rays_o, rays_d, rgb], dim=-1)  # [H, W, 9]
            print(
                f'[{i}/{args.n_pose_kd}] Using teacher to render more images... elapsed time: {(time.time() - t0):.2f}s'
            )

            # check pseudo images
            if i <= 5:
                filename = f'{datadir_kd_new}/pseudo_sample_{i}.png'
                imageio.imwrite(filename, to8b(rgb))

            # save
            save_path = f'{datadir_kd_new}/{i}.npy'
            np.save(save_path, data_.data.cpu().numpy())
            if i % 10 == 0:
                print(f'Predicted finish time: {timer()}')

    elif args.create_data in ['3x3rays']:  # for nerf_v3.4
        # set up data directory
        if os.path.exists(datadir_kd_new):
            if os.path.isfile(datadir_kd_new):
                os.remove(datadir_kd_new)
            else:
                shutil.rmtree(datadir_kd_new)
        os.makedirs(datadir_kd_new)
        print('Set up new data directory, done!')

        # set up model
        render_kwargs_ = {x: v for x, v in render_kwargs_train.items()}
        render_kwargs_['network_fn'] = render_kwargs_train[
            'teacher_fn']  # temporarily change the network_fn
        render_kwargs_['network_fine'] = render_kwargs_train[
            'teacher_fine']  # temporarily change the network_fine
        render_kwargs_.pop('teacher_fn')
        render_kwargs_.pop('teacher_fine')

        # run
        i_save, split_size = 100, 4096  # every 4096 rays will make up a .npy file
        data, t0, split = [], time.time(), 0
        timer = Timer(args.n_pose_kd)
        for i in range(1, args.n_pose_kd + 1):
            pose = get_rand_pose()
            focal_ = focal * (
                np.random.rand() +
                1) if args.use_rand_focal else focal  # scale focal by [1, 2)
            rays_o, rays_d = get_rays1(H, W, focal_,
                                       pose)  # rays_o, rays_d shape: [H, W, 3]
            batch_rays = torch.stack([rays_o, rays_d], dim=0)  # [2, H, W, 3]
            rgb, *_ = render(
                H,
                W,
                focal,
                chunk=args.chunk,
                rays=
                batch_rays,  # when batch_rays are given, it will not create rays inside 'render'
                verbose=False,
                retraw=False,
                **render_kwargs_)
            rays_o, rays_d, rgb = rays_o.data.cpu().numpy(), rays_d.cpu(
            ).data.numpy(), rgb.cpu().data.numpy()

            # for each pixel, get its neighbor pixel, add it to the data
            offset = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1],
                      [1, -1], [1, 0], [1, 1]]  # 3x3, 9 locations
            rays_d3x3 = np.zeros((H, W, len(offset) * 3))
            rgb3x3 = np.zeros((H, W, len(offset) * 3))
            for h in range(1, H - 1):  # [1, H-2]
                for w in range(1, W - 1):  # [1, W-2]
                    dirs, rgbs = [], []
                    for offset_h, offset_w in offset:
                        dirs += list(rays_d[h + offset_h,
                                            w + offset_w])  # three numbers
                        rgbs += list(rgb[h + offset_h,
                                         w + offset_w])  # three numbers
                    rays_d3x3[h, w] = dirs[:]
                    rgb3x3[h, w] = rgbs[:]
            rays_d3x3 = np.array(rays_d3x3[1:H - 1, 1:W - 1])  # [H-2, W-2, 27]
            rgb3x3 = np.array(rgb3x3[1:H - 1, 1:W - 1])  # [H-2, W-2, 27]
            rays_o = rays_o[1:H - 1, 1:W - 1]  # [H-2, W-2, 3]
            data_ = np.concatenate([rays_o, rays_d3x3, rgb3x3],
                                   axis=-1)  # [H-2, W-2, 57]
            data_ = data_.reshape(-1, data_.shape[2])  # [(H-2)*(W-2), 57]

            data += [data_]
            print(
                f'[{i}/{args.n_pose_kd}] Using teacher to render more images... elapsed time: {(time.time() - t0):.2f}s'
            )
            print(f'Predicted finish time: {timer()}')

            # check pseudo images
            if i <= 5:
                filename = f'{datadir_kd_new}/pseudo_sample_{i}.png'
                imageio.imwrite(filename, to8b(rgb))

            # save to avoid out of memory
            if i % i_save == 0:
                data = np.concatenate(data, axis=0)  # [i_save*(H-2)*(W-2), 57]

                # shuffle rays
                rand_ix1 = np.random.permutation(data.shape[0])
                rand_ix2 = np.random.permutation(data.shape[0])
                data = data[rand_ix1][rand_ix2]

                # save
                num = data.shape[0] // split_size * split_size
                for ix in range(0, num, split_size):
                    split += 1
                    save_path = f'{datadir_kd_new}/data_{split % args.max_save}.npy'  # to maintain similar total size
                    d = data[ix:ix + split_size]
                    np.save(save_path, d)
                print(
                    f'[{i}/{args.n_pose_kd}] Saved data at "{datadir_kd_new}"')
                data = []  # reset

    elif args.create_data in ['16x16patches']:  # for nerf_v3.4
        # set up data directory
        if os.path.exists(datadir_kd_new):
            if os.path.isfile(datadir_kd_new):
                os.remove(datadir_kd_new)
            else:
                shutil.rmtree(datadir_kd_new)
        os.makedirs(datadir_kd_new)
        print('Set up new data directory, done!')

        # set up model
        render_kwargs_ = {x: v for x, v in render_kwargs_train.items()}
        render_kwargs_['network_fn'] = render_kwargs_train[
            'teacher_fn']  # temporarily change the network_fn
        render_kwargs_['network_fine'] = render_kwargs_train[
            'teacher_fine']  # temporarily change the network_fine
        render_kwargs_.pop('teacher_fn')
        render_kwargs_.pop('teacher_fine')

        # run
        patch_size = 16
        t0 = time.time()
        timer = Timer(args.n_pose_kd)
        for img_ix in range(1, args.n_pose_kd + 1):
            # all the patches of the same image are stored together
            img_folder = f'{datadir_kd_new}/img_{img_ix}'
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)

            # core forward
            pose = get_rand_pose()
            focal_ = focal * (np.random.rand() + 1)  # scale focal by [1, 2)
            rays_o, rays_d = get_rays1(H, W, focal_,
                                       pose)  # rays_o, rays_d shape: [H, W, 3]
            batch_rays = torch.stack([rays_o, rays_d], dim=0)  # [2, H, W, 3]
            rgb, *_ = render(
                H,
                W,
                focal,
                chunk=args.chunk,
                rays=
                batch_rays,  # when batch_rays are given, it will not create rays inside 'render'
                verbose=False,
                retraw=False,
                **render_kwargs_)

            # save rays_o
            rays_o_save_path = f'{img_folder}/rays_o.npy'
            np.save(rays_o_save_path, rays_o[0, 0, :].data.cpu().numpy())

            # save rays_d and rgb
            data_ = torch.cat([rays_d, rgb],
                              dim=-1).data.cpu().numpy()  # [H, W, 6]
            num_h, num_w = H // patch_size, W // patch_size
            for h_ix in range(num_h):
                for w_ix in range(num_w):
                    d = data_[h_ix * patch_size:(h_ix + 1) * patch_size,
                              w_ix * patch_size:(w_ix + 1) * patch_size, :]
                    save_path = f'{img_folder}/patch_{h_ix * num_w + w_ix}.npy'
                    np.save(save_path, d)
            print(
                f'[{img_ix}/{args.n_pose_kd}] Using teacher to render more images... elapsed time: {(time.time() - t0):.2f}s'
            )
            print(f'Predicted finish time: {timer()}')

            # check pseudo images
            if img_ix <= 5:
                filename = f'{datadir_kd_new}/pseudo_sample_{img_ix}.png'
                imageio.imwrite(filename, to8b(rgb))

    elif args.create_data in ['16x16patches_v2']:  # improve above 16x16patches
        # set up data directory
        if os.path.exists(datadir_kd_new):
            if os.path.isfile(datadir_kd_new):
                os.remove(datadir_kd_new)
            else:
                shutil.rmtree(datadir_kd_new)
        os.makedirs(datadir_kd_new)
        print('Set up new data directory, done!')

        # set up model
        render_kwargs_ = {x: v for x, v in render_kwargs_train.items()}
        render_kwargs_['network_fn'] = render_kwargs_train[
            'teacher_fn']  # temporarily change the network_fn
        render_kwargs_['network_fine'] = render_kwargs_train[
            'teacher_fine']  # temporarily change the network_fine
        render_kwargs_.pop('teacher_fn')
        render_kwargs_.pop('teacher_fine')

        # run
        patch_size = 16
        t0 = time.time()
        timer = Timer(args.n_pose_kd)
        for img_ix in range(1, args.n_pose_kd + 1):
            # core forward
            pose = get_rand_pose()
            focal_ = focal  # * (np.random.rand() + 1) # scale focal by [1, 2) ==> use fixed focal!!
            rays_o, rays_d = get_rays1(H, W, focal_,
                                       pose)  # rays_o, rays_d shape: [H, W, 3]
            batch_rays = torch.stack([rays_o, rays_d], dim=0)  # [2, H, W, 3]
            rgb, *_ = render(
                H,
                W,
                focal,
                chunk=args.chunk,
                rays=
                batch_rays,  # when batch_rays are given, it will not create rays inside 'render'
                verbose=False,
                retraw=False,
                **render_kwargs_)

            # save rays_o
            rays_o_save_path = f'{datadir_kd_new}/img{img_ix}_rays_o.npy'
            np.save(rays_o_save_path, rays_o[0, 0, :].data.cpu().numpy())

            # save rays_d and rgb
            data_ = torch.cat([rays_d, rgb],
                              dim=-1).data.cpu().numpy()  # [H, W, 6]
            num_h, num_w = H // patch_size, W // patch_size
            for h_ix in range(num_h):
                for w_ix in range(num_w):
                    d = data_[h_ix * patch_size:(h_ix + 1) * patch_size,
                              w_ix * patch_size:(w_ix + 1) * patch_size, :]
                    save_path = f'{datadir_kd_new}/img{img_ix}_patch{h_ix * num_w + w_ix}_rays_d.npy'
                    np.save(save_path, d)
            print(
                f'[{img_ix}/{args.n_pose_kd}] Using teacher to render more images... elapsed time: {(time.time() - t0):.2f}s'
            )
            print(f'Predicted finish time: {timer()}')

            # check pseudo images
            if img_ix <= 5:
                filename = f'{datadir_kd_new}/pseudo_sample_{img_ix}.png'
                imageio.imwrite(filename, to8b(rgb))

    elif args.create_data in ['16x16patches_v3']:  # improve above 16x16patches
        # set up data directory
        if os.path.exists(datadir_kd_new):
            if os.path.isfile(datadir_kd_new):
                os.remove(datadir_kd_new)
            else:
                shutil.rmtree(datadir_kd_new)
        os.makedirs(datadir_kd_new)
        print(f'Set up new data directory: {datadir_kd_new}, done!')

        # set up model
        render_kwargs_ = {x: v for x, v in render_kwargs_train.items()}
        render_kwargs_['network_fn'] = render_kwargs_train[
            'teacher_fn']  # temporarily change the network_fn
        render_kwargs_['network_fine'] = render_kwargs_train[
            'teacher_fine']  # temporarily change the network_fine
        render_kwargs_.pop('teacher_fn')
        render_kwargs_.pop('teacher_fine')

        # run
        patch_size, i_save, split_size, split, data_save = 16, 100, 32, 0, []
        t0 = time.time()
        timer = Timer(args.n_pose_kd)
        for img_ix in range(1, args.n_pose_kd + 1):
            # core forward
            pose = get_rand_pose()
            focal_ = focal  # * (np.random.rand() + 1) # scale focal by [1, 2) ==> use fixed focal!!
            rays_o, rays_d = get_rays1(H, W, focal_,
                                       pose)  # rays_o, rays_d shape: [H, W, 3]
            batch_rays = torch.stack([rays_o, rays_d], dim=0)  # [2, H, W, 3]
            rgb, *_ = render(
                H,
                W,
                focal,
                chunk=args.chunk,
                rays=
                batch_rays,  # when batch_rays are given, it will not create rays inside 'render'
                verbose=False,
                retraw=False,
                **render_kwargs_)

            # save rays_d and rgb
            data_ = torch.cat([rays_o, rays_d, rgb],
                              dim=-1).data.cpu()  # [H, W, 9]
            num_h, num_w = H // patch_size, W // patch_size
            for h_ix in range(num_h):
                for w_ix in range(num_w):
                    d = data_[h_ix * patch_size:(h_ix + 1) * patch_size,
                              w_ix * patch_size:(w_ix + 1) *
                              patch_size, :]  # [patch_size, patch_size, 9]
                    data_save += [d]

            if img_ix % i_save == 0:
                data_save = torch.stack(
                    data_save, dim=0)  # [n_patch, patch_size, patch_size, 6]
                # shuffle
                n_shuffle = 5
                for _ in range(n_shuffle):
                    rand_ix = np.random.permutation(data_save.shape[0])
                    data_save = data_save[rand_ix]

                # save
                num = data_save.shape[0] // split_size * split_size
                for ix in range(0, num, split_size):
                    split += 1
                    save_path = f'{datadir_kd_new}/data_{split % args.max_save}.npy'  # to maintain similar total size
                    d = data_save[ix:ix + split_size].data.cpu().numpy()
                    np.save(save_path, d)
                print(
                    f'[{img_ix}/{args.n_pose_kd}] Saved data at "{datadir_kd_new}"'
                )
                data_save = []  # reset

            print(
                f'[{img_ix}/{args.n_pose_kd}] Using teacher to render more images... elapsed time: {(time.time() - t0):.2f}s'
            )
            print(f'Predicted finish time: {timer()}')

            # check pseudo images
            if img_ix <= 5:
                filename = f'{datadir_kd_new}/pseudo_sample_{img_ix}.png'
                imageio.imwrite(filename, to8b(rgb))


if __name__ == '__main__':
    train()
