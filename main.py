import os, sys, copy, math, random, json, time

import imageio
from tqdm import tqdm, trange
import numpy as np
import lpips as lpips_

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as benchmark

from model.nerf_raybased import NeRF, NeRF_v3_2, PositionalEmbedder, PointSampler
from dataset.load_llff import load_llff_data
from dataset.load_deepvoxels import load_dv_data
from dataset.load_blender import load_blender_data, BlenderDataset, BlenderDataset_v2, get_novel_poses
from utils.ssim_torch import ssim as ssim_
from utils.flip_loss import FLIP
from utils.run_nerf_raybased_helpers import sample_pdf, ndc_rays, get_rays, get_embedder, get_rays_np
from utils.run_nerf_raybased_helpers import parse_expid_iter, to_tensor, to_array, mse2psnr, to8b, img2mse
from utils.run_nerf_raybased_helpers import load_weights_v2, get_selected_coords, undataparallel
from smilelogging import Logger
from smilelogging.utils import Timer, LossLine, get_n_params_, get_n_flops_, AverageMeter, ProgressMeter
from option import args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

# ---------------------------------
# Set up logging directories
logger = Logger(args)
accprint = logger.log_printer.accprint
netprint = logger.log_printer.netprint
ExpID = logger.ExpID
flip = FLIP()


class MyDataParallel(torch.nn.DataParallel):

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


# Update ssim and lpips metric functions
ssim = lambda img, ref: ssim_(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))
lpips = lpips_.LPIPS(net=args.lpips_net).to(device)
# ---------------------------------


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
                render_factor=0):
    H, W, focal = hwf
    if render_factor != 0:
        # Render downsampled for speed
        H = int(H / render_factor)
        W = int(W / render_factor)
        focal = focal / render_factor

    render_kwargs['network_fn'].eval()
    rgbs, disps, errors, ssims, psnrs = [], [], [], [], []

    # for testing DONERF data
    if args.given_render_path_rays:
        loaded = torch.load(args.given_render_path_rays)
        all_rays_o = loaded['all_rays_o'].to(device)  # [N, H*W, 3]
        all_rays_d = loaded['all_rays_d'].to(device)  # [N, H*W, 3]
        if 'gt_imgs' in loaded:
            gt_imgs = loaded['gt_imgs'].to(device)  # [N, H, W, 3]
        print(f'Use given render_path rays: "{args.given_render_path_rays}"')

        model = render_kwargs['network_fn']
        for i in range(len(all_rays_o)):
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                pts = point_sampler.sample_train(
                    all_rays_o[i], all_rays_d[i],
                    perturb=0)  # [H*W, n_sample*3]
                model_input = positional_embedder(pts)
                torch.cuda.synchronize()
                t_input = time.time()
                if args.learn_depth:
                    rgbd = model(model_input)
                    rgb = rgbd[:, :3]
                else:
                    rgb = model(model_input)
                torch.cuda.synchronize()
                t_forward = time.time()
                print(
                    f'[#{i}] frame, prepare input (embedding): {t_input - t0:.4f}s'
                )
                print(
                    f'[#{i}] frame, model forward: {t_forward - t_input:.4f}s')

                # reshape to image
                if args.dataset_type == 'llff':
                    H_, W_ = H, W  # non-square images
                elif args.dataset_type == 'blender':
                    H_ = W_ = int(math.sqrt(rgb.numel() / 3))
                rgb = rgb.view(H_, W_, 3)
                disp = rgb  # placeholder, to maintain compability

                rgbs.append(rgb)
                disps.append(disp)

                # @mst: various metrics
                if gt_imgs is not None:
                    errors += [(rgb - gt_imgs[i][:H_, :W_, :]).abs()]
                    psnrs += [mse2psnr(img2mse(rgb, gt_imgs[i, :H_, :W_]))]
                    ssims += [ssim(rgb, gt_imgs[i, :H_, :W_])]

                if savedir is not None:
                    filename = os.path.join(savedir, '{:03d}.png'.format(i))
                    imageio.imwrite(filename, to8b(rgbs[-1]))
                    imageio.imwrite(filename.replace('.png', '_gt.png'),
                                    to8b(gt_imgs[i]))  # save gt images
                    if len(errors):
                        imageio.imwrite(filename.replace('.png', '_error.png'),
                                        to8b(errors[-1]))

                torch.cuda.synchronize()
                print(
                    f'[#{i}] frame, rendering done, time for this frame: {time.time()-t0:.4f}s'
                )
                print('')
    else:
        for i, c2w in enumerate(render_poses):
            torch.cuda.synchronize()
            t0 = time.time()
            print(f'[#{i}] frame, rendering begins')
            if args.model_name in ['nerf']:
                rgb, disp, acc, _ = render(H,
                                           W,
                                           focal,
                                           chunk=chunk,
                                           c2w=c2w[:3, :4],
                                           **render_kwargs)
                H_, W_ = H, W

            else:  # For R2L model
                model = render_kwargs['network_fn']
                perturb = render_kwargs['perturb']

                # Network forward
                with torch.no_grad():
                    if args.given_render_path_rays:  # To test DONERF data using our model
                        pts = point_sampler.sample_train(
                            all_rays_o[i], all_rays_d[i],
                            perturb=0)  # [H*W, n_sample*3]
                    else:
                        if args.plucker:
                            pts = point_sampler.sample_test_plucker(
                                c2w[:3, :4])
                        else:
                            pts = point_sampler.sample_test(
                                c2w[:3, :4])  # [H*W, n_sample*3]
                    model_input = positional_embedder(pts)
                    torch.cuda.synchronize()
                    t_input = time.time()
                    if args.learn_depth:
                        rgbd = model(model_input)
                        rgb = rgbd[:, :3]
                    else:
                        rgb = model(model_input)
                    torch.cuda.synchronize()
                    t_forward = time.time()
                    print(
                        f'[#{i}] frame, prepare input (embedding): {t_input - t0:.4f}s'
                    )
                    print(
                        f'[#{i}] frame, model forward: {t_forward - t_input:.4f}s'
                    )

                # Reshape to image
                if args.dataset_type == 'llff':
                    H_, W_ = H, W  # non-square images
                elif args.dataset_type == 'blender':
                    if args.train_depth:
                        H_ = W_ = int(math.sqrt(rgb.numel()))
                    else:
                        H_ = W_ = int(math.sqrt(rgb.numel() / 3))
                print(H_, W_)

                if args.train_depth:
                    rgb *= 2
                    rgb = rgb.view(H_, W_, 1)
                    rgb = rgb.expand(H_, W_, 3)
                else:
                    rgb = rgb.view(H_, W_, 3)

                disp = rgb  # Placeholder, to maintain compability

            rgbs.append(rgb)
            disps.append(disp)

            # @mst: various metrics
            if gt_imgs is not None:
                errors += [(rgb - gt_imgs[i][:H_, :W_, :]).abs()]
                psnrs += [mse2psnr(img2mse(rgb, gt_imgs[i, :H_, :W_]))]
                ssims += [ssim(rgb, gt_imgs[i, :H_, :W_])]

            if savedir is not None:
                filename = os.path.join(savedir, '{:03d}.png'.format(i))
                imageio.imwrite(filename, to8b(rgbs[-1]))
                imageio.imwrite(filename.replace('.png', '_gt.png'),
                                to8b(gt_imgs[i]))  # save gt images
                if len(errors):
                    imageio.imwrite(filename.replace('.png', '_error.png'),
                                    to8b(errors[-1]))

            torch.cuda.synchronize()
            print(
                f'[#{i}] frame, rendering done, time for this frame: {time.time()-t0:.4f}s'
            )
            print('')

    rgbs = torch.stack(rgbs, dim=0)
    disps = torch.stack(disps, dim=0)

    # https://github.com/richzhang/PerceptualSimilarity
    # LPIPS demands input shape [N, 3, H, W] and in range [-1, 1]
    misc = {}
    if gt_imgs is not None:
        rec = rgbs.permute(0, 3, 1, 2)  # [N, 3, H, W]
        ref = gt_imgs.permute(0, 3, 1, 2)  # [N, 3, H, W]
        rescale = lambda x, ymin, ymax: (ymax - ymin) / (x.max() - x.min()) * (
            x - x.min()) + ymin
        rec, ref = rescale(rec, -1, 1), rescale(ref, -1, 1)
        lpipses = []
        mini_batch_size = 8
        for i in np.arange(0, len(gt_imgs), mini_batch_size):
            end = min(i + mini_batch_size, len(gt_imgs))
            lpipses += [lpips(rec[i:end], ref[i:end])]
        lpipses = torch.cat(lpipses, dim=0)

        # -- get FLIP loss
        # flip standard values
        monitor_distance = 0.7
        monitor_width = 0.7
        monitor_resolution_x = 3840
        pixels_per_degree = monitor_distance * (monitor_resolution_x /
                                                monitor_width) * (np.pi / 180)
        flips = flip.compute_flip(rec, ref,
                                  pixels_per_degree)  # shape [N, 1, H, W]
        # --

        errors = torch.stack(errors, dim=0)
        psnrs = torch.stack(psnrs, dim=0)
        ssims = torch.stack(ssims, dim=0)
        test_loss = img2mse(rgbs,
                            gt_imgs[:, :H_, :W_])  # @mst-TODO: remove H_, W_

        misc['test_loss'] = test_loss
        misc['test_psnr'] = mse2psnr(test_loss)
        misc['test_psnr_v2'] = psnrs.mean()
        misc['test_ssim'] = ssims.mean()
        misc['test_lpips'] = lpipses.mean()
        misc['test_flip'] = flips.mean()
        misc['errors'] = errors

    render_kwargs['network_fn'].train()
    torch.cuda.empty_cache()
    return rgbs, disps, misc


def render_func(model, pose):
    with torch.no_grad():
        rgb = model(positional_embedder(point_sampler.sample_test(pose)))
    return rgb


def create_nerf(args, near, far):
    """Instantiate NeRF's MLP model.
    """
    # set up model
    model_fine = network_query_fn = None
    global embed_fn
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views,
                                                    args.i_embed)

    # @mst: use external positional embedding for our raybased nerf
    global positional_embedder
    positional_embedder = PositionalEmbedder(L=args.multires)

    grad_vars = []
    if args.model_name in ['nerf']:
        output_ch = 5 if args.N_importance > 0 else 4
        skips = [4]
        model = NeRF(D=args.netdepth,
                     W=args.netwidth,
                     input_ch=input_ch,
                     output_ch=output_ch,
                     skips=skips,
                     input_ch_views=input_ch_views,
                     use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model.parameters())

        if args.N_importance > 0:
            model_fine = NeRF(D=args.netdepth_fine,
                              W=args.netwidth_fine,
                              input_ch=input_ch,
                              output_ch=output_ch,
                              skips=skips,
                              input_ch_views=input_ch_views,
                              use_viewdirs=args.use_viewdirs).to(device)
            grad_vars += list(model_fine.parameters())

        network_query_fn = lambda inputs, viewdirs, network_fn: run_network(
            inputs,
            viewdirs,
            network_fn,
            embed_fn=embed_fn,
            embeddirs_fn=embeddirs_fn,
            netchunk=args.netchunk)

    elif args.model_name in ['nerf_v3.2', 'R2L']:
        if args.plucker:
            input_dim = 6 * positional_embedder.embed_dim
        else:
            input_dim = args.n_sample_per_ray * 3 * positional_embedder.embed_dim
        model = NeRF_v3_2(args, input_dim, 3).to(device)
        if not args.freeze_pretrained:
            grad_vars += list(model.parameters())

    elif args.model_name in ['SilhouetteNeRF']:
        input_dim = args.n_sample_per_ray * 3 * positional_embedder.embed_dim
        model = NeRF_v3_2(args, input_dim, 1).to(device)
        if not args.freeze_pretrained:
            grad_vars += list(model.parameters())

    # set up optimizer
    optimizer = torch.optim.Adam(params=grad_vars,
                                 lr=args.lrate,
                                 betas=(0.9, 0.999))

    # start iteration
    history = {'start': 0, 'best_psnr': 0, 'best_psnr_step': 0}

    # use DataParallel
    if not args.render_only:  # when rendering, use just one GPU
        model = MyDataParallel(model)
        if model_fine is not None:
            model_fine = MyDataParallel(model_fine)
        if hasattr(model.module, 'input_dim'):
            model.input_dim = model.module.input_dim
        print(f'Using data parallel')

    # load pretrained checkpoint
    if args.pretrained_ckpt:
        ckpt = torch.load(args.pretrained_ckpt)
        if 'network_fn' in ckpt:
            model = ckpt['network_fn']
            grad_vars = list(model.parameters())
            if model_fine is not None:
                assert 'network_fine' in ckpt
                model_fine = ckpt['network_fine']
                grad_vars += list(model_fine.parameters())
            optimizer = torch.optim.Adam(params=grad_vars,
                                         lr=args.lrate,
                                         betas=(0.9, 0.999))
            print(
                f'Use model arch saved in checkpoint "{args.pretrained_ckpt}", and build a new optimizer.'
            )

        # load state_dict
        load_weights_v2(model, ckpt, 'network_fn_state_dict')
        if model_fine is not None:
            load_weights_v2(model_fine, ckpt, 'network_fine_state_dict')
        print(f'Load pretrained ckpt successfully: "{args.pretrained_ckpt}".')

        if args.resume:
            history['start'] = ckpt['global_step']
            history['best_psnr'] = ckpt.get('best_psnr', 0)
            history['best_psnr_step'] = ckpt.get('best_psnr_step', 0)
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print('Resume optimizer successfully.')

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
    render_kwargs_test['perturb'] = args.perturb_test
    render_kwargs_test['raw_noise_std'] = 0.

    # get FLOPs and params
    netprint(model)
    n_params = get_n_params_(model)
    if args.model_name == 'nerf':
        dummy_input = torch.randn(1, input_ch + input_ch_views).to(device)
        n_flops = get_n_flops_(model, input=dummy_input, count_adds=False) * (
            args.N_samples + args.N_samples + args.N_importance)

    elif args.model_name in ['nerf_v3.2', 'R2L', 'SilhouetteNeRF']:
        dummy_input = torch.randn(1, model.input_dim).to(device)
        n_flops = get_n_flops_(model, input=dummy_input, count_adds=False)

    print(
        f'Model complexity per pixel: FLOPs {n_flops/1e6:.10f}M, Params {n_params/1e6:.10f}M'
    )
    return render_kwargs_train, render_kwargs_test, history, grad_vars, optimizer


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

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

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

    z_vals = z_vals.expand([N_rays, N_samples])

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

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
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


def InfiniteSampler(n):
    order = np.random.permutation(n)
    i = 0
    while True:
        yield order[i]
        i += 1
        if i == n:
            order = np.random.permutation(n)
            i = 0


from torch.utils import data


class InfiniteSamplerWrapper(data.sampler.Sampler):

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2**31


def get_dataloader(dataset_type, datadir, pseudo_ratio=0.5):
    if dataset_type in ['blender', 'llff']:
        if args.data_mode in ['images']:
            trainset = BlenderDataset(datadir, pseudo_ratio)
            trainloader = torch.utils.data.DataLoader(
                dataset=trainset,
                batch_size=1,
                num_workers=args.num_workers,
                pin_memory=True,
                sampler=InfiniteSamplerWrapper(len(trainset)))
        elif args.data_mode in ['rays']:
            if args.model_name in ['SilhouetteNeRF']:
                trainset = BlenderDataset_v2(
                    datadir,
                    dim_dir=3,
                    dim_rgb=1,
                    hold_ratio=args.pseudo_data_hold_ratio,
                    pseudo_ratio=args.pseudo_ratio)
                trainloader = torch.utils.data.DataLoader(
                    dataset=trainset,
                    batch_size=args.N_rand,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    sampler=InfiniteSamplerWrapper(len(trainset)))
            else:
                trainset = BlenderDataset_v2(
                    datadir,
                    dim_dir=3,
                    dim_rgb=3,
                    hold_ratio=args.pseudo_data_hold_ratio,
                    pseudo_ratio=args.pseudo_ratio)
                trainloader = torch.utils.data.DataLoader(
                    dataset=trainset,
                    batch_size=args.N_rand,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    sampler=InfiniteSamplerWrapper(len(trainset)))
    return iter(trainloader), len(trainset)


def get_pseudo_ratio(schedule, current_step):
    '''example of schedule: 1:0.2,500000:0.9'''
    steps, prs = [], []
    for item in schedule.split(','):
        step, pr = item.split(':')
        step, pr = int(step), float(pr)
        steps += [step]
        prs += [pr]

    # linear scheduling
    if current_step < steps[0]:
        pr = prs[0]
    elif current_step > steps[1]:
        pr = prs[1]
    else:
        pr = (prs[1] - prs[0]) / (steps[1] - steps[0]) * (current_step -
                                                          steps[0]) + prs[0]
    return pr


def save_onnx(model, onnx_path, dummy_input):
    model = copy.deepcopy(model)
    if hasattr(model, 'module'):
        model = model.module
    torch.onnx.export(model.cpu(),
                      dummy_input.cpu(),
                      onnx_path,
                      verbose=True,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      keep_initializers_as_inputs=False,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={
                          'input': {
                              0: 'batch_size'
                          },
                          'output': {
                              0: 'batch_size'
                          }
                      })
    del model


#TODO-@mst: move these utility functions to a better place
def check_onnx(model, onnx_path, dummy_input):
    r"""Refer to https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
    """
    import onnx, onnxruntime
    model = copy.deepcopy(model)
    if hasattr(model, 'module'):
        model = model.module
    model, dummy_input = model.cpu(), dummy_input.cpu()
    torch_out = model(dummy_input)

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(onnx_path)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy(
        ) if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out),
                               ort_outs[0],
                               rtol=1e-03,
                               atol=1e-05)
    print(
        "Exported model has been tested with ONNXRuntime, and the result looks good!"
    )


def train():
    # Load data
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(
            args.datadir,
            args.factor,
            recenter=True,
            bd_factor=.75,
            spherify=args.spherify,
            n_pose_video=args.n_pose_video)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape, render_poses.shape, hwf,
              args.datadir)

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

    elif args.dataset_type == 'blender':
        if args.train_depth:
            images, poses, render_poses, hwf, i_split = load_blender_data(
                args.datadir, args.half_res, args.testskip, depth=True)
        else:
            images, poses, render_poses, hwf, i_split = load_blender_data(
                args.datadir, args.half_res, args.testskip)
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

    # @mst
    if hasattr(args, 'trial') and args.trial.near > 0:
        assert args.trial.far > args.trial.near
        near, far = args.trial.near, args.trial.far
        print(f'Use provided near ({near}) and far {far}')

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
    render_kwargs_train, render_kwargs_test, history, grad_vars, optimizer = create_nerf(
        args, near, far)
    print(f'Created model {args.model_name}')
    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    start, best_psnr, best_psnr_step = history['start'], history[
        'best_psnr'], history['best_psnr_step']

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W, focal = int(H), int(W), float(focal)
    if args.focal_scale > 0:
        focal *= args.focal_scale
        print(f'!! Focal changed to {focal} (scaled by {args.focal_scale})')
    hwf = [H, W, focal]
    k = math.sqrt(float(args.N_rand) / H / W)
    patch_h, patch_w = int(H * k), int(W * k)
    global IMG_H
    global IMG_W
    IMG_H, IMG_W = H, W

    # Set up sampler
    global point_sampler
    point_sampler = PointSampler(H, W, focal, args.n_sample_per_ray, near, far)

    # Get train, test, video poses and images
    train_images, train_poses = images[i_train], poses[i_train]
    test_poses, test_images = poses[i_test], images[i_test]
    if args.train_depth:
        test_images[test_images == 1] = 0
        if args.does_terminate:
            test_images[test_images != 0] = 1
    n_original_img = len(train_images)
    if args.dataset_type == 'blender':
        # @mst: for blender dataset, get more diverse video poses
        video_poses = get_novel_poses(args, n_pose=args.n_pose_video)
    else:
        video_poses = render_poses

    # check test poses and video poses
    avg_test_pose = test_poses.mean(dim=0)
    avg_video_pose = video_poses.mean(dim=0)
    avg_train_pose = train_poses.mean(dim=0)
    netprint(f'avg_test_pose:')
    netprint(avg_test_pose)
    netprint(f'avg_video_pose:')
    netprint(avg_video_pose)
    netprint(f'avg_train_pose:')
    netprint(avg_train_pose)

    # data sketch
    print(
        f'{len(i_train)} original train views are [{" ".join([str(x) for x in i_train])}]'
    )
    print(
        f'{len(i_test)} test views are [{" ".join([str(x) for x in i_test])}]')
    print(f'{len(i_val)} val views are [{" ".join([str(x) for x in i_val])}]')
    print(
        f'train_images shape {train_images.shape} train_poses shape {train_poses.shape} test_poses shape {test_poses.shape}'
    )

    if args.test_pretrained:
        print('Testing pretrained...')
        with torch.no_grad():
            *_, misc = render_path(test_poses,
                                   hwf,
                                   4096,
                                   render_kwargs_test,
                                   gt_imgs=test_images,
                                   render_factor=args.render_factor)
        print(
            f"Pretrained test: TestLoss {misc['test_loss'].item():.4f} TestPSNR {misc['test_psnr'].item():.4f} TestPSNRv2 {misc['test_psnr_v2'].item():.4f}"
        )

    # @mst: use dataloader for training
    kd_poses = None
    if args.datadir_kd:
        global datadir_kd
        datadir_kd = args.datadir_kd.split(
            ':')[1] if ':' in args.datadir_kd else args.datadir_kd
        if args.dataset_type in ['blender', 'llff']:
            trainloader, n_total_img = get_dataloader(args.dataset_type,
                                                      datadir_kd)
        else:
            raise NotImplementedError
        print(f'Loaded data. Now total #train files: {n_total_img}')

    # @mst: get video_targets
    video_targets = None

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        expid, iter_ = parse_expid_iter(args.pretrained_ckpt)
        with torch.no_grad():
            t_ = time.time()
            if args.render_test:
                print('Rendering test images...')
                rgbs, *_, misc = render_path(test_poses,
                                             hwf,
                                             args.chunk,
                                             render_kwargs_test,
                                             gt_imgs=test_images,
                                             savedir=logger.gen_img_path,
                                             render_factor=args.render_factor)
                print(
                    f"[TEST] TestPSNR {misc['test_psnr'].item():.4f} TestPSNRv2 {misc['test_psnr_v2'].item():.4f} TestSSIM {misc['test_ssim'].item():.4f} TestLPIPS {misc['test_lpips'].item():.4f} TestFLIP {misc['test_flip'].item():.4f}"
                )
            else:
                if args.dataset_type == 'blender':
                    video_poses = get_novel_poses(args,
                                                  n_pose=args.n_pose_video)
                else:
                    video_poses = render_poses
                print(f'Rendering video... (n_pose: {len(video_poses)})')
                rgbs, *_, misc = render_path(video_poses,
                                             hwf,
                                             args.chunk,
                                             render_kwargs_test,
                                             gt_imgs=video_targets,
                                             render_factor=args.render_factor)
            t = time.time() - t_
        video_path = f'{logger.gen_img_path}/video_{expid}_iter{iter_}_{args.video_tag}.mp4'
        imageio.mimwrite(video_path, to8b(rgbs), fps=30, quality=8)
        print(f'Save video: "{video_path} (time: {t:.2f}s)')
        if 'errors' in misc:
            imageio.mimwrite(video_path.replace('.mp4', '_error.mp4'),
                             to8b(misc['errors']),
                             fps=30,
                             quality=8)
        exit(0)

    if args.convert_to_onnx:
        if args.pretrained_ckpt:
            onnx_path = args.pretrained_ckpt.replace('.tar', '.onnx')
        else:
            onnx_path = f'{logger.weights_path}/ckpt.onnx'
        mobile_H, mobile_W = 256, 256
        if args.model_name in ['nerf_v3.2', 'R2L', 'SilhouetteNeRF']:
            dummy_input = torch.randn(
                1, mobile_H, mobile_W,
                render_kwargs_test['network_fn'].input_dim).to(device)
        else:
            raise NotImplementedError

        save_onnx(render_kwargs_test['network_fn'], onnx_path, dummy_input)
        check_onnx(render_kwargs_test['network_fn'], onnx_path, dummy_input)
        print(f'Convert to onnx done. Saved at "{onnx_path}"')
        exit(0)

    if args.benchmark:
        x = video_poses[0]
        timer = benchmark.Timer(stmt='render_func(model, pose)',
                                setup='from __main__ import render_func',
                                globals={
                                    'model': render_kwargs_test['network_fn'],
                                    'pose': x
                                })
        print(timer.timeit(100))
        exit(0)

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack(
            [get_rays_np(H, W, focal, p) for p in poses[:, :3, :4]],
            0)  # [N, ro+rd, H, W, 3]
        print('done, concats')
        if isinstance(images, torch.Tensor): images = images.cpu().data.numpy()
        rays_rgb = np.concatenate([rays, images[:, None]],
                                  1)  # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb,
                                [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train],
                            0)  # train images only
        rays_rgb = np.reshape(rays_rgb,
                              [-1, 3, 3])  # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)
        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    if args.hard_ratio:
        hard_rays = to_tensor([])

    # training
    timer = Timer((args.N_iters - start) // args.i_testset)
    hist_psnr = hist_psnr1 = hist_psnr2 = n_pseudo_img = n_seen_img = hist_depthloss = 0
    global global_step
    print('Begin training')
    hard_pool_full = False
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    for i in trange(start + 1, args.N_iters + 1):
        t0 = time.time()
        global_step = i
        loss_line = LossLine()
        loss, rgb1 = 0, None

        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000

        if args.warmup_lr:  # @mst: example '0.0001,2000'
            start_lr, end_iter = [float(x) for x in args.warmup_lr.split(',')]
            if global_step < end_iter:  # increase lr until args.lrate
                new_lrate = (args.lrate -
                             start_lr) / end_iter * global_step + start_lr
            else:  # decrease lr as before
                new_lrate = args.lrate * (decay_rate**(
                    (global_step - end_iter) / decay_steps))
        else:
            new_lrate = args.lrate * (decay_rate**(global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        # >>>>>>>>>>> inner loop (get data, forward, add loss) begins >>>>>>>>>>>
        # Sample random ray batch
        if use_batching:  # @mst: False in default
            # Random over all images
            batch = rays_rgb[i_batch:i_batch + N_rand]  # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0
        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            pose = poses[img_i, :3, :4]

            # KD, update dataloader
            if args.datadir_kd:
                if args.data_mode in ['images']:
                    if i % args.i_update_data == 0:  # update trainloader, possibly load more data
                        if args.dataset_type == 'blender':
                            t_ = time.time()
                            pr = get_pseudo_ratio(args.pseudo_ratio_schedule,
                                                  i)
                            trainloader, n_total_img = get_dataloader(
                                args.dataset_type, datadir_kd, pseudo_ratio=pr)
                            print(
                                f'Iter {i}. Reloaded data (time: {time.time()-t_:.2f}s). Now total #train files: {n_total_img}'
                            )

                    # get pose and target
                    if args.dataset_type == 'blender':
                        target, pose, img_i = [
                            x[0] for x in trainloader.next()
                        ]  # batch size = 1
                        target, pose = target.to(device), pose.to(device)
                        pose = pose[:3, :4]
                        if img_i >= n_original_img:
                            n_pseudo_img += 1

                    else:  # LLFF dataset
                        use_pseudo_img = torch.rand(1) < len(kd_poses) / (
                            len(train_poses) + len(kd_poses))
                        if use_pseudo_img:
                            img_i = np.random.permutation(len(kd_poses))[0]
                            pose = kd_poses[img_i, :3, :4]
                            target = kd_targets[img_i]
                            n_pseudo_img += 1

                    n_seen_img += 1
                    loss_line.update('pseudo_img_ratio',
                                     n_pseudo_img / n_seen_img, '.4f')

                elif args.data_mode in ['rays']:
                    if i % args.i_update_data == 0:  # update trainloader, possibly load more data
                        if args.dataset_type in ['blender', 'llff']:
                            t_ = time.time()
                            trainloader, n_total_img = get_dataloader(
                                args.dataset_type, datadir_kd)
                            print(
                                f'Iter {i}. Reloaded data (time: {time.time()-t_:.2f}s). Now total #train files: {n_total_img}'
                            )

            # get rays (rays_o, rays_d, target_s)
            if N_rand is not None:
                if args.data_mode in ['images']:
                    rays_o, rays_d = get_rays(
                        H, W, focal, pose
                    )  # (H, W, 3), (H, W, 3), origin: (-1.8393, -1.0503,  3.4298)
                    if i < args.precrop_iters:
                        dH = int(H // 2 * args.precrop_frac)
                        dW = int(W // 2 * args.precrop_frac)
                        coords = torch.stack(
                            torch.meshgrid(
                                torch.linspace(H // 2 - dH, H // 2 + dH - 1,
                                               2 * dH),
                                torch.linspace(W // 2 - dW, W // 2 + dW - 1,
                                               2 * dW)), -1)
                        if i == start + 1:
                            print(
                                f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}"
                            )
                    else:
                        coords = torch.stack(
                            torch.meshgrid(torch.linspace(0, H - 1, H),
                                           torch.linspace(0, W - 1, W)),
                            -1)  # (H, W, 2)

                    # select pixels as a batch
                    select_coords, patch_bbx = get_selected_coords(
                        coords, N_rand, args.select_pixel_mode)

                    # get rays_o and rays_d for the selected pixels
                    rays_o = rays_o[select_coords[:, 0],
                                    select_coords[:, 1]]  # (N_rand, 3)
                    rays_d = rays_d[select_coords[:, 0],
                                    select_coords[:, 1]]  # (N_rand, 3)
                    batch_rays = torch.stack([rays_o, rays_d], 0)

                    # get target for the selected pixels
                    target_s = target[select_coords[:, 0],
                                      select_coords[:, 1]]  # (N_rand, 3)

                elif args.data_mode in ['rays']:
                    rays_o, rays_d, target_s = trainloader.next(
                    )  # rays_o: [N_rand, 4096, 3] rays_d: [N_rand, 4096, 3] target_s: [N_rand, 4096, 3]
                    rays_o, rays_d, target_s = rays_o.to(device), rays_d.to(
                        device), target_s.to(device)
                    rays_o = rays_o.view(-1, 3)  # [N_rand*4096, 3]
                    rays_d = rays_d.view(-1, 3)  # [N_rand*4096, 3]
                    if args.train_depth:
                        target_s = target_s.view(-1, 1)  # [N_rand*4096, 1]
                    else:
                        target_s = target_s.view(-1, 3)  # [N_rand*4096, 3]

                    if args.shuffle_input:
                        rays_d = rays_d.view(rays_d.shape[0], 3 // 3,
                                             3)  # [N_rand*4096, 3//3, 3]
                        if args.train_depth:
                            target_s = target_s.view(target_s.shape[0], 1 // 1,
                                                     1)  # [N_rand*4096, 1//1, 1]
                        else:
                            target_s = target_s.view(target_s.shape[0], 3 // 3,
                                                     3)  # [N_rand*4096, 3//3, 3]

                        shuffle_input_randix = torch.randperm(3 // 3)
                        rays_d = rays_d[:, shuffle_input_randix, :]
                        target_s = target_s[:, shuffle_input_randix, :]
                        rays_d = rays_d.view(-1, 3)  # [N_rand*4096, 3]
                        if args.train_depth:
                            target_s = target_s.view(-1, 1)  # [N_rand*4096, 1]
                        else:
                            target_s = target_s.view(-1, 3)  # [N_rand*4096, 3]

            batch_size = rays_o.shape[0]
            print(f"Batch size: {batch_size}")
            if args.hard_ratio:
                if isinstance(args.hard_ratio, list):
                    n_hard_in = int(
                        args.hard_ratio[0] * batch_size
                    )  # the number of hard samples into the hard pool
                    n_hard_out = int(
                        args.hard_ratio[1] * batch_size
                    )  # the number of hard samples out of the hard pool
                else:
                    n_hard_in = int(args.hard_ratio * batch_size)
                    n_hard_out = n_hard_in
                n_hard_in = min(n_hard_in,
                                n_hard_out)  # n_hard_in <= n_hard_out

            if hard_pool_full:
                rand_ix_out = np.random.permutation(
                    hard_rays.shape[0])[:n_hard_out]
                picked_hard_rays = hard_rays[rand_ix_out]
                rays_o = torch.cat([rays_o, picked_hard_rays[:, :3]], dim=0)
                rays_d = torch.cat([rays_d, picked_hard_rays[:, 3:3 + 3]],
                                   dim=0)
                target_s = torch.cat([target_s, picked_hard_rays[:, 3 + 3:]],
                                     dim=0)

        # update data time
        data_time.update(time.time() - t0)

        # forward and get loss
        if args.model_name == 'nerf':
            rgb, disp, acc, extras = render(H,
                                            W,
                                            focal,
                                            chunk=args.chunk,
                                            rays=batch_rays,
                                            verbose=i < 10,
                                            retraw=True,
                                            **render_kwargs_train)
            if 'rgb0' in extras:
                loss += img2mse(extras['rgb0'], target_s)

        elif args.model_name in ['nerf_v3.2', 'R2L']:
            model = render_kwargs_train['network_fn']
            perturb = render_kwargs_train['perturb']
            if args.plucker:
                pts = point_sampler.sample_train_plucker(rays_o, rays_d)
            else:
                pts = point_sampler.sample_train(rays_o,
                                                 rays_d,
                                                 perturb=perturb)
            rgb = model(positional_embedder(pts))
        elif args.model_name in ['SilhouetteNeRF']:
            model = render_kwargs_train['network_fn']
            perturb = render_kwargs_train['perturb']
            if args.plucker:
                pts = point_sampler.sample_train_plucker(rays_o, rays_d)
            else:
                pts = point_sampler.sample_train(rays_o,
                                                 rays_d,
                                                 perturb=perturb)
            depth = model(positional_embedder(pts))

        if args.train_depth:
            # depth loss
            loss_depth = img2mse(depth[:, :1], target_s[:, :1]) * args.lw_rgb
            psnr = mse2psnr(loss_depth)
            loss_line.update('psnr', psnr.item(), '.4f')
            loss += loss_depth
        else:
            # rgb loss
            loss_rgb = img2mse(rgb[:, :3], target_s[:, :3]) * args.lw_rgb
            psnr = mse2psnr(loss_rgb)
            loss_line.update('psnr', psnr.item(), '.4f')
            loss += loss_rgb

        # smoothing for log print
        if not math.isinf(psnr.item()):
            hist_psnr = psnr.item(
            ) if i == start + 1 else hist_psnr * 0.95 + psnr.item() * 0.05
            loss_line.update('hist_psnr', hist_psnr, '.4f')

        # regress depth
        if args.learn_depth:
            loss_d = img2mse(rgb[:, 3:], target_s[:, 3:])
            loss += loss_d * args.lw_depth
            hist_depthloss = loss_d.item(
            ) if i == start + 1 else hist_depthloss * 0.95 + loss_d.item(
            ) * 0.05
            loss_line.update(f'hist_depthloss (*{args.lw_depth})',
                             hist_depthloss, '.4f')

        loss_line.update('LR', new_lrate, '.10f')

        # <<<<<<<<<<< inner loop (get data, forward, add loss) ends <<<<<<<<<<<

        # backward and update
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        batch_time.update(time.time() - t0)

        # collect hard examples
        if args.hard_ratio:
            if args.train_depth:
                _, indices = torch.sort(
                    torch.mean((depth[:batch_size] - target_s[:batch_size]) ** 2,
                               dim=1))
                hard_indices = indices[-n_hard_in:]
                hard_rays_ = torch.cat([
                    rays_o[hard_indices], rays_d[hard_indices],
                    target_s[hard_indices]
                ],
                    dim=-1)
                if hard_pool_full:
                    hard_rays[rand_ix_out[:n_hard_in]] = hard_rays_  # replace
                else:
                    hard_rays = torch.cat([hard_rays, hard_rays_], dim=0)  # append
                    if hard_rays.shape[0] >= batch_size * args.hard_mul:
                        hard_pool_full = True
            else:
                _, indices = torch.sort(
                    torch.mean((rgb[:batch_size] - target_s[:batch_size])**2,
                               dim=1))
                hard_indices = indices[-n_hard_in:]
                hard_rays_ = torch.cat([
                    rays_o[hard_indices], rays_d[hard_indices],
                    target_s[hard_indices]
                ],
                                       dim=-1)
                if hard_pool_full:
                    hard_rays[rand_ix_out[:n_hard_in]] = hard_rays_  # replace
                else:
                    hard_rays = torch.cat([hard_rays, hard_rays_], dim=0)  # append
                    if hard_rays.shape[0] >= batch_size * args.hard_mul:
                        hard_pool_full = True

        # print logs of training
        if i % args.i_print == 0:
            logstr = f"[TRAIN] Iter {i} data_time {data_time.val:.4f} ({data_time.avg:.4f}) batch_time {batch_time.val:.4f} ({batch_time.avg:.4f}) " + loss_line.format(
            )
            print(logstr)

            # save image for check
            if rgb1 is not None and i % (100 * args.i_print) == 0:
                for ix in range(min(5,
                                    rgb1.shape[0])):  # save 5 images at most
                    img = rgb1[ix].permute(1, 2, 0)  # to shape [h, w, 3]
                    save_path = f'{logger.gen_img_path}/rgb1_{ExpID}_iter{i}_img{ix}.png'
                    imageio.imwrite(save_path, to8b(img))

        # test: using the splitted test images
        if i % args.i_testset == 0:
            testsavedir = f'{logger.gen_img_path}/testset_{ExpID}_iter{i}'  # save the renderred test images
            os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                print(f'Iter {i} Testing...')
                t_ = time.time()
                *_, misc = render_path(test_poses,
                                       hwf,
                                       args.chunk,
                                       render_kwargs_test,
                                       gt_imgs=test_images,
                                       savedir=testsavedir,
                                       render_factor=args.render_factor)
                t_test = time.time() - t_

            # save the best model
            if misc['test_psnr_v2'] > best_psnr:
                best_psnr = misc['test_psnr_v2'].item()
                best_psnr_step = i
                path = save_ckpt(f'ckpt_best.tar', render_kwargs_train,
                                 optimizer, best_psnr, best_psnr_step)
                print(f'Iter {i} Save the best checkpoint: "{path}".')

            accprint(
                f"[TEST] Iter {i} TestPSNR {misc['test_psnr'].item():.4f} TestPSNRv2 {misc['test_psnr_v2'].item():.4f} \
BestPSNRv2 {best_psnr:.4f} (Iter {best_psnr_step}) \
TestSSIM {misc['test_ssim'].item():.4f} TestLPIPS {misc['test_lpips'].item():.4f} TestFLIP {misc['test_flip'].item():.4f} \
TrainHistPSNR {hist_psnr:.4f} LR {new_lrate:.8f} Time {t_test:.1f}s")
            print(f'Saved rendered test images: "{testsavedir}"')
            print(f'Predicted finish time: {timer()}')

        # test: using novel poses
        if i % args.i_video == 0:
            with torch.no_grad():
                print(
                    f'Iter {i} Rendering video... (n_pose: {len(video_poses)})'
                )
                t_ = time.time()
                rgbs, disps, misc = render_path(
                    video_poses,
                    hwf,
                    args.chunk,
                    render_kwargs_test,
                    gt_imgs=video_targets,
                    render_factor=args.render_factor)
                t_video = time.time() - t_
            video_path = f'{logger.gen_img_path}/video_{ExpID}_iter{i}_{args.video_tag}.mp4'
            imageio.mimwrite(video_path, to8b(rgbs), fps=30, quality=8)
            if args.model_name in [
                    'nerf'
            ]:  # @mst: raybased nerf does not predict depth right now
                imageio.mimwrite(disp_path,
                                 to8b(disps / np.max(disps)),
                                 fps=30,
                                 quality=8)
            print(
                f'[VIDEO] Rendering done. Time {t_video:.2f}s. Save video: "{video_path}"'
            )

            if video_targets is not None:  # given ground truth, psnr will be calculated, deprecated, will be removed
                print(f"[VIDEO] video_psnr {misc['test_psnr'].item():.4f}")
                imageio.mimwrite(video_path.replace('.mp4', '_error.mp4'),
                                 to8b(errors),
                                 fps=30,
                                 quality=8)

        # save checkpoint
        if i % args.i_weights == 0:
            ckpt_name = f'ckpt_{i}.tar' if args.save_intermediate_models else 'ckpt.tar'
            path = save_ckpt(ckpt_name, render_kwargs_train, optimizer,
                             best_psnr, best_psnr_step)
            print(f'Iter {i} Save checkpoint: "{path}".')


def save_ckpt(file_name, render_kwargs_train, optimizer, best_psnr,
              best_psnr_step):
    path = os.path.join(logger.weights_path, file_name)
    to_save = {
        'global_step':
        global_step,
        'best_psnr':
        best_psnr,
        'best_psnr_step':
        best_psnr_step,
        'network_fn_state_dict':
        undataparallel(render_kwargs_train['network_fn'].state_dict()),
        'optimizer_state_dict':
        optimizer.state_dict(),
    }
    if args.model_name in ['nerf'] and args.N_importance > 0:
        to_save['network_fine_state_dict'] = undataparallel(
            render_kwargs_train['network_fine'].state_dict())
    if args.model_name in ['nerf_v3.2', 'R2L', 'SilhouetteNeRF']:
        to_save['network_fn'] = undataparallel(
            render_kwargs_train['network_fn'])
    torch.save(to_save, path)
    # # convert to onnx
    # onnx_path = path.replace('.tar', '.onnx')
    # save_onnx(model, onnx_path)
    # save_log += f', onnx saved at "{onnx_path}"'
    return path


if __name__ == '__main__':
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
