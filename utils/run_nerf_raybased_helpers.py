import copy, os, time, math
from collections import OrderedDict

import numpy as np

import torch

torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F

# Misc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
to_tensor = lambda x: x.to(device) if isinstance(
    x, torch.Tensor) else torch.Tensor(x).to(device)
to_array = lambda x: x if isinstance(x, np.ndarray) else x.data.cpu().numpy()
to_list = lambda x: x if isinstance(x, list) else to_array(x).tolist()
to8b = lambda x: (255 * np.clip(to_array(x), 0, 1)).astype(np.uint8)
img2mse = lambda x, y: torch.mean((x - y)**2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(to_tensor([10.]))


# Positional encoding (section 5.1)
class Embedder:

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(
                    lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim  # @mst: (1) for x: 63 = (2x10+1)x3. 10 from the paper. 1 because of the 'include_input = True' (2) for view, 27 = (2x4+1)x3

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


def raw2outputs(raw,
                z_vals,
                rays_d,
                raw_noise_std=0,
                white_bkgd=False,
                pytest=False,
                global_step=-1,
                print=print):
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
        [dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)],
        -1)  # [N_rays, N_samples]
    # @mst: 1e10 for infinite distance

    dists = dists * torch.norm(
        rays_d[..., None, :],
        dim=-1)  # @mst: direction vector needs normalization. why this * ?

    rgb = torch.sigmoid(
        raw[..., :3])  # [N_rays, N_samples, 3], RGB for each sampled point
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]

    # print to check alpha
    if global_step % 100 == 0:
        for i_ray in range(0, alpha.shape[0], 100):
            logtmp = ['%.4f' % x for x in alpha[i_ray]]
            print('%4d: ' % i_ray + ' '.join(logtmp))

    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1),
        -1)[:, :-1]  # @mst: [N_rays, N_samples]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map),
                              depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


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


def translate_origin_v2(rays_o, rays_d):
    rays_o_new = []
    rays_o_norm = 3.6
    for ro, rd in zip(rays_o, rays_d):
        rd_ = rd / rd.norm()
        # m^2 + d^2 -2mcos(theta) * d = n^2
        # d^2 - 2mcos(theta) * d + m^2 - n^2 = 0
        m, n = ro.norm(), rays_o_norm
        cos_theta = -(ro * rd_).sum() / m
        d1 = m * cos_theta + torch.sqrt(m**2 * cos_theta**2 - m**2 + n**2)
        d2 = m * cos_theta - torch.sqrt(m**2 * cos_theta**2 - m**2 + n**2)
        d = max(d1, d2) if d1 * d2 < 0 else d1.sign() * min(d1.abs(), d2.abs())
        ro_ = ro + d * rd_
        rays_o_new += [ro_]
    return torch.stack(rays_o_new, dim=0)


def translate_origin(rays_o, rays_d):
    rays_o_new = []
    rays_o_norm = 3.6
    ro, rd = rays_o[0], rays_d[0]
    rd_ = rd / rd.norm()
    for d in range(100):  # 100 is hand-chosen
        if (ro + d * rd_).norm() <= rays_o_norm:
            break
    return rays_o + d * (rays_d / rays_d.norm(dim=1))


def translate_origin_fixed(rays_o, rays_d, scale, n_print=0):
    '''hand-tuned for blender dataset.
    '''
    rd = rays_d / rays_d.norm(dim=-1, keepdim=True)  # [H, W, 1]
    ro = rays_o + scale * rd

    if n_print > 0:
        k = int(math.sqrt(n_print))
        rand_h, rand_w = np.random.permutation(
            ro.shape[0])[:k], np.random.permutation(ro.shape[1])[:k]
        for h in rand_h:
            for w in rand_w:
                print(f'{(h, w)} rays_o norm: {ro[h, w].norm().item():.4f}')
    return ro


# Ray helpers
def get_rays(H, W, focal, c2w, trans_origin='', focal_scale=1):
    focal *= focal_scale
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                          torch.linspace(
                              0, H - 1,
                              H))  # pytorch's meshgrid has indexing='ij'
    i, j = i.t().to(device), j.t().to(device)
    dirs = torch.stack(
        [(i - W * .5) / focal, -(j - H * .5) / focal, -torch.ones_like(i)],
        -1)  # TODO-@mst: check if this H/W or W/H is in the right order
    # Rotate ray directions from camera frame to the world frame
    # rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs.unsqueeze(dim=-2) * c2w[:3, :3],
                       -1)  # shape: [H, W, 3]

    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    if trans_origin:
        if trans_origin == 'adapative':
            rays_o = translate_origin_adapative(rays_o, rays_d)
        else:
            scale = 30 if trans_origin == 'fixed' else float(trans_origin)
            rays_o = translate_origin_fixed(rays_o,
                                            rays_d,
                                            scale=scale,
                                            n_print=10)
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] -
                                     rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] -
                                     rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf],
                    -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    # inds = searchsorted(cdf, u, side='right')
    inds = torch.searchsorted(
        cdf, u, right=True
    )  # See issue: https://github.com/yenchenlin/nerf-pytorch/pull/35
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def parse_expid_iter(path):
    r"""Parse out experiment id and iteration for pretrained ckpt.
    path example: Experiments/nerfv2__lego__S4W1024D32Skip8,16,24_DPRGB_KDWRenderPose100All_BS16384_SERVER142-20210704-150540/weights/200000.tar
    then, expid is 'SERVER142-20210704-150540'; iter_ is '200000'
    """
    if 'SERVER' in path:
        expid = 'SERVER' + path.split('_SERVER')[1].split('/')[0]
        iter_ = path.split('/')[-1].split('.tar')[0]
    else:
        expid = 'Unknown'
        iter_ = 'Unknown'
    return expid, iter_


def load_weights(model, ckpt_path, key):
    from smilelogging.utils import check_path
    from collections import OrderedDict
    ckpt_path = check_path(ckpt_path)
    ckpt = torch.load(ckpt_path)
    state_dict = ckpt[key]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            k = k[7:]
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    return ckpt_path, ckpt


def load_weights_v2(model, ckpt, key):
    from collections import OrderedDict

    model_dataparallel = False
    for name, module in model.named_modules():
        if name.startswith('module.'):
            model_dataparallel = True
            break

    state_dict = ckpt[key]
    weights_dataparallel = False
    for k, v in state_dict.items():
        if k.startswith('module.'):
            weights_dataparallel = True
            break

    if model_dataparallel and weights_dataparallel or (
            not model_dataparallel) and (not weights_dataparallel):
        model.load_state_dict(state_dict)
    else:
        raise NotImplementedError


def get_selected_coords(coords, N_rand, mode):
    coords = coords.long()  # [H, W, 2]
    H, W = coords.shape[:2]
    if mode == 'rand_pixel':
        rand_ix = np.random.choice(H * W, size=[N_rand], replace=False)
        coords = coords.view(-1, 2)  # [H*W, 2]
        selected_coords = coords[rand_ix]  # [N_rand, 2]
        return selected_coords, None  # None is placeholder for patch indices

    elif mode == 'rand_patch':
        k = math.sqrt(float(N_rand) / H / W)
        patch_h, patch_w = int(H * k), int(W * k)
        bbh1 = np.random.randint(0, H - patch_h)
        bbw1 = np.random.randint(0, W - patch_w)
        bbh2 = bbh1 + patch_h
        bbw2 = bbw1 + patch_w
        selected_coords = coords[bbh1:bbh2,
                                 bbw1:bbw2, :]  # [patch_h, patch_w, 2]
        selected_coords = selected_coords.reshape([-1,
                                                   2])  # [patch_h*patch_w, 2]
        return selected_coords, [bbh1, bbw1, bbh2, bbw2]


def undataparallel(input):
    '''remove the module. prefix caused by nn.DataParallel'''
    if isinstance(input, nn.Module):
        model = input
        if hasattr(model, 'module'):
            model = model.module
        return model
    elif isinstance(input, OrderedDict):
        state_dict = input
        w = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                assert len(k.split('module.')) == 2
                k = k.split('module.')[1]
            w[k] = v
        return w
    else:
        raise NotImplementedError


def get_rays_np(H, W, focal, c2w):
    c2w, focal = to_array(c2w), to_array(focal)
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    dirs = np.stack(
        [(i - W * .5) / focal, -(j - H * .5) / focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(
        dirs[..., np.newaxis, :] * c2w[:3, :3],
        -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def visualize_3d(xyzs,
                 savepath,
                 cmaps,
                 connect=False,
                 save_pickle=True,
                 lim=None):
    import matplotlib.pyplot as plt
    # plt.style.use(['science']) # 'ieee'
    plt.rcParams["font.family"] = "Times New Roman"
    label_fs, ticklabelsize = 14, 9
    from mpl_toolkits import mplot3d
    import pickle
    fig = plt.figure()
    ax3d = plt.axes(projection='3d')
    for ix, item in enumerate(xyzs):
        x, y, z = item
        ax3d.scatter3D(x, y, z, cmap=cmaps[ix])
        if connect:
            ax3d.plot3D(x, y, z)
    ax3d.scatter3D(0, 0, 0, marker='d', color='red')
    if lim is not None:
        ax3d.set_xlim(lim)
        ax3d.set_ylim(lim)
        ax3d.set_zlim(lim)
    ax3d.set_xlabel('X axis', fontsize=label_fs)
    ax3d.set_ylabel('Y axis', fontsize=label_fs)
    ax3d.set_zlabel('Z axis', fontsize=label_fs)
    ax3d.tick_params(axis='both', labelsize=ticklabelsize)
    if save_pickle:
        pickle_savepath = os.path.splitext(savepath)[0] + '.fig.pickle'
        pickle.dump(fig, open(pickle_savepath, 'wb'))
    ax3d.grid(True, linestyle='dotted')
    fig.savefig(savepath, bbox_inches='tight')
