from os import getgroups
import torch

torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np, time, math

THRESHOLD = 0.1
sphere_radius = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Misc
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


class PointSampler():

    def __init__(self, H, W, focal, n_sample, near, far):
        self.H, self.W = H, W
        i, j = torch.meshgrid(
            torch.linspace(0, W - 1, W).to(device),
            torch.linspace(0, H - 1, H).to(device))
        i, j = i.t(), j.t()
        self.dirs = torch.stack(
            [(i - W * .5) / focal, -(j - H * .5) / focal, -torch.ones_like(i)],
            dim=-1).to(device)  # [H, W, 3]

        t_vals = torch.linspace(0., 1.,
                                steps=n_sample).to(device)  # [n_sample]
        self.z_vals = near * (1 - t_vals) + far * (t_vals)  # [n_sample]
        self.z_vals_test = self.z_vals[None, :].expand(
            H * W, n_sample)  # [H*W, n_sample]

    def sample_test(self, c2w):  # c2w: [3, 4]
        origin = c2w[:3, -1]
        distance = torch.sqrt(origin[0] ** 2 + origin[1] ** 2 + origin[2] ** 2)
        rays_d = torch.sum(
            self.dirs.unsqueeze(dim=-2) * c2w[:3, :3], dim=-1).view(
                -1,
                3)  # [H*W, 3] # TODO-@mst: improve this non-intuitive impl.
        rays_o = c2w[:3, -1].expand(rays_d.shape)  # [H*W, 3]
        offsets = torch.zeros((list(rays_d.shape)[0], 1))
        # If the camera position is outside of the bounding sphere, reproject the ray origins
        # Ray-sphere intersection reference: http://kylehalladay.com/blog/tutorial/math/2013/12/24/Ray-Sphere-Intersection.html
        if torch.max(distance) > sphere_radius + THRESHOLD:
            # Calculate the direction vector normalized for all rays
            rays_d_normalized = rays_d / torch.norm(rays_d, dim=1, keepdim=True)
            center = torch.tensor([0, 0, 0], dtype=torch.float32).to("cuda:0")
            vec_to_center = torch.sub(center, origin)
            L_sqrd = torch.dot(vec_to_center, vec_to_center)
            # torch.sum is used to calculate the element-wise dot product
            tc = torch.sum(vec_to_center * rays_d_normalized, dim=1)
            d = torch.sqrt(L_sqrd - torch.pow(tc, 2))
            th = torch.sqrt(sphere_radius ** 2 - torch.pow(d, 2))
            t = tc - th

            intersection_points = rays_o + t.unsqueeze(1) * rays_d_normalized
            offsets = (rays_o - intersection_points).pow(2).sum(1).sqrt()
            rays_o = intersection_points

        # Sample points along ray
        pts = rays_o[..., None, :] + rays_d[..., None, :] * self.z_vals_test[
            ..., :, None]  # [H*W, n_sample, 3]
        return pts.view(pts.shape[0], -1), offsets  # [H*W, n_sample*3]

    def sample_test2(self, c2w):  # c2w: [3, 4]
        rays_d = torch.sum(
            self.dirs.unsqueeze(dim=-2) * c2w[:3, :3], dim=-1).view(
                -1,
                3)  # [H*W, 3] # TODO-@mst: improve this non-intuitive impl.
        rays_o = c2w[:3, -1].expand(rays_d.shape)  # [H*W, 3]
        pts = rays_o[..., None, :] + rays_d[..., None, :] * self.z_vals_test[
            ..., :, None]  # [H*W, n_sample, 3]
        return pts  # [..., n_sample, 3]

    def sample_train(self, rays_o, rays_d, perturb):
        z_vals = self.z_vals[None, :].expand(
            rays_o.shape[0], self.z_vals.shape[0])  # depth [n_ray, n_sample]
        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1], mids], dim=-1)
            t_rand = torch.rand(z_vals.shape).to(device)  # [n_ray, n_sample]
            z_vals = lower + (upper - lower) * t_rand
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[
            ..., :, None]  # [n_ray, n_sample, DIM_DIR]
        return pts.view(pts.shape[0], -1)  # [n_ray, n_sample * DIM_DIR]

    def sample_train2(self, rays_o, rays_d, perturb):
        '''rays_o: [n_img, patch_h, patch_w, 3] for CNN-style. Keep this for back-compatibility, please use sample_train_cnnstyle'''
        z_vals = self.z_vals[None, None, None, :].expand(
            *rays_o.shape[:3],
            self.z_vals.shape[0])  # [n_img, patch_h, patch_w, n_sample]
        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1]
                         )  # [n_img, patch_h, patch_w, n_sample]
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1], mids], dim=-1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape[0]).to(device)  # [n_img]
            t_rand = t_rand[:, None, None, None].expand_as(
                z_vals)  # [n_img, patch_h, patch_w, n_sample]
            z_vals = lower + (
                upper - lower) * t_rand  # [n_img, patch_h, patch_w, n_sample]
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[
            ..., :, None]  # [n_img, patch_h, patch_w, n_sample, 3]
        return pts

    def sample_train_cnnstyle(self, rays_o, rays_d, perturb):
        '''rays_o and rayd_d: [n_patch, 3, patch_h, patch_w]'''
        z_vals = self.z_vals[None, None, None, :].expand(
            *rays_o.shape[:3],
            self.z_vals.shape[0])  # [n_img, patch_h, patch_w, n_sample]
        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1]
                         )  # [n_img, patch_h, patch_w, n_sample]
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1], mids], dim=-1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape[0]).to(device)  # [n_img]
            t_rand = t_rand[:, None, None, None].expand_as(
                z_vals)  # [n_img, patch_h, patch_w, n_sample]
            z_vals = lower + (
                upper - lower) * t_rand  # [n_img, patch_h, patch_w, n_sample]
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[
            ..., :, None]  # [n_img, patch_h, patch_w, n_sample, 3]
        return pts

    def sample_train_plucker(self, rays_o, rays_d):
        r"""Use Plucker coordinates as ray representation.
        Refer to: https://faculty.sites.iastate.edu/jia/files/inline-files/plucker-coordinates.pdf
        """
        m = torch.cross(rays_o, rays_d, dim=-1)  # [n_ray, 3]
        pts = torch.cat([rays_d, m], dim=-1)  # [n_ray, 6]
        return pts

    def sample_test_plucker(self, c2w):  # c2w: [3, 4]
        r"""Use Plucker coordinates as ray representation.
        """
        rays_d = torch.sum(
            self.dirs.unsqueeze(dim=-2) * c2w[:3, :3], dim=-1).view(
                -1,
                3)  # [H*W, 3] # TODO-@mst: improve this non-intuitive impl.
        rays_o = c2w[:3, -1].expand(rays_d.shape)  # [H*W, 3]
        m = torch.cross(rays_o, rays_d, dim=-1)  # [H*W, 3]
        pts = torch.cat([rays_d, m], dim=-1)  # [H*W, 6]
        return pts


class PositionalEmbedder():

    def __init__(self, L, include_input=True):
        self.weights = 2**torch.linspace(0, L - 1, steps=L).to(device)  # [L]
        self.include_input = include_input
        self.embed_dim = 2 * L + 1 if include_input else 2 * L

    def __call__(self, x):
        y = x[
            ...,
            None] * self.weights  # [n_ray, dim_pts, 1] * [L] -> [n_ray, dim_pts, L]
        y = torch.cat([torch.sin(y), torch.cos(y)],
                      dim=-1)  # [n_ray, dim_pts, 2L]
        if self.include_input:
            y = torch.cat([y, x.unsqueeze(dim=-1)],
                          dim=-1)  # [n_ray, dim_pts, 2L+1]
        return y.view(y.shape[0],
                      -1)  # [n_ray, dim_pts*(2L+1)], example: 48*21=1008

    def embed(self, x):
        ''''for CNN-style. Keep this for back-compatibility, please use embed_cnnstyle'''
        y = x[..., :, None] * self.weights
        y = torch.cat([torch.sin(y), torch.cos(y)], dim=-1)
        if self.include_input:
            y = torch.cat([y, x.unsqueeze(dim=-1)], dim=-1)
        return y  # [n_img, patch_h, patch_w, n_sample, 3, 2L+1]

    def embed_cnnstyle(self, x):
        y = x[..., :, None] * self.weights
        y = torch.cat([torch.sin(y), torch.cos(y)], dim=-1)
        if self.include_input:
            y = torch.cat([y, x.unsqueeze(dim=-1)], dim=-1)
        return y  # [n_img, patch_h, patch_w, n_sample, 3, 2L+1]


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
        [dists,
         torch.Tensor([1e10]).to(device).expand(dists[..., :1].shape)],
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


class NeRF(nn.Module):

    def __init__(self,
                 D=8,
                 W=256,
                 input_ch=3,
                 input_ch_views=3,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList([nn.Linear(input_ch, W)] + [
            nn.Linear(W, W) if i not in
            self.skips else nn.Linear(W + input_ch, W) for i in range(D - 1)
        ])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList(
            [nn.Linear(input_ch_views + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(
            x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(
                np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(
                np.transpose(weights[idx_pts_linears + 1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_feature_linear + 1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(
            np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(
            np.transpose(weights[idx_views_linears + 1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_rbg_linear + 1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_alpha_linear + 1]))


class ResMLP(nn.Module):

    def __init__(self,
                 width,
                 inact=nn.ReLU(True),
                 outact=None,
                 res_scale=1,
                 n_learnable=2):
        '''inact is the activation func within block. outact is the activation func right before output'''
        super(ResMLP, self).__init__()
        m = [nn.Linear(width, width)]
        for _ in range(n_learnable - 1):
            if inact is not None: m += [inact]
            m += [nn.Linear(width, width)]
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
        self.outact = outact

    def forward(self, x):
        x = self.body(x).mul(self.res_scale) + x
        if self.outact is not None:
            x = self.outact(x)
        return x


def get_activation(act):
    if act.lower() == 'relu':
        func = nn.ReLU(inplace=True)
    elif act.lower() == 'lrelu':
        func = nn.LeakyReLU(inplace=True)
    elif act.lower() == 'none':
        func = None
    else:
        raise NotImplementedError
    return func


class NeRF_v3_2(nn.Module):
    '''Based on NeRF_v3, move positional embedding out'''

    def __init__(self, args, input_dim, output_dim):
        super(NeRF_v3_2, self).__init__()
        self.args = args
        D, W = args.netdepth, args.netwidth

        # get network width
        if args.layerwise_netwidths:
            Ws = [int(x) for x in args.layerwise_netwidths.split(',')] + [3]
            print('Layer-wise widths are given. Overwrite args.netwidth')
        else:
            Ws = [W] * (D - 1) + [3]

        # the main non-linear activation func
        act = get_activation(args.act)

        # head
        self.input_dim = input_dim
        self.head = nn.Sequential(*[nn.Linear(input_dim, Ws[0]), act])

        # body
        body = []
        for i in range(1, D - 1):
            body += [nn.Linear(Ws[i - 1], Ws[i]), act]

        # >>> new implementation of the body. Will replace the above
        if hasattr(args, 'trial'):
            inact = get_activation(args.trial.inact)
            outact = get_activation(args.trial.outact)
            if args.trial.body_arch in ['resmlp']:
                n_block = (
                    D - 2
                ) // 2  # 2 layers in a ResMLP, deprecated since there can be >2 layers in a block, use --trial.n_block
                if args.trial.n_block > 0:
                    n_block = args.trial.n_block
                body = [
                    ResMLP(W,
                           inact=inact,
                           outact=outact,
                           res_scale=args.trial.res_scale,
                           n_learnable=args.trial.n_learnable)
                    for _ in range(n_block)
                ]
            elif args.trial.body_arch in ['mlp']:
                body = []
                for i in range(1, D - 1):
                    body += [nn.Linear(Ws[i - 1], Ws[i]), act]
        # <<<

        self.body = nn.Sequential(*body)

        # tail
        self.tail = nn.Linear(
            W, output_dim) if args.linear_tail else nn.Sequential(
                *[nn.Linear(Ws[D - 2], output_dim),
                  nn.Sigmoid()])

    def forward(self, x):  # x: embedded position coordinates
        if x.shape[-1] != self.input_dim:  # [N, C, H, W]
            x = x.permute(0, 2, 3, 1)
        x = self.head(x)
        x = self.body(x) + x if self.args.use_residual else self.body(x)
        return self.tail(x)
