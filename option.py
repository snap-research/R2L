from random import choice
from numpy.random import default_rng
from smilelogging.utils import check_path, strdict_to_dict, update_args
from smilelogging import argparser as parser

parser.add_argument('--config', is_config_file=True, help='config file path')
parser.add_argument("--expname", type=str, help='experiment name')
parser.add_argument("--basedir",
                    type=str,
                    default='./logs/',
                    help='where to store ckpts and logs')
parser.add_argument("--datadir",
                    type=str,
                    default='./data/llff/fern',
                    help='input data directory')

# training options
parser.add_argument("--netdepth",
                    type=int,
                    default=8,
                    help='layers in network')
parser.add_argument("--netwidth",
                    type=int,
                    default=256,
                    help='channels per layer')
parser.add_argument("--netdepth_fine",
                    type=int,
                    default=8,
                    help='layers in fine network')
parser.add_argument("--netwidth_fine",
                    type=int,
                    default=256,
                    help='channels per layer in fine network')
parser.add_argument(
    "--N_rand",
    type=int,
    default=32 * 32 * 4,
    help='batch size (number of random rays per gradient step)')
parser.add_argument("--lrate", type=float, default=5e-4, help='learning rate')
parser.add_argument("--lrate_decay",
                    type=int,
                    default=250,
                    help='exponential learning rate decay (in 1000 steps)')
parser.add_argument(
    "--chunk",
    type=int,
    default=1024 * 32,
    help=
    'number of rays processed in parallel, decrease if running out of memory')
parser.add_argument(
    "--netchunk",
    type=int,
    default=1024 * 64,
    help=
    'number of pts sent through network in parallel, decrease if running out of memory'
)
parser.add_argument("--no_batching",
                    action='store_true',
                    help='only take random rays from 1 image at a time')
parser.add_argument("--no_reload",
                    action='store_true',
                    help='do not reload weights from saved ckpt')
parser.add_argument(
    "--ft_path",
    type=str,
    default=None,
    help='specific weights npy file to reload for coarse network')

# rendering options
parser.add_argument("--N_samples",
                    type=int,
                    default=64,
                    help='number of coarse samples per ray')
parser.add_argument("--N_importance",
                    type=int,
                    default=0,
                    help='number of additional fine samples per ray')
parser.add_argument("--perturb",
                    type=float,
                    default=1.,
                    help='set to 0. for no jitter, 1. for jitter')
parser.add_argument("--perturb_test",
                    type=float,
                    default=0,
                    help='set to 0. for no jitter, 1. for jitter')
parser.add_argument("--use_viewdirs",
                    action='store_true',
                    help='use full 5D input instead of 3D')
parser.add_argument("--i_embed",
                    type=int,
                    default=0,
                    help='set 0 for default positional encoding, -1 for none')
parser.add_argument(
    "--multires",
    type=int,
    default=10,
    help='log2 of max freq for positional encoding (3D location)')
parser.add_argument(
    "--multires_views",
    type=int,
    default=4,
    help='log2 of max freq for positional encoding (2D direction)')
parser.add_argument(
    "--raw_noise_std",
    type=float,
    default=0.,
    help='std dev of noise added to regularize sigma_a output, 1e0 recommended'
)

parser.add_argument(
    "--render_only",
    action='store_true',
    help='do not optimize, reload weights and render out render_poses path')
parser.add_argument("--render_test",
                    action='store_true',
                    help='render the test set instead of render_poses path')
parser.add_argument(
    "--render_factor",
    type=float,
    default=0,
    help=
    'downsampling factor to speed up rendering, set 4 or 8 for fast preview')

parser.add_argument(
    "--scaling_factor",
    type=float,
    default=1,
    help=
    'factor to scale disparity outputs, set 1 for true disparity')

# training options
parser.add_argument("--precrop_iters",
                    type=int,
                    default=0,
                    help='number of steps to train on central crops')
parser.add_argument("--precrop_frac",
                    type=float,
                    default=.5,
                    help='fraction of img taken for central crops')

# dataset options
parser.add_argument("--dataset_type",
                    type=str,
                    default='llff',
                    help='options: llff / blender / deepvoxels')
parser.add_argument(
    "--testskip",
    type=int,
    default=8,
    help=
    'will load 1/N images from test/val sets, useful for large datasets like deepvoxels'
)

# deepvoxels flags
parser.add_argument("--shape",
                    type=str,
                    default='greek',
                    help='options : armchair / cube / greek / vase')

# blender flags
parser.add_argument(
    "--white_bkgd",
    action='store_true',
    help='set to render synthetic data on a white bkgd (always use for dvoxels)'
)
parser.add_argument(
    "--half_res",
    action='store_true',
    help='load blender synthetic data at 400x400 instead of 800x800')

# llff flags
parser.add_argument("--factor",
                    type=int,
                    default=8,
                    help='downsample factor for LLFF images')
parser.add_argument(
    "--no_ndc",
    action='store_true',
    help=
    'do not use normalized device coordinates (set for non-forward facing scenes)'
)
parser.add_argument("--lindisp",
                    action='store_true',
                    help='sampling linearly in disparity rather than depth')
parser.add_argument("--spherify",
                    action='store_true',
                    help='set for spherical 360 scenes')
parser.add_argument(
    "--llffhold",
    type=int,
    default=8,
    help='will take every 1/N images as LLFF test set, paper uses 8')

# logging/saving options
parser.add_argument("--i_print",
                    type=int,
                    default=100,
                    help='frequency of console printout and metric loggin')
parser.add_argument("--i_img",
                    type=int,
                    default=500,
                    help='frequency of tensorboard image logging')
parser.add_argument("--i_weights",
                    type=int,
                    default=10000,
                    help='frequency of weight ckpt saving')
parser.add_argument("--i_testset",
                    type=int,
                    default=1000,
                    help='frequency of testset saving')
parser.add_argument("--i_video",
                    type=int,
                    default=10000,
                    help='frequency of render_poses video saving')

# R2L related
parser.add_argument('--model_name',
                    type=str,
                    default='R2L',
                    choices=['nerf', 'nerf_v3.2', 'R2L', 'DeLFT'])
parser.add_argument('--N_iters', type=int, default=200000)
parser.add_argument('--skips', type=str, default='4')
parser.add_argument('--D_head', type=int, default=4)
parser.add_argument('--n_sample_per_ray', type=int, default=192)
parser.add_argument('--encode_input', action='store_true')
parser.add_argument('--pretrained_ckpt', type=str, default='')
parser.add_argument('--test_pretrained', action="store_true")
parser.add_argument('--resume',
                    action="store_true",
                    help='if True, resume the optimizer')
parser.add_argument('--lw_kd', type=float, default=0.001)
parser.add_argument('--split_layer', type=int, default=-1)
parser.add_argument('--dropout_layer', type=str, default='')
parser.add_argument('--dropout_ratio', type=float, default=0.5)
parser.add_argument('--n_pose_video',
                    type=str,
                    default='40',
                    help='num of poses in rendering the video')
parser.add_argument('--n_pose_kd',
                    type=str,
                    default='100',
                    help='num of poses in rendering the video when using KD')
parser.add_argument('--video_tag', type=str, default='')
parser.add_argument('--video_poses_perturb', action="store_true")
parser.add_argument('--datadir_kd', type=str, default='')
parser.add_argument('--create_data_chunk', type=int, default=100)
parser.add_argument('--create_data', type=str, default='spiral_evenly_spaced')
parser.add_argument('--no_rand_focal',
                    dest='use_rand_focal',
                    action='store_false',
                    default=True,
                    help='use random focal when creating data')
parser.add_argument('--max_save', type=int, default=40000)
parser.add_argument(
    '--i_update_data',
    type=int,
    default=1000000000,
    help='interval of updating training data (changing pseudo data)')
parser.add_argument('--pseudo_ratio', type=float, default=-1.)
parser.add_argument('--trans_origin', type=str, default='')
parser.add_argument('--select_pixel_mode',
                    type=str,
                    default='rand_pixel',
                    choices=['rand_pixel', 'rand_patch'])
parser.add_argument('--freeze_pretrained', action='store_true')
parser.add_argument('--focal_scale', type=float, default=1.)
parser.add_argument(
    '--data_mode',
    type=str,
    default='images',
    choices=['images', 'rays'],
    help=
    'which data is used in training, sample rays from images or directly load rays'
)
parser.add_argument('--rm_existing_data',
                    action='store_true',
                    help='remove existing data when creating data')
parser.add_argument('--num_workers',
                    type=int,
                    default=8,
                    help='#cpus when loading data')
parser.add_argument('--hard_ratio',
                    type=str,
                    default='',
                    help='hard rays ratio in a batch; seperated by comma')
parser.add_argument('--hard_mul',
                    type=float,
                    default=1,
                    help='hard_mul * batch_size is the size of hard ray pool')
parser.add_argument('--use_residual', action='store_true')
parser.add_argument('--linear_tail', action='store_true')
parser.add_argument('--layerwise_netwidths', type=str, default='')
parser.add_argument('--layerwise_netwidths2', type=str, default='')
parser.add_argument('--render_iters',
                    type=int,
                    default=1,
                    help='the number of forwards when rendering one image')
parser.add_argument('--convert_to_onnx', action='store_true')
parser.add_argument(
    '--benchmark',
    action='store_true',
    help=
    'check inference speed (time of rendering a frame) with a trained model')
parser.add_argument('--use_bn', action='store_true')
parser.add_argument('--shuffle_input', action='store_true')
parser.add_argument('--kernel_size', type=int, default=1)
parser.add_argument('--padding', type=int, default=0)
parser.add_argument('--body_arch',
                    type=str,
                    default='conv',
                    choices=['conv', 'resblock'])
parser.add_argument('--lw_rgb', type=float, default=1)
parser.add_argument('--lw_rgb1', type=float, default=1)
parser.add_argument('--act',
                    type=str,
                    default='relu',
                    choices=['relu', 'lrelu'],
                    help='main activation func in a network')
parser.add_argument('--warmup_lr', type=str, default='')
parser.add_argument('--lpips_net', type=str, default='alex')
parser.add_argument(
    '--pseudo_data_hold_ratio',
    type=float,
    default=0,
    help=
    'hold a part of pseudo data, not used for training, to control sample size'
)
parser.add_argument('--given_render_path_rays', type=str, default='')
parser.add_argument('--learn_depth',
                    type=str,
                    default='',
                    choices=['depth', 'surface'])
parser.add_argument('--save_depth',action='store_true')
parser.add_argument('--does_terminate',action='store_true')
parser.add_argument('--train_depth',action='store_true')
parser.add_argument('--lw_depth', type=float, default=0.1)
parser.add_argument('--save_intermediate_models', action='store_true')
parser.add_argument('--plucker', action='store_true')

# Create data
parser.add_argument('--teacher_ckpt', type=str)
parser.add_argument('--test_teacher', action='store_true')

# Try related features
parser.add_argument('--trial.ON', action='store_true')
parser.add_argument('--trial.body_arch',
                    type=str,
                    default='mlp',
                    choices=['mlp', 'resmlp'])
parser.add_argument('--trial.res_scale', type=float, default=1.)
parser.add_argument('--trial.n_learnable',
                    type=int,
                    default=2,
                    help='num of learnable layers')
parser.add_argument('--trial.inact',
                    default='relu',
                    choices=['none', 'relu', 'lrelu'],
                    help='the within activation func in a block')
parser.add_argument('--trial.outact',
                    default='none',
                    choices=['none', 'relu', 'lrelu'],
                    help='the output activation func in a block')
parser.add_argument('--trial.n_block',
                    type=int,
                    default=-1,
                    help='num of block in network body')
parser.add_argument('--trial.near', type=float, default=-1)
parser.add_argument('--trial.far', type=float, default=-1)

args = parser.parse_args()

if args.video_tag == '':
    args.video_tag = f'pose{args.n_pose_video}'


def check_n_pose(n_pose):
    if n_pose.lower() == 'none':
        return None
    if n_pose.isdigit():
        return int(n_pose)
    else:
        return n_pose.split(',')


args.n_pose_kd = check_n_pose(args.n_pose_kd)
args.n_pose_video = check_n_pose(args.n_pose_video)
args.pretrained_ckpt = check_path(args.pretrained_ckpt)

if args.hard_ratio != '':
    if ',' not in args.hard_ratio:
        args.hard_ratio = float(args.hard_ratio)
    else:
        args.hard_ratio = [float(x) for x in args.hard_ratio.split(',')]

# easier for controlling irrelevant args
args = update_args(args)
