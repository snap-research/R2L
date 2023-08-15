# R2L: Distilling NeRF to NeLF

### [Project](https://snap-research.github.io/R2L/) | [ArXiv](https://arxiv.org/abs/2203.17261) | [PDF](https://arxiv.org/pdf/2203.17261.pdf) 

<div align="center">
    <a><img src="figs/snap.svg"  height="120px" ></a>
    &nbsp
    <a><img src="figs/smile.png"  height="100px" ></a>
</div>

This repository is for the new neral light field (NeLF) method introduced in the following ECCV'22 paper:
> **[R2L: Distilling Neural Radiance Field to Neural Light Field for Efficient Novel View Synthesis](https://snap-research.github.io/R2L/)** \
> [Huan Wang](http://huanwang.tech/) <sup>1,2</sup>, [Jian Ren](https://alanspike.github.io/) <sup>1</sup>, [Zeng Huang](https://zeng.science/) <sup>1</sup>, [Kyle Olszewski](https://kyleolsz.github.io/) <sup>1</sup>, [Menglei Chai](https://mlchai.com/) <sup>1</sup>, [Yun Fu](http://www1.ece.neu.edu/~yunfu/) <sup>2</sup>, and [Sergey Tulyakov](http://www.stulyakov.com/) <sup>1</sup> \
> <sup>1</sup> Snap Inc. <sup>2</sup> Northeastern University \
> Work done when Huan was an intern at Snap Inc.

**[TL;DR]** We present R2L, a deep (88-layer) residual MLP network that can represent the neural *light* field (NeLF) of complex synthetic and real-world scenes. It is featured by compact representation size (~20MB storage size), faster rendering speed (~30x speedup than NeRF), significantly improved visual quality (1.4dB boost than NeRF), with no whistles and bells (no special data structure or parallelism required).

<div align="center">
    <a><img src="figs/frontpage.png"  width="700" ></a>
</div>


## Reproducing Our Results
Below we only show the example of scene `lego`. You may test on other scenes simply by changing all the `lego` word segment to other scene names. Scripts have been doubled-checked. You should be able to **run them simply by copy-paste**.  

### 0. Download the code
```
git clone git@github.com:snap-research/R2L.git && cd R2L
```


### 1. Set up (original) data
```bash
sh scripts/download_example_data.sh
```

### 2. Set up environment with Anaconda
- `conda create --name R2L python=3.9.6`
- `conda activate R2L`
- `pip install -r requirements.txt` (We use torch 1.9.0, torchvision 0.10.0)

### 3. Quick start: test our trained models
- Download models:
```
sh scripts/download_R2L_models.sh
```

- Run
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --model_name R2L --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --use_residual --trial.ON --trial.body_arch resmlp --pretrained_ckpt R2L_Blender_Models/lego.tar --render_only --render_test --testskip 1 --experiment_name Test__R2L_W256D88__blender_lego
```  
 
### 4. Train R2L models
There are two major steps in R2L training. (1) Use *pretrained* NeRF model to generate synthetic data and train R2L network on the synthetic data -- this step can make our R2L model perform *comparably* to the NeRF teacher; (2) Finetune the R2L model in (1) with the *real* data -- this step will further boost the performance and make our R2L model *outperform* the NeRF teacher.

The detailed step-by-step training pipeline is as follows.

#### Step 1. 
Train a NeRF model:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --model_name nerf --config configs/lego.txt --experiment_name NeRF__blender_lego
```

You can also download the teachers we trained to continue first:
```bash
sh scripts/download_NeRF_models.sh
```

To test the download teachers, you can use
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --model_name nerf --config configs/lego.txt --pretrained_ckpt NeRF_Blender_Models/lego.tar --render_only --render_test --testskip 1 --experiment_name Test__NeRF__blender_lego
```


#### Step 2. 
Use the pretrained NeRF model to generate synthetic data (saved in `.npy` format):
```bash
CUDA_VISIBLE_DEVICES=0 python utils/create_data.py --create_data rand --config configs/lego.txt --teacher_ckpt Experiments/NeRF__blender_lego*/weights/200000.tar --n_pose_kd 10000 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_pseudo_images10k --experiment_name NeRF__blender_lego__create_pseudo
```

If you are using the downloaded teachers, please use this snippet:
```bash
CUDA_VISIBLE_DEVICES=0 python utils/create_data.py --create_data rand --config configs/lego.txt --teacher_ckpt NeRF_Blender_Models/lego.tar --n_pose_kd 10000 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_pseudo_images10k --experiment_name NeRF__blender_lego__create_pseudo
```

The pseudo data will be saved in `data/nerf_synthetic/lego_pseudo_images10k`. Every 4096 rays are saved in one .npy file. For 10k images (400x400 resoltuion), there will be 309600 .npy files. On our RTX 2080Ti GPU, rendering 1 image with NeRF takes around 8.5s, so 10k images would take around 24hrs. **If you want to try our method quicker, you may download the lego data we synthesized** (500 images, 2.8GB) and go to Step 3:
```bash
sh scripts/download_lego_pseudo_images500.sh
```
The data will be extracted under `data/nerf_synthetic/lego_pseudo_images500`. Using only 500 pseudo images for training would lead to degraded quality, but based on our ablation study (see Fig. 6 in our paper), it works farily good.


#### Step 3.
Train R2L model on the synthetic data:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --model_name R2L --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_synthetic/lego_pseudo_images10k --n_pose_video 20,1,1 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --trial.ON --trial.body_arch resmlp --num_worker 8 --warmup_lr 0.0001,200 --experiment_name R2L__blender_lego
```

If you are using the downloaded `lego_pseudo_images500` data, please use this snippet:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --model_name R2L --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_synthetic/lego_pseudo_images500 --n_pose_video 20,1,1 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --trial.ON --trial.body_arch resmlp --num_worker 8 --warmup_lr 0.0001,200 --experiment_name R2L__blender_lego
```

#### Step 4. 
Convert original real data (images) to our `.npy` format:
* For blender data:
```bash
python utils/convert_original_data_to_rays_blender.py --splits train --datadir data/nerf_synthetic/lego
```
The converted data will be saved in `data/nerf_synthetic/lego_real_train`.

* For llff data:
```bash
python utils/convert_original_data_to_rays_llff.py --splits train --datadir data/nerf_llff_data/flower
```
The converted data will be saved in `data/nerf_llff_data/room_real_train`.


#### Step 5. 
Finetune the R2L model in Step 3 on the data in Step 4:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --model_name R2L --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_synthetic/lego_real_train --n_pose_video 20,1,1 --N_iters 1600000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --trial.ON --trial.body_arch resmlp --num_worker 8 --warmup_lr 0.0001,200 --save_intermediate_models --pretrained_ckpt Experiments/R2L__blender_lego_SERVER*/weights/ckpt_1200000.tar --resume --experiment_name R2L__blender_lego__ft
```
Note, this step is pretty fast and prone to overfitting, so do not finetune it too much. We simply set the finetuning steps based on our validation.


## Results
The quantitative and qualitative comparison are shown below. See more results and videos on our [webpage](https://snap-research.github.io/R2L/).
<div align="center">
    <a><img src="figs/blender_psnr_comparison.png"  width="700" ></a><br>
    <a><img src="figs/blender_visual_comparison.png"  width="700"></a>
</div>


## Acknowledgments
In this code we refer to the following implementations: [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch) and [smilelogging](https://github.com/MingSun-Tse/smilelogging). Great thanks to them! We especially thank [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch). Our code is largely built upon their wonderful implementation. We also greatly thank the anounymous ECCV'22 reviewers for the constructive comments to help us improve the paper.

## Reference

If our work or code helps you, please consider to cite our paper. Thank you!
```BibTeX
@inproceedings{wang2022r2l,
  author = {Huan Wang and Jian Ren and Zeng Huang and Kyle Olszewski and Menglei Chai and Yun Fu and Sergey Tulyakov},
  title = {R2L: Distilling Neural Radiance Field to Neural Light Field for Efficient Novel View Synthesis},
  booktitle = {ECCV},
  year = {2022}
}
```



