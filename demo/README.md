# Shadow map demo
All the following instructions should be done from this directory.

## Set up environment
- `conda create --name shadow-demo python=3.9.6`
- `conda activate shadow-demo`
- `pip install -r requirements.txt`

## (Optional) Download the lego model
If this step is not done, the displayed object will be a cube, but the shadows will still correspond to the lego model. Rendering the cube is much faster.

```bash
sh ../scripts/download_lego_model.sh
```

## Run the shadow demo
With cube model:
```bash
python demoShadows.py cube.obj
```

With lego model:
```bash
python demoShadows.py lego.obj
```
