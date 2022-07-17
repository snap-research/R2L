wget https://github.com/MingSun-Tse/Efficient-NeRF/releases/download/v0.3/lego_pseudo_images500.zip
DIR='data/nerf_synthetic/'
[ ! -d "$DIR" ] && mkdir -p "$DIR"
unzip lego_pseudo_images500.zip