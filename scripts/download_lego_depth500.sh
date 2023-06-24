wget https://github.com/MihneaToader/Depth-Light-Field-Training/releases/download/data/lego_depth500.zip
DIR='data/nerf_synthetic/'
[ ! -d "$DIR" ] && mkdir -p "$DIR"
unzip lego_depth500.zip