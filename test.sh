# installations (for rest, follow the author)
# conda create -n hybridsort python=3.8 -y
# conda activate hybridsort
# pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
# pip install "numpy<1.24"

# create dataset symlinks
# ln -s /ssd0/shenzhen/Datasets/tracking datasets

# NOTE: need to change hybrid_sort.py 
# (ref: https://github.com/ymzis69/HybridSORT/issues/49)

# test with pre-trained model (DanceTrack)
# exp_name="hybridsort_dancetrack_val"
# python tools/run_hybrid_sort_dance.py \
#     -f exps/example/mot/yolox_dancetrack_val_hybrid_sort.py \
#     -b 1 -d 1 --fp16 --fuse --expn ${exp_name}

# test with re-trained model (DanceTrack)
# exp_name="yolox_dancetrack_val_hybrid_sort"
# python tools/run_hybrid_sort_dance.py \
#     -f exps/example/mot/yolox_dancetrack_val_hybrid_sort.py \
#     -b 1 -d 1 --fp16 --fuse --expn ${exp_name} \
#     -c YOLOX_outputs/${exp_name}/best_ckpt.pth.tar

# test with re-trained model (DanceTrack)
exp_name="yolox_dancetrack_val_hybrid_sort_2_8_v1"
python tools/run_hybrid_sort_dance.py \
    -f exps/example/mot/yolox_dancetrack_val_hybrid_sort.py \
    -b 1 -d 1 --fp16 --fuse --expn ${exp_name} \
    -c YOLOX_outputs/${exp_name}/best_ckpt.pth.tar

# TODO: let's add warp-unwarp, see if the code can works with yolox