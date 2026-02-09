exp_name="yolox_dancetrack_val_hybrid_sort_2_8_v1"
python tools/train.py \
  -f exps/example/mot/yolox_dancetrack_val_hybrid_sort.py \
  -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth \
  -expn $exp_name