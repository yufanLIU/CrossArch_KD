torchrun --nproc_per_node=8 dist_train_cakd.py --batch-size 32 --lr 0.1 \
--lr-warmup-epochs 5 --lr-warmup-method linear \
--auto-augment ta_wide --epochs 120 --random-erase 0.1 --mixup-alpha 0.2 --weight-decay 0.00002 \
--norm-weight-decay 0.0 --label-smoothing 0.1 \
--train-crop-size 224 --model-ema --val-resize-size 224 --ra-sampler --ra-reps 4 \
--output-dir results/resnet50/cakd_vitb16
