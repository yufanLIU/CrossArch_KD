torchrun --nproc_per_node=8 dist_train_logits.py --batch-size 32 --lr 0.1 \
--epochs 120 \
--label-smoothing 0.1 \
--train-crop-size 224 --model-ema --val-resize-size 224 --ra-sampler --ra-reps 4 \
--output-dir results/resnet50/logits_vitb16
