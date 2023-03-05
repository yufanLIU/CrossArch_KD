torchrun --nproc_per_node=8 dist_train_student.py --batch-size 32 --lr 0.1 \
--train-crop-size 224 --val-resize-size 224 \
--output-dir results/resnet50/baseline
