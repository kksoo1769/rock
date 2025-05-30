export OMP_NUM_THREADS=1

N_GPUS=2
MASTER_PORT=$((12000 + $RANDOM % 20000))


torchrun --nproc_per_node="$N_GPUS" --master_port="$MASTER_PORT" ../../main.py \
    --batch_size 64 \
    --epochs 50 \
    --update_freq 1 \
    --model convnextv2_large \
    --drop_path 0.3 \
    --input_size 384 \
    --model_ema True \
    --model_ema_decay 0.999 \
    --model_ema_eval True \
    --opt_betas 0.9 0.999 \
    --clip_grad 5.0 \
    --weight_decay 0.001 \
    --lr 3e-4 \
    --layer_decay 0.6 \
    --min_lr 1e-6 \
    --warmup_epochs 5 \
    --scale 0.1 1.0 \
    --color_jitter 0.4 \
    --aa 'rand-m9-mstd0.5' \
    --smoothing 0.1 \
    --reprob 0.25 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --finetune '/nas/Dataset/Dacon/rock/pretrained_weights/convnextv2_large_22k_384_ema.pt' \
    --data_path '/nas/Dataset/Dacon/rock_split/train' \
    --val_data_path '/nas/Dataset/Dacon/rock_split/val' \
    --nb_classes 7 \
    --imagenet_default_mean_and_std True \
    --data_set 'image_folder' \
    --output_dir '/home/kks/workspace/rock/exp/large-amp-384-2' \
    --log_dir '/home/kks/workspace/rock/exp/large-amp-384-2' \
    --save_ckpt True \
    --dist_eval True \
    --num_workers 8 \
    --pin_mem True \
    --use_amp True \
    --auto_resume True \