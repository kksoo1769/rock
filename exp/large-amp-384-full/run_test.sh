export OMP_NUM_THREADS=1

N_GPUS=1
MASTER_PORT=$((12000 + $RANDOM % 20000))


torchrun --nproc_per_node="$N_GPUS" --master_port="$MASTER_PORT" ../../main.py \
    --batch_size 256 \
    --model convnext_large \
    --model_ema True \
    --input_size 384 \
    --data_path '/nas/Dataset/Dacon/rock/train' \
    --test_data_path '/nas/Dataset/Dacon/rock/test' \
    --output_csv_path '/home/kks/workspace/rock/exp/large-amp-384-full/submission.csv' \
    --ema_output_csv_path '/home/kks/workspace/rock/exp/large-amp-384-full/submission-ema.csv' \
    --sample_submission_path '/nas/Dataset/Dacon/rock/sample_submission.csv' \
    --nb_classes 7 \
    --imagenet_default_mean_and_std True \
    --data_set 'image_folder' \
    --output_dir '/home/kks/workspace/rock/exp/large-amp-384-full' \
    --log_dir '/home/kks/workspace/rock/exp/large-amp-384-full' \
    --resume '/home/kks/workspace/rock/exp/large-amp-384-full/checkpoint-49.pth' \
    --dist_eval False \
    --disable_eval True \
    --eval True \
    --num_workers 8 \
    --pin_mem True \
    --use_amp True \
