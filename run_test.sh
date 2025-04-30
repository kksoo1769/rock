export OMP_NUM_THREADS=1

N_GPUS=1
MASTER_PORT=$((12000 + $RANDOM % 20000))


torchrun --nproc_per_node="$N_GPUS" --master_port="$MASTER_PORT" main.py \
    --batch_size 64 \
    --model convnext_base \
    --input_size 224 \
    --data_path '/nas/Dataset/Dacon/rock/train' \
    --test_data_path '/nas/Dataset/Dacon/rock/test' \
    --output_csv_path '/home/kks/workspace/rock/exp/base-amp-224/submissioncsv' \
    --sample_submission_path '/nas/Dataset/Dacon/rock/sample_submission.csv' \
    --nb_classes 7 \
    --imagenet_default_mean_and_std True \
    --data_set 'image_folder' \
    --output_dir '/home/kks/workspace/rock/exp/base-amp-224' \
    --log_dir '/home/kks/workspace/rock/exp/base-amp-224' \
    --resume '/home/kks/workspace/rock/exp/base-amp-224/checkpoint-best.pth' \
    --dist_eval False \
    --disable_eval True \
    --eval True \
    --num_workers 8 \
    --pin_mem True \
