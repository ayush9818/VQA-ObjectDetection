# !/bin/bash

train_image_dir=/nfs/home/scg1143/MLDS/Quarter3/DeepLearning/Project/VQA-ObjectDetection/vizviz/vqa/train
val_image_dir=/nfs/home/scg1143/MLDS/Quarter3/DeepLearning/Project/VQA-ObjectDetection/vizviz/vqa/train
train_file_path=/nfs/home/scg1143/MLDS/Quarter3/DeepLearning/Project/VQA-ObjectDetection/vizviz/vqa/train_df_thresh_3.csv
val_file_path=/nfs/home/scg1143/MLDS/Quarter3/DeepLearning/Project/VQA-ObjectDetection/vizviz/vqa/val_df_thresh_3.csv
save_dir=/nfs/home/scg1143/MLDS/Quarter3/DeepLearning/Project/VQA-ObjectDetection/runs/vqa_clip_model
model_name=clip_vqa
clip_model=RN50

num_epochs=60
learning_rate=0.01
batch_size=32


python vqa/train.py \
    --train-image-dir=$train_image_dir \
    --val-image-dir=$val_image_dir \
    --train-file-path=$train_file_path \
    --val-file-path=$val_file_path \
    --save-dir=$save_dir \
    --num-epochs=$num_epochs \
    --lr=$learning_rate \
    --batch-size=$batch_size \
    --model-name=$model_name \
    --clip-model=$clip_model
