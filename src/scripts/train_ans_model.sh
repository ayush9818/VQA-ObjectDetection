# !/bin/bash

train_image_dir=/nfs/home/scg1143/MLDS/Quarter3/DeepLearning/Project/VQA-ObjectDetection/vizviz/vqa/train
val_image_dir=/nfs/home/scg1143/MLDS/Quarter3/DeepLearning/Project/VQA-ObjectDetection/vizviz/vqa/val
train_file_path=/nfs/home/scg1143/MLDS/Quarter3/DeepLearning/Project/VQA-ObjectDetection/vizviz/vqa/train_df.csv
val_file_path=/nfs/home/scg1143/MLDS/Quarter3/DeepLearning/Project/VQA-ObjectDetection/vizviz/vqa/eval_df.csv
save_dir=/nfs/home/scg1143/MLDS/Quarter3/DeepLearning/Project/VQA-ObjectDetection/runs/ans_model

num_epochs=1
learning_rate=0.0001
batch_size=64


python answerability_detection/train.py \
    --train-image-dir=$train_image_dir \
    --val-image-dir=$val_image_dir \
    --train-file-path=$train_file_path \
    --val-file-path=$val_file_path \
    --save-dir=$save_dir \
    --num-epochs=$num_epochs \
    --lr=$learning_rate \
    --batch-size=$batch_size
