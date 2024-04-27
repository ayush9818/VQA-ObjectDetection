# !/bin/bash

data_path=/nfs/home/scg1143/MLDS/Quarter3/DeepLearning/Project/VQA-ObjectDetection/vizviz/vqa/val_df.json
feat_save_dir=/nfs/home/scg1143/MLDS/Quarter3/DeepLearning/Project/VQA-ObjectDetection/vizviz/vqa/val_feats
save_path=/nfs/home/scg1143/MLDS/Quarter3/DeepLearning/Project/VQA-ObjectDetection/vizviz/vqa/val_df_RN50.json

python3 clip_vqa/extract_features.py \
--data-path=$data_path \
--feat-save-dir=$feat_save_dir \
--save-path=$save_path