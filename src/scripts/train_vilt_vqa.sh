# !/bin/bash

config_path=/nfs/home/scg1143/MLDS/Quarter3/DeepLearning/Project/VQA-ObjectDetection/src/vilt_vqa/config.yaml

python vilt_vqa/train.py $config_path
