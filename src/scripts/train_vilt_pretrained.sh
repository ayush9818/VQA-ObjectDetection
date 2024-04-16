# !/bin/bash

base_dir=/nfs/home/scg1143/MLDS/Quarter3/DeepLearning/Project/VQA-ObjectDetection/dataset
model_name=vilt
epochs=60
learning_rate=5e-5
batch_size=128
pretrained_model_path=/nfs/home/scg1143/MLDS/Quarter3/DeepLearning/Project/VQA-ObjectDetection/dataset/runs/0/model_state_19.pth

python train.py $base_dir \
    --model-name=$model_name \
    --num-epochs=$epochs \
    --lr=$learning_rate \
    --batch-size=$batch_size \
    --pretrained-model-path=$pretrained_model_path