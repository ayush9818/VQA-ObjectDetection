# !/bin/bash

base_dir=/nfs/home/scg1143/MLDS/Quarter3/DeepLearning/Project/VQA-ObjectDetection/dataset
model_name=vilt
epochs=4
learning_rate=5e-5
batch_size=128

python train.py $base_dir \
    --model-name=$model_name \
    --num-epochs=$epochs \
    --lr=$learning_rate \
    --batch-size=$batch_size
