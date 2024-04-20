# !/bin/bash

base_dir=/nfs/home/scg1143/MLDS/Quarter3/DeepLearning/Project/VQA-ObjectDetection/dataset
model_name=vilt
epochs=40
learning_rate=2e-3
batch_size=256
save_dir=/nfs/home/scg1143/MLDS/Quarter3/DeepLearning/Project/VQA-ObjectDetection/runs
freeze_layers=5
device=cuda:1
optimizer=adam
freeze_embeddings=1

python train.py $base_dir \
    --model-name=$model_name \
    --num-epochs=$epochs \
    --lr=$learning_rate \
    --batch-size=$batch_size\
    --save-dir=$save_dir\
    --freeze-layers=$freeze_layers \
    --device=$device \
    --optimizer=$optimizer\
    --freeze-embeddings=$freeze_embeddings
