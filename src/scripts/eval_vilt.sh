base_dir=/nfs/home/scg1143/MLDS/Quarter3/DeepLearning/Project/VQA-ObjectDetection/dataset
model_name=vilt
batch_size=64
pretrained_model_path=/nfs/home/scg1143/MLDS/Quarter3/DeepLearning/Project/VQA-ObjectDetection/runs/2/model_state_0.pth

python eval.py $base_dir \
    --model-name=$model_name \
    --batch-size=$batch_size \
    --pretrained-model-path=$pretrained_model_path