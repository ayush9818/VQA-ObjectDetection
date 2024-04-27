import warnings
warnings.filterwarnings('ignore')

import clip 
import numpy as np 
import os
import pandas as pd
from PIL import Image
import torch
import argparse
from loguru import logger

def load_clip(model_name='RN50', device='cpu'):
    """Load Clip Model and Preprocessor"""
    clip_model, preprocess = clip.load(model_name, device=device)
    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()]):,}")
    print(f"Input resolution: {clip_model.visual.input_resolution}")
    print(f"Context length: {clip_model.context_length}")
    print(f"Vocab size: {clip_model.vocab_size}")
    return clip_model, preprocess

def extract_features(image_path, question, clip_model, transform, device='cpu'):
    """Takes input as image_path and question and extract text and img clip featueres"""
    image = Image.open(image_path).convert('RGB')
    if transform is not None:
        image = transform(image).unsqueeze(0).to(device)
    question = clip.tokenize(question).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(question)
    return image_features.detach().cpu(), text_features.detach().cpu()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path",type=str)
    parser.add_argument("--feat-save-dir",type=str)
    parser.add_argument("--clip-model", type=str, default='RN50')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--save-path",type=str)

    args = parser.parse_args()

    assert os.path.exists(args.data_path)

    feat_save_dir = args.feat_save_dir
    device=args.device
    os.makedirs(feat_save_dir, exist_ok=True)
    clip_model_name = args.clip_model
    save_path = args.save_path

    clip_model, preprocess = load_clip(model_name='RN50', device=device)

    data = pd.read_json(args.data_path)
    #data = data.iloc[:501]
    img_feat_list = []
    text_feat_list = []

    total = len(data)
    done = 0
    logger.info("Extracting Features ......")
    for idx,row in data.iterrows():
        image_path = row['image_path']
        question = row['question']
        text_feat_name = row['image'].split('.')[0] + '_text.pt'
        img_feat_name = row['image'].split('.')[0] + '_img.pt'
        
        img_feat, text_feat = extract_features(image_path=image_path,
                                       question=question,
                                       clip_model=clip_model,
                                       transform=preprocess,
                                       device=device
                                    )
        img_feat_path = os.path.join(feat_save_dir,img_feat_name)
        text_feat_path = os.path.join(feat_save_dir,text_feat_name)
        torch.save(img_feat, img_feat_path)
        torch.save(text_feat, text_feat_path)

        img_feat_list.append(img_feat_path)
        text_feat_list.append(text_feat_path)
        done+=1

        if done % 500 == 0:
            logger.info(f"Total={total} Done={done}")

    data['img_feat'] = img_feat_list
    data['text_feat'] = text_feat_list

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data.to_json(save_path)
    logger.info(f"saved to {save_path}")
    logger.info("bye bye")