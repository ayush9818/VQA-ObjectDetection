import numpy as np
import json
import clip
from dataset import VizWizDataset
from torch.utils.data import DataLoader
import torch
from loguru import logger

CLIP_MODELS = [
    "RN50",
    "RN101",
    "RN50x4",
    "RN50x16",
    "RN50x64",
    "ViT-B/32",
    "ViT-B/16",
    "ViT-L/14",
    "ViT-L/14@336px",
]

OPTIMIZERS = ["adam"]


def load_clip(model_name="RN50", device="cpu"):
    logger.info("Loading CLIP.....")
    assert model_name in CLIP_MODELS, f"clip models available {CLIP_MODELS}"
    clip_model, preprocess = clip.load(model_name, device=device)
    return clip_model, preprocess


def get_dataloader(df_path,
                image_dir,
                clip_model,
                device,
                batch_size,
                label2id=None,
                id2label=None,
                shuffle=True,
                transform=None,):
    dataset = VizWizDataset(
        df_path=df_path,
        image_dir=image_dir,
        clip_model=clip_model,
        device=device,
        transform=transform,
        id2label=id2label, 
        label2id=label2id
    )
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset, data_loader


def get_optimizer(network, learning_rate, optim_name):
    if optim_name.lower() == "adam":
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Optimizer Name should be one of the following {OPTIMIZERS}")
    return optimizer


def vqa_accuracy(train_df, id2label, train_index, pred_index):
    pred_label = id2label[int(pred_index)]
    train_row = train_df.iloc[int(train_index)]
    answer_set = np.array(
        [ans["answer"] for ans in json.loads(train_row["answers"].replace("'", '"'))]
    )

    score = min(1, len(np.where(answer_set == pred_label)[0]) / 3)
    return score