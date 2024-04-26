import argparse
import torch
from network import VQAModelV1
from utils import load_clip, get_dataloader, get_optimizer, vqa_accuracy
import numpy as np 
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time 
import os
from loguru import logger
import json


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-image-dir", type=str)
    parser.add_argument("--val-image-dir", type=str)
    parser.add_argument("--train-file-path", type=str)
    parser.add_argument("--val-file-path", type=str)
    parser.add_argument("--save-dir", type=str, default="../runs/ans_model")
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--clip-model", type=str, default="RN50")

    # Training Params
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    return args


def train_vqa(model, data_loader, id2label, criterion, optimizer, device):
    model.train()
    train_loss = 0
    accuracy = 0
    for i, (index, x, answers, _) in enumerate(data_loader):
        x = x.to(device) 
        answers = answers.to(device)
        # Forward Pass
        outputs = model(x).squeeze(1)
        loss = criterion(outputs, answers)
        
        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        
        # Update Weights
        optimizer.step()

        # Loss and Accuracy Calculations
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        for ip, idx in enumerate(index):
            accuracy += vqa_accuracy(data_loader.dataset.df, id2label, int(idx), predicted[ip])

    # id2label, train_index, pred_index
    train_loss /= len(data_loader.dataset)
    accuracy /= len(data_loader.dataset)
    
    return train_loss, accuracy

def validate_vqa(model, data_loader, id2label, criterion, device):
    model.eval()
    val_loss = 0
    accuracy = 0
    with torch.no_grad():
        for i, (index, x, answers, _) in enumerate(data_loader):
            x = x.to(device)
            answers = answers.to(device)
            # Forward Pass
            outputs = model(x).squeeze(1)
            
            loss = criterion(outputs, answers)
            
            # Loss and Accuracy Calculations
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            for ip, idx in enumerate(index):
                accuracy += vqa_accuracy(data_loader.dataset.df, id2label, int(idx), predicted[ip])

    val_loss /= len(data_loader.dataset)
    accuracy /= len(data_loader.dataset)

    return val_loss, accuracy

def main(cfg):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    clip_model, clip_preprocess = load_clip(model_name=cfg.clip_model, device=device)

    logger.info("Creating Dataloaders")
    train_dataset, train_loader = get_dataloader(
        df_path=cfg.train_file_path,
        image_dir=cfg.train_image_dir,
        clip_model=clip_model,
        device=device,
        batch_size=cfg.batch_size,
        shuffle=True,
        transform=clip_preprocess, 
        id2label=None,
        label2id=None
    )
    val_dataset, val_loader = get_dataloader(
        df_path=cfg.val_file_path,
        image_dir=cfg.val_image_dir,
        clip_model=clip_model,
        device=device,
        batch_size=cfg.batch_size,
        shuffle=False,
        transform=clip_preprocess,
        id2label=train_dataset.id2label,
        label2id=train_dataset.label2id
    )

    network = VQAModelV1(input_dim=2048, hidden_dim=cfg.hidden_dim, output_dim=len(train_dataset.label2id)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(network=network, learning_rate=cfg.lr, optim_name=cfg.optimizer)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=False)

    train_acc_history = []
    train_loss_history = []
    val_acc_history = []
    val_loss_history = []
    logger.info("Starting Training")
    logger.info(f"Batch Size : {cfg.batch_size}, Learning Rate : {cfg.lr}")
    counter = 0
    best_val_loss = float('inf')
    for epoch in range(cfg.num_epochs):
        logger.info(f"Epoch [{epoch + 1}/{cfg.num_epochs}]:")
        start_time = time.perf_counter()
        
        train_loss, train_acc = train_vqa(network, train_loader, train_dataset.id2label, criterion, optimizer, device)
        val_loss, val_acc = validate_vqa(network, val_loader, train_dataset.id2label, criterion, device)
        
        epoch_time = time.perf_counter() - start_time
        avg_step_time = epoch_time / (len(train_loader) + len(val_loader))
            
        train_acc_history.append(train_acc)
        train_loss_history.append(train_loss)
        val_acc_history.append(val_acc)
        val_loss_history.append(val_loss)
        
        # Check if the validation loss has improved
        if val_loss < best_val_loss:
            MODEL_NAME = cfg.model_name+f"_{val_acc}.pth"
            logger.info(f"val_loss improved from {best_val_loss:.5f} to {val_loss:.5f}, saving model to {MODEL_NAME}")
            best_val_loss = val_loss
            counter = 0
            # Save the model checkpoint
            checkpoint_path_ans = os.path.join(cfg.checkpoint_dir, MODEL_NAME)
            torch.save(network, checkpoint_path_ans)
        else:
            counter += 1
            # if counter >= cfg.patience:
            #     logger.info(f"val_loss hasn't improved for {cfg.patience} epochs. Early stopping.")
            #     break
        
        logger.info(f"{int(np.round(epoch_time))}s {avg_step_time*1e3:.4f}ms/step - loss: {train_loss:.4f} - accuracy: {train_acc*100:.4f}% - val_loss: {val_loss:.4f} - val_accuracy: {val_acc*100:.4f}% - lr: {optimizer.param_groups[0]['lr']}")
        
        lr_scheduler.step(val_loss)
        print()

    history = {
        "train_acc" : train_acc_history,
        "train_loss" : train_loss_history,
        "val_acc" : val_acc_history,
        "val_loss" : val_loss_history
    }
    with open(os.path.join(cfg.checkpoint_dir, 'history.json'), 'w') as f:
        json.dump(history, f)

if __name__ == "__main__":
    cfg = parse_arguments()

    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)
    existing_runs = os.listdir(save_dir)
    if len(existing_runs) == 0:
        curr_run = 0 
    else:
        curr_run = max([int(run) for run in existing_runs]) + 1
    run_dir = os.path.join(save_dir, str(curr_run))
    os.makedirs(run_dir)
    logger.info(f"Setting {run_dir} as the save dir for the model")
    log_file_path = os.path.join(run_dir, 'trainingLogs.log')
    logger.add(log_file_path)

    cfg.checkpoint_dir = run_dir
    logger.info(cfg.__dict__)

    main(cfg)
