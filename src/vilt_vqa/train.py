import os
import argparse
import torch 
import pandas as pd
from loguru import logger
from tqdm import tqdm
import csv
from dataset import get_score, add_label_score, VQADataset, collate_fn
from transformers import ViltProcessor, ViltForQuestionAnswering
from utils import EarlyStopping, get_optimizer, Config
from torch.utils.data import DataLoader
import sys 
import yaml 
from torch.utils.tensorboard import SummaryWriter


torch.manual_seed(1234)

def train_model(data_loader, model, optimizer, scheduler, device):
    total_samples = len(data_loader.dataset)
    # Define loss
    total_loss = 0
    total_acc = 0
    model.train()
    for batch in tqdm(data_loader):
        # get the inputs;
        inputs = {k:v.to(device) for k,v in batch.items()}
        # forward pass
        outputs = model(**inputs)
        loss = outputs.loss
        # backward and optimize
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.logits.argmax(-1)
        scores, labels = batch["labels"].to(device).topk(10, -1)
        
        # Calculate accuracy
        for idx in range(len(scores)):
            total_acc += min(scores[idx][preds[idx] == labels[idx]].sum(),1)
        #scheduler.step()
        
    # Loss over batches
    train_loss = total_loss / total_samples
    train_acc = total_acc / total_samples
    return train_acc ,train_loss


def val_model(data_loader, model, device):
    total_samples = len(data_loader.dataset)
    # Define loss and accuracy
    total_loss = 0
    total_acc = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader):
            inputs = {k:v.to(device) for k,v in batch.items()}
            outputs = model(**inputs)

            # Calculate loss
            loss = outputs.loss
            total_loss += loss.item()

            # Get top predict for each question
            preds = outputs.logits.argmax(-1)
            # Get ground truth answers for each questiojn
            scores, labels = batch["labels"].to(device).topk(10, -1)
            # Calculate accuracy
            for idx in range(len(scores)):
                total_acc += min(scores[idx][preds[idx] == labels[idx]].sum(),1)
            
    # Accuracy over batches
    val_acc = total_acc / total_samples
    # Loss over batches
    val_loss = total_loss / total_samples

    return val_acc, val_loss


def main(cfg):
    writer = SummaryWriter(cfg.checkpoint_dir)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    with open(cfg.classmapping_dir, "r") as f:
        next(f)  # Skip the header
        reader = csv.reader(f, skipinitialspace=True)
        class_mapping = dict(reader)
        label2id = {k: int(v) for k, v in class_mapping.items()}
        id2label = {v: k for k, v in label2id.items()}

    train_data = pd.read_json(cfg.train_data_path)
    val_data = pd.read_json(cfg.val_data_path)

    train_data = add_label_score(train_data, label2id)
    val_data = add_label_score(val_data, label2id)

    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

    train_dataset = VQADataset(train_data, processor, label2id)
    val_dataset = VQADataset(val_data, processor, label2id)


    train_dataloader = DataLoader(train_dataset, 
                                  collate_fn=lambda x : collate_fn(x, processor), 
                                  batch_size=cfg.batch_size, 
                                  shuffle=True, 
                                  num_workers=2, 
                                  pin_memory=True, 
                                  drop_last=True)
    val_dataloader = DataLoader(val_dataset, 
                                collate_fn=lambda x : collate_fn(x, processor), 
                                batch_size=cfg.batch_size, 
                                shuffle=False, 
                                num_workers=2, 
                                pin_memory=True)

    device=torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm",
                                                    id2label=id2label,
                                                    label2id=label2id)

    model.to(device)

    optimizer, lr_scheduler = get_optimizer(model, cfg)


    NUM_EPOCHS = cfg.num_epochs
    best_val_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        logger.info(f"Epoch [{epoch}/{NUM_EPOCHS-1}]")
        train_acc, train_loss = train_model(train_dataloader, model, optimizer, lr_scheduler, device)
        val_acc, val_loss = val_model(val_dataloader, model, device)
        

        # Display
        logger.info(f"Train loss: {train_loss:.5f} - Val loss: {val_loss:.5f}")
        logger.info(f"Train Accuracy : {train_acc:.5f} - Val accuracy: {val_acc:.5f}\n")

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Valid', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Valid', val_acc, epoch)

        
        if val_acc >  best_val_acc:
            logger.info(f"val_acc improved from {best_val_acc:.5f} to {val_acc:.5f}")
            best_val_acc = val_acc
            checkpoint_name = f"{cfg.model_name}_{best_val_acc:.4f}.pth"
            torch.save(model, os.path.join(cfg.checkpoint_dir, checkpoint_name))
            logger.info(f"Model Saved as {checkpoint_name}")
        
        # Adjust learning rate
        lr_scheduler.step(val_loss)

    checkpoint_name = f"{cfg.model_name}_{val_acc:.4f}.pth"
    torch.save(model, os.path.join(cfg.checkpoint_dir, checkpoint_name))
    logger.info(f"Model Saved as {checkpoint_name}")
    writer.close()


if __name__ == "__main__":
    config_path = sys.argv[1]
    config = yaml.full_load(open(config_path))


    save_dir = config['save_dir']
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

    config['checkpoint_dir'] = run_dir

    logger.info(config)

    config = Config(config)
    main(config)