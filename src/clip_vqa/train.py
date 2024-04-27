import warnings
warnings.filterwarnings("ignore")

import os 
import pandas as pd
from loguru import logger 
import yaml
import sys 
from network import VQAModel
from dataset import VizWizDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import OrdinalEncoder
import numpy as np 
import pickle
from tqdm import tqdm 
import torch 
from utils import get_optimizer, accuracy_vqa, Config
import torch.nn as nn 
import time 
from torch.utils.tensorboard import SummaryWriter


def train_vqa(model, data_loader, criterion, optimizer, device, enc):
    model.train()
    train_loss = 0
    accuracy = 0
    for index, x, answers, _ in tqdm(data_loader):
        x = x.to(device) 
        answers = torch.as_tensor(enc.transform(np.array(answers).reshape(-1, 1)).astype(int)).to(device).squeeze(1)
        
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
        predicted = enc.inverse_transform(np.array(predicted.to('cpu')).reshape(-1,1))
        for ip, idx in enumerate(index):
            accuracy += accuracy_vqa(data_loader.dataset.df, int(idx), predicted[ip])

    train_loss /= len(data_loader.dataset)
    accuracy /= len(data_loader.dataset)
    
    return train_loss, accuracy

def validate_vqa(model, data_loader, criterion, device, enc):
    model.eval()
    val_loss = 0
    accuracy = 0
    with torch.no_grad():
        for index, x, answers, _ in tqdm(data_loader):
            x = x.to(device)
            answers = torch.as_tensor(enc.transform(np.array(answers).reshape(-1, 1)).astype(int)).to(device).squeeze(1)
            
            # Forward Pass
            outputs = model(x).squeeze(1)
            loss = criterion(outputs, answers)
            
            # Loss and Accuracy Calculations
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            predicted = enc.inverse_transform(np.array(predicted.to('cpu')).reshape(-1,1))
            for ip, idx in enumerate(index):
                accuracy += accuracy_vqa(data_loader.dataset.df, int(idx), predicted[ip])

    val_loss /= len(data_loader.dataset)
    accuracy /= len(data_loader.dataset)

    return val_loss, accuracy

def main(model_config, data_config):
    writer = SummaryWriter(data_config.checkpoint_dir)
    os.makedirs(data_config.checkpoint_dir, exist_ok=True)

    # Creating Dataset and Dataloaders
    train_dataset = VizWizDataset(df_path=data_config.train_data_path, filter_answerable=False)
    valid_dataset = VizWizDataset(df_path=data_config.val_data_path, filter_answerable=False)

    train_loader = DataLoader(train_dataset, batch_size=model_config.batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=model_config.batch_size, shuffle=True)
    logger.info("Dataset Loaded")

    # Fitting Label Encoder
    ANSWER_CANDIDATES = train_dataset.df['final_answer'].nunique()
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=ANSWER_CANDIDATES)
    enc.fit(np.array(train_dataset.df['final_answer']).reshape(-1, 1))

    # saving the encoder
    with open(os.path.join(data_config.checkpoint_dir, 'encoder.pkl'), 'wb') as f:
        pickle.dump(enc, f)

    output_dim = ANSWER_CANDIDATES + 1

    device = torch.device(model_config.device if torch.cuda.is_available() else 'cpu')

    model_vqa = VQAModel(model_config.input_dim, model_config.hidden_dim, output_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer, lr_scheduler = get_optimizer(model_config, model_vqa)


    # Define the variables for saving the best checkpoint
    best_val_acc = 0.0
    #patience = 10
        
    # Defining Lists to store training and validation accuracies and losses
    train_vqa_acc_history = []
    train_vqa_loss_history = []
    val_vqa_acc_history = []
    val_vqa_loss_history = []

    #counter = 0

    for epoch in range(model_config.num_epochs):
        print(f"Epoch [{epoch + 1}/{model_config.num_epochs}]:")
        start_time = time.perf_counter()
        
        train_loss, train_acc = train_vqa(model_vqa, train_loader, criterion, optimizer, device, enc)
        val_loss, val_acc = validate_vqa(model_vqa, val_loader, criterion, device, enc)
        
        epoch_time = time.perf_counter() - start_time
        avg_step_time = epoch_time / (len(train_loader) + len(val_loader))
            
        train_vqa_acc_history.append(train_acc)
        train_vqa_loss_history.append(train_loss)
        val_vqa_acc_history.append(val_acc)
        val_vqa_loss_history.append(val_loss)

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Valid', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Valid', val_acc, epoch)
        
        # Check if the validation loss has improved
        if val_acc >  best_val_acc:
            logger.info(f"val_acc improved from {best_val_acc:.5f} to {val_acc:.5f}")
            best_val_acc = val_acc
            checkpoint_name = f"{model_config.model_name}_{best_val_acc:.4f}.pth"
            torch.save(model_vqa, os.path.join(data_config.checkpoint_dir, checkpoint_name))
            logger.info(f"Model Saved as {checkpoint_name}")
            #counter = 0
            # Save the model checkpoint)
        # else:
        #     counter += 1
        #     if counter >= patience:
        #         print(f"val_loss hasn't improved for {patience} epochs. Early stopping.")
        #         break
        
        logger.info(f"{int(np.round(epoch_time))}s {avg_step_time*1e3:.4f}ms/step - loss: {train_loss:.4f} - accuracy: {train_acc*100:.4f}% - val_loss: {val_loss:.4f} - val_accuracy: {val_acc*100:.4f}% - lr: {optimizer.param_groups[0]['lr']}")
        
        lr_scheduler.step(val_loss)
        print()
    writer.close()



if __name__ == "__main__":
    config_path = sys.argv[1]
    config = yaml.full_load(open(config_path))

    model_config = config['model_config']
    data_config = config['data_config']

    save_dir = data_config['save_dir']
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

    data_config['checkpoint_dir'] = run_dir

    logger.info(model_config)
    logger.info(data_config)

    model_config = Config(model_config)
    data_config = Config(data_config)
    main(model_config, data_config)