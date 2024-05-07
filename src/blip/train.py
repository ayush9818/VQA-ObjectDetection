import warnings
warnings.filterwarnings("ignore")

import logging
logging.disable(logging.WARNING)

from tqdm import tqdm 
import torch 
from dataset import VQADataset, collate_fn

from transformers import BlipProcessor, BlipForQuestionAnswering,BlipImageProcessor
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CyclicLR
from loguru import logger 
import time 
import os 
from torch.utils.tensorboard import SummaryWriter
import numpy as np 
import sys 
import yaml


class Config:
    def __init__(self, config):
        for key, value in config.items():
            setattr(self, key, value)

def train_one_epoch(model, train_dataloader, optimizer, scheduler, device):
    model.train()
    epoch_loss = 0.0
    for batch in tqdm(train_dataloader):
        # get the inputs;
        batch = {k:v.to(device) for k,v in batch.items()}

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(**batch)
        loss = outputs.loss
        epoch_loss+= loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    epoch_loss_avg = epoch_loss / len(train_dataloader.dataset)
    return epoch_loss_avg

def validate_model(model, test_dataloader, device):
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = {k:v.to(device) for k,v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            valid_loss+= loss.item()
    
    valid_loss_avg = valid_loss / len(test_dataloader.dataset)
    return valid_loss_avg


def main(cfg):
    writer = SummaryWriter(cfg.checkpoint_dir)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    text_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    image_processor = BlipImageProcessor.from_pretrained("Salesforce/blip-vqa-base")

    train_vqa_dataset = VQADataset(cfg.train_data_path, text_processor,image_processor)
    val_vqa_dataset = VQADataset(cfg.val_data_path, text_processor,image_processor)

    train_dataloader = DataLoader(train_vqa_dataset,
                                  collate_fn=collate_fn,
                                  batch_size=cfg.batch_size,
                                  shuffle=False)
    val_dataloader = DataLoader(val_vqa_dataset,
                                collate_fn=collate_fn,
                                batch_size=cfg.batch_size,
                                shuffle=False)

    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base" )
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"using {device}")
    model.to(device)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.learning_rate, steps_per_epoch=len(train_dataloader), epochs= cfg.num_epochs)
    image_mean = image_processor.image_mean
    image_std = image_processor.image_std

    NUM_EPOCHS = cfg.num_epochs
    best_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]:")
        start_time = time.perf_counter()
        
        train_loss = train_one_epoch(model, train_dataloader, optimizer, scheduler, device)
        val_loss = validate_model(model, val_dataloader, device)
        
        epoch_time = time.perf_counter() - start_time
        avg_step_time = epoch_time / (len(train_dataloader) + len(val_dataloader))
            

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Valid', val_loss, epoch)
        
        # Check if the validation loss has improved
        if val_loss <  best_loss:
            logger.info(f"val_acc improved from {best_loss:.5f} to {val_loss:.5f}")
            best_loss = val_loss
            checkpoint_name = f"{cfg.model_name}_{best_loss:.4f}.pth"
            state_dict = {
                "state_dict" : model.state_dict(), 
                "optimizer" : optimizer.state_dict(),
                "img_mean" : image_mean,
                "img_std" : image_std
            }
            torch.save(state_dict, os.path.join(cfg.checkpoint_dir, checkpoint_name))
            logger.info(f"Model Saved as {checkpoint_name}")
        
        logger.info(f"{int(np.round(epoch_time))}s {avg_step_time*1e3:.4f}ms/step - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - lr: {optimizer.param_groups[0]['lr']}")
        print()

    checkpoint_name = f"{cfg.model_name}_{val_loss:.4f}.pth"
    state_dict = {
        "state_dict" : model.state_dict(), 
        "optimizer" : optimizer.state_dict(),
        "img_mean" : image_mean,
        "img_std" : image_std
    }
    torch.save(state_dict, os.path.join(cfg.checkpoint_dir, checkpoint_name))
    logger.info(f"Model Saved as {checkpoint_name}")


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