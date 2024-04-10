import os
import argparse
import torch 
from utils import (
    create_label_mappings,
    create_dataset,
    get_preprocessor,
    create_dataloaders,
    create_model, 
    get_optimizer
)
import pandas as pd
from loguru import logger
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", type=str, help="base data directory wrt to dataset")
    parser.add_argument("--image-dir", type=str, default="images")
    parser.add_argument("--train-file", type=str, default="data_train.csv")
    parser.add_argument("--eval-file", type=str, default="data_eval.csv")
    parser.add_argument("--answer-space", type=str, default="answer_space.txt")
    parser.add_argument("--model-name", type=str, default="vilt")

    # Training Params
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--optimizer", type=str, default="adam")
    args = parser.parse_args()
    return args


def train(cfg, dataloaders, label_mappings):
    model = create_model(model_name=cfg.model_name, label_mappings=label_mappings)
    optimizer = get_optimizer(optimizer_name=cfg.optimizer, model=model, learning_rate=cfg.lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device : {device}")
    model.to(device)

    num_epochs = cfg.num_epochs
    logger.info(f"Total Epochs : {num_epochs}")
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        epoch_loss = 0.0
        total_samples = 0  
        for phase in ['train', 'eval']:
            loader = dataloaders[phase]
            if phase == 'train':
                model.train()
            else:
                model.eval()
            for batch in tqdm(loader):
                # get the inputs;
                batch = {k:v.to(device) for k,v in batch.items()}
                total_samples += batch['input_ids'].size(0)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(**batch)
                loss = outputs.loss
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                epoch_loss+=loss.item()
            epoch_loss = epoch_loss / total_samples
            logger.info(f"Epoch : {epoch} Phase : {phase} Loss : {epoch_loss}")


if __name__ == "__main__":
    cfg = parse_arguments()

    assert os.path.exists(cfg.base_dir), f"{cfg.base_dir} does not exists"

    image_dir = os.path.join(cfg.base_dir, cfg.image_dir)
    train_file = os.path.join(cfg.base_dir, cfg.train_file)
    eval_file = os.path.join(cfg.base_dir, cfg.eval_file)
    answer_space_file = os.path.join(cfg.base_dir, cfg.answer_space)

    for path in [image_dir, train_file, eval_file, answer_space_file]:
        assert os.path.exists(path), f"{path} not found"

    mapping = create_label_mappings(answer_space_path=answer_space_file)

    # Reading Training and Eval Dataframe
    train_df = pd.read_csv(train_file)
    eval_df = pd.read_csv(eval_file)

    logger.info(
        f"Train Data Size : {train_df.shape[0]} Eval Data Size : {eval_df.shape[0]}"
    )

    model_name = cfg.model_name
    processor = get_preprocessor(model_name=model_name)

    dataset = create_dataset(
        train_df=train_df,
        eval_df=eval_df,
        label2id=mapping["label2id"],
        id2label=mapping["id2label"],
        image_dir=image_dir,
        processor=processor,
    )

    dataloaders = create_dataloaders(
        dataset=dataset, processor=processor, batch_size=cfg.batch_size
    )

    train(cfg, dataloaders, mapping)