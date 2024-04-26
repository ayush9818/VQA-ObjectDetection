import os
import argparse
import torch 
from utils import (
    create_label_mappings,
    create_dataset,
    get_preprocessor,
    create_dataloaders,
    create_model
)
import pandas as pd
from loguru import logger
from tqdm import tqdm
from metrics import MetricComputer

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", type=str, help="base data directory wrt to dataset")
    parser.add_argument("--image-dir", type=str, default="images")
    parser.add_argument("--eval-file", type=str, default="data_eval.csv")
    parser.add_argument("--answer-space", type=str, default="answer_space.txt")
    parser.add_argument("--model-name", type=str, default="vilt")
    parser.add_argument("--pretrained-model-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    return args

def eval(model, loader, device):
    model.to(device)

    metric_computer  = MetricComputer()
    model.eval()
    num_corrects = 0.0
    test_loss = 0.0
    total_samples = 0.0
    with torch.no_grad():
        for batch in tqdm(loader):
            # get the inputs;
            batch = {k:v.to(device) for k,v in batch.items()}
            total_samples += batch['input_ids'].size(0)


            # forward + backward + optimize
            outputs = model(**batch)
            loss = outputs.loss
            test_loss+=loss.item()
            labels = batch['labels']
            logits = outputs.logits
            num_corrects+= metric_computer.calculate_correct_preds(labels=labels, logits=logits)
    
    test_acc = round(float((num_corrects / total_samples)) * 100,4)
    test_loss = test_loss / total_samples
    return model, test_loss, test_acc

if __name__ == "__main__":
    cfg = parse_arguments()

    assert os.path.exists(cfg.base_dir), f"{cfg.base_dir} does not exists"

    image_dir = os.path.join(cfg.base_dir, cfg.image_dir)
    eval_file = os.path.join(cfg.base_dir, cfg.eval_file)
    answer_space_file = os.path.join(cfg.base_dir, cfg.answer_space)

    for path in [image_dir, eval_file, answer_space_file]:
        assert os.path.exists(path), f"{path} not found"

    eval_df = pd.read_csv(eval_file)

    logger.info(
        f"Eval Data Size : {eval_df.shape[0]}"
    )

    model_name = cfg.model_name
    processor = get_preprocessor(model_name=model_name)

    mapping = create_label_mappings(answer_space_path=answer_space_file)

    dataset = create_dataset(
        train_df=eval_df,
        eval_df=eval_df,
        label2id=mapping["label2id"],
        id2label=mapping["id2label"],
        image_dir=image_dir,
        processor=processor,
    )


    dataloaders = create_dataloaders(
        dataset=dataset, processor=processor, batch_size=cfg.batch_size
    )
    pretrained_model_path = cfg.pretrained_model_path
    model = create_model(model_name=cfg.model_name, label_mappings=mapping, pretrained=pretrained_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, test_loss, test_acc = eval(model, dataloaders['eval'], device)
    logger.info(f"Test Loss : {test_loss} Test Accuracy : {test_acc}")