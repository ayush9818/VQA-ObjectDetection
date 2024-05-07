import torch
import os
from loguru import logger
from transformers import ViltProcessor, ViltForQuestionAnswering
from torch.utils.data import DataLoader
import torch
from torch import optim, nn

SUPPORTED_MODELS = ["vilt"]
SUPPORTED_OPTIMIZERS = ["adam", "sgd"]


def create_label_mappings(answer_space_path: str) -> dict:
    """Creates id2label and label2id mapping from the answer space"""
    assert os.path.exists(answer_space_path), f"{answer_space_path} Not Found"
    logger.info(f"Creating Label Mappings")
    with open(answer_space_path, "r") as f:
        answer_space = f.readlines()
    answer_space = [ans.strip() for ans in answer_space]
    label2id = {label: idx for idx, label in enumerate(answer_space)}
    id2label = {v: k for k, v in label2id.items()}
    mappings = {"label2id": label2id, "id2label": id2label}
    logger.info(f"LABEL2ID Size = {len(mappings.get('label2id'))}")
    logger.info(f"ID2LABEL Size = {len(mappings.get('id2label'))}")
    return mappings


def create_dataset(train_df, eval_df, label2id, id2label, image_dir, processor):
    # Prepare train and validation annotations
    train_annotations = prepare_annotations(data_df=train_df, label2id=label2id)
    eval_annotations = prepare_annotations(data_df=eval_df, label2id=label2id)

    # Create train and validation dataset
    dataset = {
        mode: VQADataset(
            annotations=anno,
            processor=processor,
            image_dir=image_dir,
            id2label=id2label,
        )
        for mode, anno in [("train", train_annotations), ("eval", eval_annotations)]
    }
    for mode in dataset.keys():
        logger.info(f"Mode : {mode}, Size : {len(dataset[mode])}")
    return dataset


def get_preprocessor(model_name):
    if model_name.lower() == "vilt":
        processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    else:
        raise ValueError(
            f"{model_name} not supported. Supported Models are {SUPPORTED_MODELS}"
        )
    return processor


def create_dataloaders(dataset, processor, batch_size):
    dataloaders = {
        mode: DataLoader(
            dataset[mode],
            collate_fn=lambda batch: collate_fn(batch, processor),
            batch_size=batch_size,
            shuffle=True,
        )
        for mode in dataset.keys()
    }
    logger.info(f"{dataloaders.keys()}")
    return dataloaders


def create_model(model_name, freeze_layers, freeze_embeddings, label_mappings, pretrained=None):
    if model_name.lower() == "vilt":
        model = ViltForQuestionAnswering.from_pretrained(
            "dandelin/vilt-b32-mlm",
            id2label=label_mappings["id2label"],
            label2id=label_mappings["label2id"],
        )

        if pretrained is not None and os.path.exists(pretrained):
            logger.info("Loading Pretrained Model")
            state_dict = torch.load(pretrained)
            logger.info(
                f"Pretrained Model => Accuracy : {state_dict['best_epoch_acc']} Epoch : {state_dict['best_epoch']}"
            )
            model.load_state_dict(state_dict['state_dict'])
            logger.info("Model Loaded Successfully")
        else:
            logger.info("Pretrained Model Path is None or not found")

        if freeze_embeddings > 0:
            logger.info("Freezing Embedding Layers")
            for param in model.vilt.embeddings.parameters():
                param.requires_grad = False 

        logger.info(f"Freezing {freeze_layers} Layers")
        for layer in model.vilt.encoder.layer[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False 
    else:
        raise ValueError(
            f"{model_name} not supported. Supported Models are {SUPPORTED_MODELS}"
        )
    return model


def get_optimizer(model, cfg):
    if cfg.optimizer.lower() == "adam":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                            factor=0.3, 
                                                            patience=cfg.lr_patience,
                                                            verbose=True)
    elif cfg.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum)
        lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer=optimizer, 
                                                   base_lr=cfg.base_lr,
                                                   max_lr=cfg.learning_rate)
    else:
        raise ValueError(
            f"{cfg.optimizer} not supported. Supported optimizers are {SUPPORTED_OPTIMIZERS}"
        )
    return optimizer, lr_scheduler


def save_model(
    model, optimizer, epoch_loss, epoch_acc, epoch, label_mappings, save_dir
):
    save_dict = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_epoch": epoch,
        "best_epoch_loss": epoch_loss,
        "best_epoch_acc": epoch_acc,
        "label2id": label_mappings["label2id"],
        "id2label": label_mappings["id2label"],
    }

    model_name = f"model_state_{epoch}.pth"
    torch.save(save_dict, os.path.join(save_dir, model_name))



class EarlyStopping:

    def __init__(self, patience, model_save_path, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.model_save_path = model_save_path
        self.counter = 0
        self.min_validation_loss = np.inf
        self.best_epoch = 0
        self.early_stop = False


    def __call__(self, epoch, model, validation_loss):
        delta_loss = self.min_validation_loss - validation_loss
        # Check if val loss is smaller than min loss
        if delta_loss > self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
            # Save best model
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.model_save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early Stopping.")
                print(f"Save best model at epoch {self.best_epoch}")
                self.early_stop = True


class Config:
    def __init__(self, config):
        for key,value in config.items():
            setattr(self, key, value)