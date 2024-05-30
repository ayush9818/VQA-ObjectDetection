from utils import Config
from transformers import ViltProcessor, ViltForQuestionAnswering
import torch
from PIL import Image
from loguru import logger
import numpy as np
import os
import csv

class VQAWithVILTModule:
    def __init__(self, device='cpu'):
        self.device = device 
        self.model = None
        self.preprocess = None
        self.encoder = None

        self.load_vilt_model()

    def load_mappings(self):
        with open(Config.classmapping_dir, "r") as f:
            next(f)  # Skip the header
            reader = csv.reader(f, skipinitialspace=True)
            class_mapping = dict(reader)
            label2id = {k: int(v) for k, v in class_mapping.items()}
            id2label = {v: k for k, v in label2id.items()}
            return label2id, id2label

    def load_vilt_model(self):
        if Config.vilt_model_path is not None and os.path.exists(Config.vilt_model_path):
            logger.info("Loading VILT Pretrained Model")

            self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
            self.id2label = None
            # self.label2id = None
            self.model = None

            if Config.vilt_model_path is not None and os.path.exists(Config.vilt_model_path):
                logger.info("Loading Pretrained Model")
                
                label2id, id2label = self.load_mappings()

                model_dict = torch.load(Config.vilt_model_path, map_location=torch.device(self.device))
                logger.info("Model Loaded Successfully")

                self.model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm", label2id=label2id, id2label=id2label)
                self.model.load_state_dict(model_dict)
            else:
                logger.info("Pretrained Model Path is None or not found")

            logger.info("Model Loaded Successfully")
            self.model.eval()
            self.model.to(self.device)
            print("Model Loaded")
        else:
            logger.info("Pretrained VILT Model Path is None or not found")

    def predict(self, image_path, question):
        print(f"Cleaned question: {question}")

        image = Image.open(image_path)
        encoding = self.processor(image, question, padding="max_length", truncation=True, return_tensors="pt")
        pixel_mask = self.processor.image_processor.pad(encoding['pixel_values'], return_tensors="pt")['pixel_mask']
        encoding['pixel_mask'] = pixel_mask
        encoding.to(self.device)

        with torch.no_grad():   
            outputs = self.model(**encoding)
            # logits = torch.sigmoid(outputs.logits)
            # logits = logits.detach().cpu().numpy()[0]
            # sorted_indices = np.argsort(logits)[::-1] 
            # answer = self.id2label[sorted_indices[0]]
            logits = outputs.logits
            idx = logits.argmax(-1).item()
            print('!!!!')
            print(idx)
            answer = self.model.config.id2label[idx]
            
            # Answerability
            return answer

