import torch.nn as nn 
from loguru import logger
from utils import Config
import os
import torch
import clip
from PIL import Image
import numpy as np
import pickle
import torch.nn.functional as F

class ADModule:
    def __init__(self, device='cpu'):
        self.device = device 
        self.load_model()
    
    def load_model(self):
        if Config.ans_model_path is not None and os.path.exists(Config.ans_model_path):
            logger.info("Loading Answerability Detection Pretrained Model")
            self.clip_model, self.preprocesser = clip.load("RN50", device=self.device)
            print("Clip Loaded")

            self.model = torch.load(Config.ans_model_path, map_location=torch.device(self.device)).to(torch.device(self.device))
            logger.info("Model Loaded Successfully")
            self.model.eval()  # Set model to evaluation mode
            self.model.to(self.device)
            print("Model Loaded")
        else:
            logger.info("Pretrained Answerability Detection Model Path is None or not found")

    def predict(self, image_path, question):        
        print(f"Cleaned question: {question}")

        image = Image.open(image_path)
        image = self.preprocesser(image).unsqueeze(0).to(self.device)
        question = clip.tokenize(question).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            text_features = self.clip_model.encode_text(question)
            x = torch.cat((image_features, text_features), 1).to(torch.float32)
            # Visual Question Answerability
            sigmoid_activation = nn.Sigmoid()
            outputs = sigmoid_activation(self.model(x))
    
        # Answerability
        return outputs.item()
