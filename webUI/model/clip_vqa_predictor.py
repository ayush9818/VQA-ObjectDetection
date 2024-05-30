from utils import Config
from transformers import ViltForQuestionAnswering, BlipProcessor, BlipForQuestionAnswering,BlipImageProcessor
import torch
from PIL import Image
from loguru import logger
import numpy as np
import os
import clip
import pickle

class VQAWithCLIPModule:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = None
        self.preprocess = None
        self.encoder = None

        self.load_clip_model()

    def load_clip_model(self):
        if Config.clip_model_path is not None and os.path.exists(Config.clip_model_path):
            logger.info("Loading CLIP Pretrained Model")

            self.clip_model, self.preprocess = clip.load("RN50", device=self.device)
            print("CLIP Loaded")

            with open(Config.encoder_path, 'rb') as f:
                encoder = pickle.load(f)
            self.encoder = encoder

            self.model = torch.load(Config.clip_model_path, map_location=torch.device(self.device)).to(torch.device(self.device))
            logger.info("Model Loaded Successfully")
            self.model.eval()
            self.model.to(self.device)
            print("Model Loaded")
        else:
            logger.info("Pretrained CLIP Model Path is None or not found")

    def predict(self, image_path, question):
        print(f"Cleaned question: {question}")

        image = Image.open(image_path)
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        question = clip.tokenize(question).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            text_features = self.clip_model.encode_text(question)
            x = torch.cat((image_features, text_features), 1).to(torch.float32)
            # Visual Question Answering
            outputs = self.model(x).squeeze(1)
            _, predicted = outputs.max(1)
            answer = self.encoder.inverse_transform(np.array(predicted.to('cpu')).reshape(-1,1))
            
            # Answerability
            return answer.item()
