from utils import Config
from transformers import ViltForQuestionAnswering, BlipProcessor, BlipForQuestionAnswering,BlipImageProcessor
import torch
from PIL import Image
from loguru import logger
import numpy as np
import os
import clip
import pickle

class VQAWithBLIPModule:
    def __init__(self, device='cpu'):
        self.device = device 
        self.model = None
        self.text_processor = None
        self.image_processor = None
        self.max_length = 20
        self.image_height = 128
        self.image_width = 128

        self.load_blip_model()

    def load_blip_model(self):
        if Config.blip_model_path is not None and os.path.exists(Config.blip_model_path):
            logger.info("Loading BLIP Pretrained Model")

            self.text_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
            self.image_processor = BlipImageProcessor.from_pretrained("Salesforce/blip-vqa-base")

            self.model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base" )

            if Config.blip_model_path is not None and os.path.exists(Config.blip_model_path):
                logger.info("Loading Pretrained Model")
                state_dict = torch.load(Config.blip_model_path, map_location=torch.device(self.device))
                self.model.load_state_dict(state_dict['state_dict'])
            else:
                logger.info("Pretrained Model Path is None or not found")

            logger.info("Model Loaded Successfully")
            self.model.eval()
            self.model.to(self.device)
            print("Model Loaded")
        else:
            logger.info("Pretrained BLIP Model Path is None or not found")

    def predict(self, image_path, question):
        print(f"Cleaned question: {question}")

        image = Image.open(image_path)
        with torch.no_grad():
            image_encoding = self.image_processor(image,
                                    do_resize=True,
                                    size=(self.image_height,self.image_width),
                                    return_tensors="pt")
            
            encoding = self.text_processor(text=question, 
                                           padding="max_length",
                                           truncation=True,
                                           max_length = self.max_length,
                                           return_tensors="pt")
            # # remove batch dimension
            for k,v in encoding.items():
                encoding[k] = v.squeeze()

            encoding["pixel_values"]  = image_encoding["pixel_values"][0]

            encoded_data = {k: v.unsqueeze(0).to(self.device) for k,v in encoding.items()}

            # forward pass
            outputs = self.model.generate(pixel_values=encoded_data['pixel_values'],
                                    input_ids=encoded_data['input_ids'])
  
            answer = self.text_processor.decode(outputs[0], skip_special_tokens=True)
        
        # Answerability
        return answer