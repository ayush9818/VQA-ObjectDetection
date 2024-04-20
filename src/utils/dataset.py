import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch 
from PIL import Image
from torch.utils.data import Dataset
import clip

class VQADataset:
    def __init__(self, cfg, type_='train'):
        assert type_ in ('train', 'eval'), f"Two type_ are supported train and eval"

        if type_ == "train":
            self.data = pd.read_csv(cfg.train_file_path)
            self.image_dir = cfg.train_image_dir
        else:
            self.data = pd.read_csv(cfg.val_file_path)
            self.image_dir = cfg.val_image_dir

        self.image_list = os.listdir(self.image_dir)

        print(f"Dataset Size : {self.data.shape[0]}")

    def fetch_data(self, idx):
        assert idx in self.data.index, f"idx value out of bounds"
        row = self.data.iloc[idx]
        
        ques = row['question']
        ans = row['final_answer']
        image_id = row['image']

        if image_id not in self.image_list:
            raise ValueError(f"{image_id} not present in image dir")

        image_path = os.path.join(self.image_dir, image_id)
        return {"question" : ques, "answer" : ans, "image_path" : image_path}
        

    def display(self, idx):
        assert idx in self.data.index, f"idx value out of bounds"
        row = self.data.iloc[idx]
        
        ques = row['question']
        ans = row['final_answer']
        image_id = row['image']

        if image_id not in self.image_list:
            raise ValueError(f"{image_id} not present in image dir")

        image_path = os.path.join(self.image_dir, image_id)
        image = cv2.imread(image_path)

        plt.figure(figsize=(10,20))
        plt.title(f"Question : {ques}\nAnswer : {ans}")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


class VizWizDataset(Dataset):
    def __init__(self, df_path, image_dir, clip_model, device='cpu', transform=None):
        assert os.path.exists(df_path), f"{df_path} does not exists"
        self.df = pd.read_csv(df_path)
        self.n_samples = self.df.shape[0]
        self.image_path = self.df["image"].apply(lambda x : os.path.join(image_dir, x))
        self.transform = transform
        self.clip_model = clip_model 
        # Initalizing Tensor to Store [Image, Text] Emneddings
        self.X = torch.empty((len(self.df), 2048), dtype=torch.float32)
        for i in range(len(self.df)):
            image = Image.open(self.image_path.iloc[i]).convert('RGB')
            if self.transform is not None:
                image = self.transform(image).unsqueeze(0).to(device)

            question = clip.tokenize(self.df['question'].iloc[i]).to(device)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image)
                text_features = self.clip_model.encode_text(question)
            self.X[i] = torch.cat((image_features, text_features), 1).to(torch.float32)
                
    def __getitem__(self, index):
        return index, self.X[index], self.df['final_answer'].iloc[index], self.df['answerable'].iloc[index]
        
    def __len__(self):
        return self.n_samples