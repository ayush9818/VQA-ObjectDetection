import os
import pandas as pd
import torch 
from PIL import Image
from torch.utils.data import Dataset
import clip
from tqdm import tqdm 


class VizWizDataset(Dataset):
    def __init__(self, 
                df_path, 
                image_dir, 
                clip_model,
                label2id=None,
                id2label=None,
                device='cpu',
                transform=None):
        assert os.path.exists(df_path), f"{df_path} does not exists"
        self.df = pd.read_csv(df_path)

        # using only Answerable Instances
        self.df = self.df[self.df.answerable == 1]

        answer_candidates = self.df.final_answer.unique().tolist()
        if label2id is None:
            self.label2id = {ans:idx for idx,ans in enumerate(answer_candidates)}
            self.id2label = {v:k for k,v in self.label2id.items()}
        else:
            self.label2id = label2id
            self.id2label = id2label

        #self.df = self.df.iloc[:100]
        self.n_samples = self.df.shape[0]
        self.image_path = self.df["image"].apply(lambda x : os.path.join(image_dir, x))
        self.transform = transform
        self.clip_model = clip_model 
        # Initalizing Tensor to Store [Image, Text] Emneddings
        self.X = torch.empty((len(self.df), 2048), dtype=torch.float32)
        self.device=device
        for index in tqdm(range(len(self.df))):
            image = Image.open(self.image_path.iloc[index]).convert('RGB')
            if self.transform is not None:
                image = self.transform(image).unsqueeze(0).to(self.device)
            question = clip.tokenize(self.df['question'].iloc[index]).to(self.device)
            with torch.no_grad():
                    image_features = self.clip_model.encode_image(image)
                    text_features = self.clip_model.encode_text(question)
            self.X[index] = torch.cat((image_features, text_features), 1).to(torch.float32)
                
    def __getitem__(self, index):
        return index, self.X[index], self.label2id[self.df['final_answer'].iloc[index]], self.df['answerable'].iloc[index]
        
    def __len__(self):
        return self.n_samples