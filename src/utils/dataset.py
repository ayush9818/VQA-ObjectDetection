import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2

class VQADataset:
    def __init__(self, cfg, type_='train'):
        assert type_ in ('train', 'eval'), f"Two type_ are supported train and eval"

        if type_ == "train":
            self.data = pd.read_csv(os.path.join(cfg.data_dir, cfg.train_file))
        else:
            self.data = pd.read_csv(os.path.join(cfg.data_dir, cfg.eval_file))

        self.image_dir = cfg.image_dir
        self.image_list = os.listdir(self.image_dir)

        print(f"Dataset Size : {self.data.shape[0]}")

    def fetch_data(self, idx):
        assert idx in self.data.index, f"idx value out of bounds"
        row = self.data.iloc[idx]
        
        ques = row['question']
        ans = row['answer']
        image_id = f"{row['image_id']}.png"

        if image_id not in self.image_list:
            raise ValueError(f"{image_id} not present in image dir")

        image_path = os.path.join(self.image_dir, image_id)
        return {"question" : ques, "answer" : ans, "image_path" : image_path}
        

    def display(self, idx):
        assert idx in self.data.index, f"idx value out of bounds"
        row = self.data.iloc[idx]
        
        ques = row['question']
        ans = row['answer']
        image_id = f"{row['image_id']}.png"

        if image_id not in self.image_list:
            raise ValueError(f"{image_id} not present in image dir")

        image_path = os.path.join(self.image_dir, image_id)
        image = cv2.imread(image_path)

        plt.figure(figsize=(10,20))
        plt.title(f"Question : {ques}\nAnswer : {ans}")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
