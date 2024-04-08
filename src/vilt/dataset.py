import warnings
warnings.filterwarnings('ignore')

import os 
import pandas as pd 
import torch
from PIL import Image

def prepare_annotations(data_df : pd.DataFrame, label2id : dict) -> dict: 
    annotations = []
    for idx,row in data_df.iterrows():
        question = row['question']
        image_id = row['image_id']
        answer = [ans.strip() for ans in row['answer'].split(',')]
        answer_count = {}
        for answer_ in answer:
            answer_count[answer_] = answer_count.get(answer_, 0) + 1
        
        labels = []
        scores = []
        for answer_ in answer_count:
            labels.append(label2id[str(answer_)])
            scores.append(1.0)
        
        annotations_dict = {
            'question' : question,
            'image_id' : image_id,
            'answer' : answer,
            'labels' : labels,
            'scores' : scores
        }
        annotations.append(annotations_dict)
    return annotations


class VQADataset(torch.utils.data.Dataset):
    """VQA (v2) dataset."""

    def __init__(self, annotations, processor, image_dir, id2label):
        self.annotations = annotations
        self.processor = processor
        self.image_dir = image_dir
        self.id2label = id2label

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # get image + text
        annotation = self.annotations[idx]
        image_id = annotation['image_id']
        image = Image.open(os.path.join(self.image_dir, f'{image_id}.png'))
        text = annotation['question']

        encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
        # remove batch dimension
        for k,v in encoding.items():
          encoding[k] = v.squeeze()
        # add labels
        labels = annotation['labels']
        scores = annotation['scores']
        # based on: https://github.com/dandelin/ViLT/blob/762fd3975c180db6fc88f577cf39549983fa373a/vilt/modules/objectives.py#L301
        targets = torch.zeros(len(self.id2label))
        for label, score in zip(labels, scores):
              targets[label] = score
        encoding["labels"] = targets
        return encoding


def collate_fn(batch, processor):
  input_ids = [item['input_ids'] for item in batch]
  pixel_values = [item['pixel_values'] for item in batch]
  attention_mask = [item['attention_mask'] for item in batch]
  token_type_ids = [item['token_type_ids'] for item in batch]
  labels = [item['labels'] for item in batch]

  # create padded pixel values and corresponding pixel mask
  encoding = processor.image_processor.pad(pixel_values, return_tensors="pt")

  # create new batch
  batch = {}
  batch['input_ids'] = torch.stack(input_ids)
  batch['attention_mask'] = torch.stack(attention_mask)
  batch['token_type_ids'] = torch.stack(token_type_ids)
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = torch.stack(labels)

  return batch

# lambda batch: my_collate(batch, arg="myarg")