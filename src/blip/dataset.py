import torch 
import pandas as pd 
from PIL import Image 

class VQADataset(torch.utils.data.Dataset):
    def __init__(self, data_path ,text_processor, image_processor):
        self.data = pd.read_json(data_path)
        self.data = self.data[self.data.answerable == 1]
        print(self.data.shape)
        self.questions = self.data['question'].tolist()
        self.answers = self.data['final_answer'].tolist()
        self.image_paths = self.data['image_path'].tolist()
        self.text_processor = text_processor
        self.image_processor = image_processor
        self.max_length = 20
        self.image_height = 128
        self.image_width = 128

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # get image + text
        answers = self.answers[idx]
        image = Image.open(self.image_paths[idx]).convert('RGB')
        text = self.questions[idx]
        image_encoding = self.image_processor(image,
                                  do_resize=True,
                                  size=(self.image_height,self.image_width),
                                  return_tensors="pt")
        encoding = self.text_processor(text=text,padding="max_length",truncation=True,max_length = self.max_length,return_tensors="pt")
        # # remove batch dimension
        for k,v in encoding.items():
            encoding[k] = v.squeeze()
        encoding["pixel_values"] = image_encoding["pixel_values"][0]
        # # add labels
        labels = self.text_processor.tokenizer.encode(
            answers,
            max_length= self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )[0]
        encoding["labels"] = labels

        return encoding

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    # create new batch
    batch = {}
    batch['input_ids'] = torch.stack(input_ids)
    batch['attention_mask'] = torch.stack(attention_mask)
    batch['pixel_values'] = torch.stack(pixel_values)
    batch['labels'] = torch.stack(labels)

    return batch