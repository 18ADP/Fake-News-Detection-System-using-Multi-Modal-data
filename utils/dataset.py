import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import cv2

class FakeNewsDataset(Dataset):
    def __init__(self, dataframe, max_len=128):
        self.df = dataframe
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text = str(row['text'])
        img_path = row['image']
        label = int(row['label'])

        # TEXT → BERT tokenizer
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        # IMAGE → OpenCV
        image = cv2.imread(img_path)
        image = cv2.resize(image, (224, 224))
        image = image / 255.0

        image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)
        }