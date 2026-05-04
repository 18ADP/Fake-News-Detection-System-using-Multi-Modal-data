import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.dataset import FakeNewsDataset
from models.text_model import TextModel

# Load dataset
df = pd.read_csv("data/processed/final_balanced_data.csv")

dataset = FakeNewsDataset(df)
loader = DataLoader(dataset, batch_size=4)

# Load model
model = TextModel()

# Get one batch
batch = next(iter(loader))

input_ids = batch['input_ids']
attention_mask = batch['attention_mask']

# Forward pass
outputs = model(input_ids, attention_mask)

print("Output shape:", outputs.shape)