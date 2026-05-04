import pandas as pd
from torch.utils.data import DataLoader
from utils.dataset import FakeNewsDataset

# Load dataset
df = pd.read_csv("data/processed/final_balanced_data.csv")

dataset = FakeNewsDataset(df)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Get one batch
batch = next(iter(loader))

print("input_ids shape:", batch['input_ids'].shape)
print("attention_mask shape:", batch['attention_mask'].shape)
print("image shape:", batch['image'].shape)
print("labels:", batch['label'])