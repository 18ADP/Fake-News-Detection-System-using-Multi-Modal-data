import pandas as pd
import os

df = pd.read_csv("data/processed/final_balanced_data.csv")

print("Total samples:", len(df))
print("\nLabel distribution:")
print(df['label'].value_counts())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check if image paths exist
missing_images = 0

for path in df['image']:
    if not os.path.exists(path):
        missing_images += 1

print(f"\nMissing image files: {missing_images}")

# Show sample
print("\nSample rows:")
print(df.head())