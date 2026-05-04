import pandas as pd
import os

# Load dataset
df = pd.read_csv("data/raw/multimodal_train.tsv", sep='\t')

print("Original size:", len(df))

# Keep required columns
df = df[['clean_title', '2_way_label']]
df.columns = ['text', 'label']

# Clean text
df = df.dropna(subset=['text'])
df = df[df['text'].str.strip() != ""]

# Folder where images are stored
image_folder = "data/images"

valid_data = []

# Iterate and keep only rows with existing images
for idx, row in df.iterrows():
    img_path = os.path.join(image_folder, f"{idx}.jpg")

    if os.path.exists(img_path):
        valid_data.append({
            "text": row["text"],
            "image": img_path,
            "label": int(row["label"])
        })

clean_df = pd.DataFrame(valid_data)

print("After removing missing images:", len(clean_df))

# Optional: limit dataset to 10k (recommended for now)
if len(clean_df) > 10000:
    clean_df = clean_df.sample(n=10000, random_state=42)

print("Final dataset size:", len(clean_df))

# Save cleaned dataset
output_path = "data/processed/clean_data.csv"
clean_df.to_csv(output_path, index=False)

print(f"Saved to {output_path}")