import os
import pandas as pd
import requests
from tqdm import tqdm

# Load filtered dataset
df = pd.read_csv("data/processed/filtered_data.csv")

os.makedirs("data/images", exist_ok=True)

def download_image(url, save_path):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except:
        return False
    return False

# Download images
success_count = 0

for idx, row in tqdm(df.iterrows(), total=len(df)):
    image_url = row['image_url']
    image_path = f"data/images/{idx}.jpg"

    if download_image(image_url, image_path):
        df.loc[idx, 'image_path'] = image_path
        success_count += 1
    else:
        df.loc[idx, 'image_path'] = None

print("Downloaded images:", success_count)

# Remove failed downloads
df = df.dropna(subset=['image_path'])

# Save updated dataset
df.to_csv("data/processed/final_data.csv", index=False)

print("Final dataset saved.")