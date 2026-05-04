import pandas as pd

df = pd.read_csv("data/processed/clean_data.csv")

print("Before balancing:")
print(df['label'].value_counts())

# Split classes
real = df[df['label'] == 0]
fake = df[df['label'] == 1]

# Find smaller class
min_size = min(len(real), len(fake))

# Sample both equally
real_balanced = real.sample(min_size, random_state=42)
fake_balanced = fake.sample(min_size, random_state=42)

# Combine and shuffle
df_balanced = pd.concat([real_balanced, fake_balanced])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nAfter balancing:")
print(df_balanced['label'].value_counts())

# Save
df_balanced.to_csv("data/processed/final_balanced_data.csv", index=False)

print("\nSaved to data/processed/final_balanced_data.csv")