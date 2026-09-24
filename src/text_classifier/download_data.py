import os
import pandas as pd
from datasets import load_dataset

print("Downloading deepset/prompt-injections dataset...")
dataset = load_dataset("deepset/prompt-injections")

# Convert Hugging Face Dataset splits to Pandas DataFrames and combine them
df_train = pd.DataFrame(dataset["train"])
df_test = pd.DataFrame(dataset["test"])
df = pd.concat([df_train, df_test], ignore_index=True)

# Determine the directory where THIS script resides (src/text_classifier/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the target data directory inside src/text_classifier/
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Define full output path: src/text_classifier/data/prompt_injection_dataset.csv
output_path = os.path.join(DATA_DIR, "prompt_injection_dataset.csv")

# Save directly to the target folder
df.to_csv(output_path, index=False)

print(f"\nSaved to {output_path} successfully!")
print("Dataset Shape:", df.shape)
print("\nClass breakdown (0 = Safe, 1 = Injection):")
print(df["label"].value_counts())