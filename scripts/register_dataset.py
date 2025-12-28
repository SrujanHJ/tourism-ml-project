# scripts/register_dataset.py
import os
import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi

HF_USERNAME = "srujanhj"
DATASET_REPO = f"{HF_USERNAME}/tourism_package_dataset"

def main():
    print("==== Registering Dataset to HuggingFace Hub ====")

    # Load your dataset
    df = pd.read_csv("data/tourism.csv")

    # Convert to HF dataset
    hf_dataset = Dataset.from_pandas(df)

    # Push to hub
    hf_dataset.push_to_hub(DATASET_REPO)

    print(f"Dataset successfully uploaded to {DATASET_REPO}")

if __name__ == "__main__":
    main()
