# scripts/register_dataset.py
import os
import pandas as pd
from datasets import Dataset

HF_USERNAME = "srujanhj"
DATASET_REPO = f"{HF_USERNAME}/tourism_package_dataset"

def main():
    print("==== Registering Dataset to HuggingFace Hub ====")

    df = pd.read_csv("data/tourism.csv")

    # Add missing columns to match existing schema
    if "__index_level_0__" not in df.columns:
        df["__index_level_0__"] = range(len(df))

    if "CustomerID" not in df.columns:
        df["CustomerID"] = range(1, len(df)+1)

    # Convert to HF dataset
    hf_dataset = Dataset.from_pandas(df)

    # Push to hub (update existing)
    hf_dataset.push_to_hub(DATASET_REPO, private=False)

    print(f"Dataset successfully uploaded to {DATASET_REPO}")

if __name__ == "__main__":
    main()
