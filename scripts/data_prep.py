# scripts/data_prep.py
from datasets import load_dataset
import pandas as pd

def main():
    print("==== Data Preparation Started ====")

    dataset = load_dataset("srujanhj/tourism_package_dataset")
    df = dataset["train"].to_pandas()

    # Basic cleaning
    df = df.dropna()

    # Save processed file
    df.to_csv("processed_data.csv", index=False)

    print("Processed dataset saved as processed_data.csv")

if __name__ == "__main__":
    main()
