import pandas as pd
from datasets import load_dataset
import os
from sklearn.model_selection import train_test_split

DATASET_REPO = "srujanhj/tourism_package_dataset"

def main():
    print("==== Loading Dataset from HuggingFace Hub ====")
    dataset = load_dataset(DATASET_REPO)
    df = dataset["train"].to_pandas()

    print("==== Cleaning ====")
    df = df.drop(columns=["CustomerID"], errors="ignore")
    df = df.dropna()

    os.makedirs("data/processed", exist_ok=True)

    train, test = train_test_split(df, test_size=0.2, random_state=42)

    train.to_csv("data/processed/train.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)

    print("Saved:")
    print(" data/processed/train.csv")
    print(" data/processed/test.csv")

if __name__ == "__main__":
    main()
