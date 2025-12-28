# scripts/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from huggingface_hub import HfApi, HfFolder, Repository
import os
import tempfile

MODEL_REPO = "srujanhj/tourism_model"

def main():
    print("==== Training Model Started ====")

    df = pd.read_csv("processed_data.csv")

    feature_cols = [
        "Age",
        "CityTier",
        "NumberOfTrips",
        "Passport",
        "PitchSatisfactionScore",
        "OwnCar",
        "NumberOfFollowups",
        "DurationOfPitch"
    ]

    target = "ProdTaken"

    X = df[feature_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    model_filename = "best_model.pkl"
    joblib.dump(model, model_filename)

    print("Model Trained & Saved")

    # Upload model to HF
    api = HfApi()
    api.upload_file(
        path_or_fileobj=model_filename,
        path_in_repo=model_filename,
        repo_id=MODEL_REPO,
        repo_type="model",
        token=os.getenv("HF_TOKEN")
    )

    print(f"Model uploaded to HuggingFace Repo: {MODEL_REPO}")

if __name__ == "__main__":
    main()
