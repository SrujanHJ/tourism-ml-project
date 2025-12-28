import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from huggingface_hub import HfApi, upload_file
import os


DATA_PATH = "data/processed/train.csv"

MODEL_REPO = "srujanhj/tourism_model"
MODEL_NAME = "best_model.pkl"

def train_model():
    print("==== Training Model ====")

    df = pd.read_csv(DATA_PATH)

    X = df.drop("ProdTaken", axis=1)
    y = df["ProdTaken"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Model Accuracy: {acc}")

    os.makedirs("model", exist_ok=True)
    model_path = f"model/{MODEL_NAME}"
    joblib.dump(model, model_path)

    print("==== Uploading Model to HuggingFace ====")
    upload_file(
        path_or_fileobj=model_path,
        path_in_repo=MODEL_NAME,
        repo_id=MODEL_REPO,
        repo_type="model",
    )

    print("Model uploaded successfully!")

if __name__ == "__main__":
    train_model()
