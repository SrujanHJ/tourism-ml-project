import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
from huggingface_hub import HfApi
import os

DATA_PATH = "data/processed/train.csv"
MODEL_PATH = "best_model.pkl"
MODEL_REPO = "srujanhj/tourism_wellness_best_model"

def train_model():
    print("==== Training Model ====")

    df = pd.read_csv(DATA_PATH)

    y = df["ProdTaken"]
    X = df.drop(columns=["ProdTaken"], errors="ignore")

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    print("Model saved as best_model.pkl")

    token = os.getenv("HF_TOKEN")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=MODEL_PATH,
        path_in_repo="best_model.pkl",
        repo_id=MODEL_REPO,
        repo_type="model",
        token=token
    )

    print("Model uploaded to HuggingFace successfully")

if __name__ == "__main__":
    train_model()
