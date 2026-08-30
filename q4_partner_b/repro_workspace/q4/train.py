import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.client import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data_path", type=str, default="")
    args = parser.parse_args()

    mlflow.set_tracking_uri("http://localhost:5000")

    train_df = pd.read_csv(args.data_path)
    y = np.array(train_df['label'])
    X = np.array(train_df.drop(columns = ['label']))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed)

    with mlflow.start_run(run_name="q4") as run:
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("seed", args.seed)

        model = RandomForestClassifier(
            n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)
        mlflow.sklearn.log_model(model, name="model")
        run_id = run.info.run_id

    model_name = "MNIST_model"
    model_uri = f"run:/{run_id}/model"
    model_version_info = mlflow.register_model(model_uri, model_name)
    client = MlflowClient()
    client.transition_model_version_stage(name = model_name, version = model_version_info.version, stage = "Staging")


if __name__ == "__main__":
    main()
