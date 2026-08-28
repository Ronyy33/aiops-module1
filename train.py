import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mnist-mlp-classifier")

SEED = 42

print("Loading MNIST (first run downloads ~15MB)...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(int)

rng = np.random.RandomState(SEED)
idx = rng.choice(len(X), 20000, replace=False)
X, y = X[idx], y[idx]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
CLASSES = np.unique(y)
print(f"train={len(X_train)}  val={len(X_val)}")


def train_and_log(hidden_layer_sizes=(100,), learning_rate_init=0.001,
                  batch_size=200, max_iter=20, run_name=None):

    with mlflow.start_run(run_name=run_name):

        mlflow.log_param("hidden_layer_sizes", str(hidden_layer_sizes))
        mlflow.log_param("learning_rate_init", learning_rate_init)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("seed", SEED)

        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate_init=learning_rate_init,
            batch_size=batch_size,
            random_state=SEED,
            max_iter=1,
            warm_start=True,
        )

        train_losses, val_accs = [], []

        for epoch in range(1, max_iter + 1):
            model.partial_fit(X_train, y_train, classes=CLASSES)

            train_loss = float(model.loss_)
            val_pred = model.predict(X_val)
            val_acc = float(accuracy_score(y_val, val_pred))
            train_acc = float(accuracy_score(y_train, model.predict(X_train)))
            f1 = float(f1_score(y_val, val_pred, average="macro"))

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("f1_macro", f1, step=epoch)
            mlflow.log_metric("generalization_gap", train_acc - val_acc, step=epoch)

            train_losses.append(train_loss)
            val_accs.append(val_acc)
            print(f"  epoch {epoch:02d}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")

        mlflow.log_metric("final_train_loss", train_losses[-1])
        mlflow.log_metric("final_val_accuracy", val_accs[-1])
        mlflow.log_metric("best_val_accuracy", max(val_accs))
        mlflow.log_metric("best_epoch", int(np.argmax(val_accs)) + 1)

        mlflow.set_tag("team", "data-science")
        mlflow.set_tag("model_type", "MLP")
        mlflow.sklearn.log_model(model, name="model",
                                 serialization_format="pickle")

        print(f"{run_name}: best_val_acc={max(val_accs):.4f} "
              f"@epoch {int(np.argmax(val_accs)) + 1}")
        return mlflow.active_run().info.run_id


if __name__ == "__main__":
    n = 0
    for lr in [0.0001, 0.001, 0.01]:
        for bs in [64, 256]:
            n += 1
            print(f"\n=== run {n}/6  lr={lr}  batch_size={bs} ===")
            train_and_log(
                hidden_layer_sizes=(100,),
                learning_rate_init=lr,
                batch_size=bs,
                max_iter=20,
                run_name=f"mlp-run-{n}",
            )
