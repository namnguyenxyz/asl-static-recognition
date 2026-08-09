from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.data.dataloader import make_dataset


def evaluate_model(config: dict) -> dict:
    import tensorflow as tf
    p, d, t = config["paths"], config["data"], config["training"]
    test = make_dataset(Path(p["splits"]) / "test.csv", config["classes"], d["image_size"], t["batch_size"])
    model = tf.keras.models.load_model(p["model"])
    probability = model.predict(test, verbose=1)
    y_pred = probability.argmax(axis=1)
    manifest = pd.read_csv(Path(p["splits"]) / "test.csv")
    mapping = {label: index for index, label in enumerate(config["classes"])}
    y_true = manifest.label.map(mapping).to_numpy()
    assert len(y_true) == len(y_pred), "Test manifest and dataset order differ"
    labels = list(range(len(config["classes"])))
    report = classification_report(y_true, y_pred, labels=labels, target_names=config["classes"], output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    Path(p["metrics"]).mkdir(parents=True, exist_ok=True); Path(p["figures"]).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report).T.to_csv(Path(p["metrics"]) / "classification_report.csv")
    pd.DataFrame(matrix, index=config["classes"], columns=config["classes"]).to_csv(Path(p["metrics"]) / "confusion_matrix.csv")
    oi, zi = mapping["O"], mapping["0"]
    o_zero = {"accuracy": float(accuracy_score(y_true, y_pred)), "O_recall": report["O"]["recall"],
              "0_recall": report["0"]["recall"], "O_to_0": int(matrix[oi, zi]), "0_to_O": int(matrix[zi, oi])}
    Path(p["metrics"], "o_zero_analysis.json").write_text(json.dumps(o_zero, indent=2), encoding="utf-8")
    plt.figure(figsize=(16, 13)); sns.heatmap(matrix, cmap="Blues", xticklabels=config["classes"], yticklabels=config["classes"])
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout(); plt.savefig(Path(p["figures"]) / "confusion_matrix.png", dpi=180); plt.close()
    return o_zero
