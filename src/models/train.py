from pathlib import Path
import json
from src.data.dataloader import make_dataset
from src.models.transfer import build_transfer_model
from src.utils.config import ensure_output_dirs
from src.utils.seed import set_global_seed


def train_baseline(config: dict):
    import tensorflow as tf
    ensure_output_dirs(config); set_global_seed(config["seed"])
    p, t, d = config["paths"], config["training"], config["data"]
    train = make_dataset(Path(p["splits"]) / "train.csv", config["classes"], d["image_size"], t["batch_size"], True)
    val = make_dataset(Path(p["splits"]) / "val.csv", config["classes"], d["image_size"], t["batch_size"])
    model = build_transfer_model(t.get("architecture", "MobileNetV2"), len(config["classes"]), d["image_size"], t["dropout"], t.get("fine_tune_at"))
    model.compile(optimizer=tf.keras.optimizers.Adam(t["learning_rate"]), loss="categorical_crossentropy", metrics=["accuracy"])
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(p["model"], monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.2),
        tf.keras.callbacks.CSVLogger(str(Path(p["logs"]) / "baseline_history.csv")),
    ]
    history = model.fit(train, validation_data=val, epochs=t["epochs"], callbacks=callbacks)
    Path(p["logs"]).mkdir(parents=True, exist_ok=True)
    Path(p["logs"], "baseline_history.json").write_text(json.dumps(history.history, indent=2), encoding="utf-8")
    return model, history
