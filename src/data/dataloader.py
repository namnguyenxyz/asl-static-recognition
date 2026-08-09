from pathlib import Path
import pandas as pd


def make_dataset(manifest_path: str | Path, classes: list[str], image_size: int, batch_size: int,
                 training: bool = False):
    import tensorflow as tf
    frame = pd.read_csv(manifest_path)
    lookup = tf.keras.layers.StringLookup(vocabulary=classes, num_oov_indices=0)
    paths, labels = frame.processed_path.astype(str).values, frame.label.astype(str).values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(frame), reshuffle_each_iteration=True)
    def load(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [image_size, image_size])
        # Models apply their own architecture-specific ImageNet preprocessing.
        image = tf.cast(image, tf.float32)
        return image, tf.one_hot(lookup(label), len(classes))
    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
