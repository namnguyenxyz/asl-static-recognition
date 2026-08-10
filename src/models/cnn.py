"""A compact convolutional baseline trained from scratch."""


def build_cnn(num_classes: int, image_size: int = 128, dropout: float = 0.3):
    """Build a small CNN for RGB ASL hand crops.

    Input images are expected in the ``[0, 255]`` range, as produced by the
    dataloader. Rescaling is kept inside the model so training, evaluation and
    webcam inference share identical preprocessing.
    """
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    x = tf.keras.layers.Rescaling(1.0 / 255, name="rescale")(inputs)
    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(dropout / 2)(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="features")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classification")(x)
    return tf.keras.Model(inputs, outputs, name="cnn_asl")
