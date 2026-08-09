"""Backbone factory for controlled transfer-learning comparisons."""


def build_transfer_model(architecture: str, num_classes: int, image_size: int = 224,
                         dropout: float = 0.2, fine_tune_at: int | None = None):
    import tensorflow as tf
    factories = {
        "MobileNetV2": tf.keras.applications.MobileNetV2,
        "ResNet50": tf.keras.applications.ResNet50,
        "EfficientNetB0": tf.keras.applications.EfficientNetB0,
    }
    if architecture not in factories:
        raise ValueError(f"Unsupported architecture {architecture}. Choose one of {sorted(factories)}")
    backbone = factories[architecture](include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3))
    backbone.trainable = fine_tune_at is not None
    if fine_tune_at is not None:
        for layer in backbone.layers[:fine_tune_at]:
            layer.trainable = False
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    if architecture == "MobileNetV2":
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    elif architecture == "ResNet50":
        x = tf.keras.applications.resnet.preprocess_input(inputs)
    else:  # EfficientNetB0 includes its rescaling preprocessing layer.
        x = inputs
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classification")(x)
    return tf.keras.Model(inputs, outputs, name=f"{architecture.lower()}_asl")
