def build_mobilenetv2(num_classes: int, image_size: int = 224, dropout: float = 0.2,
                      fine_tune_at: int | None = None):
    import tensorflow as tf
    backbone = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3)
    )
    backbone.trainable = fine_tune_at is not None
    if fine_tune_at is not None:
        for layer in backbone.layers[:fine_tune_at]:
            layer.trainable = False
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classification")(x)
    return tf.keras.Model(inputs, outputs, name="mobilenetv2_asl")
