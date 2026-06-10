"""Model definition (MobileNetV2 transfer learning) and load helper."""

import tensorflow as tf

from src.data_loader import get_augmentation_layer
from src.utils import IMG_SIZE


def build_model(num_classes: int) -> tf.keras.Model:
    """Build a MobileNetV2-based transfer learning model for fine-tuning."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = get_augmentation_layer()(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_model(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)
