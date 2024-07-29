from math import radians, cos, sin, asin, sqrt
import requests
import random
import os
import pandas as pd
import tensorflow as tf
import os
import cv2
import numpy as np
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from sklearn.model_selection import train_test_split
from tensorflow import math as tfmath
from keras import mixed_precision
from keras.regularizers import l1
from matplotlib import pyplot as plt
from scale_labels import scale_coordinates
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.ops.numpy_ops import np_config
import time
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    Embedding,
)

import gc
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

np_config.enable_numpy_behavior()


def create_labels(file_path):
    data = pd.read_csv(file_path)
    data = data.to_numpy()
    return data


def calculate_distance(batch_1, batch_2):
    lon1 = batch_1[:, 0]
    lon2 = batch_2[:, 0]
    lat1 = batch_1[:, 1]
    lat2 = batch_2[:, 1]

    pi_on_180 = 0.017453292519943295

    lon1 = lon1 * pi_on_180
    lon2 = lon2 * pi_on_180
    lat1 = lat1 * pi_on_180
    lat2 = lat2 * pi_on_180

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        tfmath.sin(dlat / 2) ** 2
        + tfmath.cos(lat1) * tfmath.cos(lat2) * tfmath.sin(dlon / 2) ** 2
    )

    c = 2 * tfmath.asin(tfmath.sqrt(a))
    r = 6371
    return c * r


def cnn_model(input_shape):
    model = Sequential()

    # Convolutional layers
    model.add(
        Conv2D(16, (3, 3), activation="relu", input_shape=input_shape, padding="same")
    )
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    return model


MAX_LAT = 71.0
MIN_LAT = 36.0
MAX_LON = 40.0
MIN_LON = -10.0


def inverse_scale_coordinates(scaled_lat, scaled_lon):
    lat = scaled_lat * (MAX_LAT - MIN_LAT) + MIN_LAT
    lon = scaled_lon * (MAX_LON - MIN_LON) + MIN_LON

    # Stack the lat and lon tensors to create a tensor with shape (2,)
    return tf.stack([lat, lon], axis=0)


@tf.function
def custom_coordinate_loss(y_true, y_pred):
    # y_true = tf.reshape(y_true, (2,))
    # y_pred = tf.reshape(y_pred, (2,))

    # y_true = inverse_scale_coordinates(y_true[0], y_true[1])
    # y_pred = inverse_scale_coordinates(y_pred[0], y_pred[1])

    # if tf.abs(y_pred[0]) > 90 or tf.abs(y_pred[1]) > 180:
    #     penalty = 1_000_000_00

    distance = (calculate_distance(y_true, y_pred)) ** 2
    # print("-------------------------------")
    # print(tf.reduce_mean(distance))

    # print("-------------------------------")
    return tf.reduce_mean(distance)


@tf.function
def mean_absolute_distance(y_true, y_pred):
    # Extract the x and y coordinate predictions
    # y_true = tf.reshape(y_true, (2,))
    # y_pred = tf.reshape(y_pred, (2,))
    # y_true = inverse_scale_coordinates(y_true[0], y_true[1])
    # y_pred = inverse_scale_coordinates(y_pred[0], y_pred[1])

    distance = calculate_distance(y_true, y_pred)

    # print("\n")
    # print(f"y_true: {y_true}")
    # print("\n")
    # print(f"y pred: {y_pred}")
    # print("\n")
    # print(f"distance: {tf.reduce_mean(distance)}")
    # print("\n")
    return tf.reduce_mean(distance)


def preprocess_single_image(image_path, target_height=200, target_width=800):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [target_height, target_width])
    image = tf.cast(image, tf.float32) / 255.0
    return image


def create_dataset(image_folder, labels, batch_size=32):
    def image_generator(image_folder):
        all_files = os.listdir(image_folder)
        # Filter the files to include only the PNG files
        png_files = [file for file in all_files if file.endswith(".jpg")]

        for ix, filename in enumerate(png_files):
            image_path = os.path.join(image_folder, filename)
            preprocessed_image = preprocess_single_image(image_path)
            label = labels[ix]
            yield preprocessed_image, label

    image_dataset = tf.data.Dataset.from_generator(
        lambda: image_generator(image_folder),
        output_signature=(
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
            tf.TensorSpec(
                shape=(2,), dtype=tf.float32
            ),  # Change the shape to match your label shape (e.g., (2,) for x, y coordinates)
        ),
    )

    return image_dataset.batch(batch_size)

    def get_label(value, label):
        return label

    def get_value(value, label):
        return value


def read_labels(file_path):
    coordinates = []

    # Open the file in read mode
    with open(file_path, "r") as file:
        # Read each line from the file
        for line in file:
            # Split the line into latitude and longitude values based on the comma separator
            try:
                latitude, longitude = map(float, line.strip().split(","))
                # Append the coordinate pair to the 'coordinates' list
                coordinates.append((latitude, longitude))
            except:
                pass
    return coordinates


datagen = ImageDataGenerator(
    zoom_range=0.1,
    brightness_range=[0.9, 1.1],
    rotation_range=15,
    shear_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05,
    fill_mode="nearest",
)


@tf.function
def augment_data(image, label):
    augmented_image = datagen.random_transform(image)
    return augmented_image, label


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = tf.keras.Sequential(
            [
                Dense(ff_dim, activation="relu"),
                Dense(embed_dim),
            ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        # Add a sequence dimension
        inputs_expanded = tf.expand_dims(inputs, axis=1)
        attn_output = self.att(inputs_expanded, inputs_expanded)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output[:, 0, :])
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def create_transformer_regression_model(
    input_shape, num_heads=8, embed_dim=128, ff_dim=256, num_transformer_blocks=4
):
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=input_shape
    )
    base_model.trainable = False

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(embed_dim, activation="relu")(x)

    # Add multiple Transformer layers
    for _ in range(num_transformer_blocks):
        transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        x = transformer_block(x)

    # Adding fully connected layers
    x = Dense(512, activation="relu")(x)  # First fully connected layer
    x = Dense(256, activation="relu")(x)  # Second fully connected layer

    outputs = Dense(2)(x)  # Output layer

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss=custom_coordinate_loss,
        metrics=[mean_absolute_distance],
    )
    return model


def main():
    tf.config.run_functions_eagerly(True)
    path = "/home/karolito/DL/maps/big_dataset_3.tfrecord"

    devices = tf.config.list_physical_devices()

    gpu_devices = [device for device in devices if "GPU" in device.device_type]

    if gpu_devices:
        print("GPU acceleration is available. Devices:", gpu_devices)
    else:
        print("GPU acceleration is not available.")

    physical_devices = tf.config.list_physical_devices("GPU")
    for device in physical_devices:
        print(device)
        tf.config.experimental.set_memory_growth(device, True)
    dataset = tf.data.Dataset.load(path).unbatch()

    # for image, label in dataset.skip(random.randint(1, 10_000)).take(1):
    #     # Convert the image to a numpy array and ensure it's float32
    #     image_np = image.numpy().astype('float32')

    #     # Normalize the image if it's not already in the range [0, 1]
    #     if image_np.max() > 1.0:
    #         image_np /= 255.0

    #     # Remove the extra dimension if the image is grayscale
    #     if image_np.ndim == 3 and image_np.shape[-1] == 1:
    #         image_np = image_np.squeeze(-1)

    #     plt.figure()
    #     plt.imshow(image_np)
    #     plt.title(f'Label: {label.numpy()}')
    #     plt.axis('off')
    #     plt.show()

    def data_augmentation():
        data_augmentation = tf.keras.Sequential()
        data_augmentation.add(tf.keras.layers.GaussianNoise(0.2))
        # data_augmentation.add(
        #     tf.keras.layers.experimental.preprocessing.RandomContrast(0.2)
        # )
        return data_augmentation

    print(
        "cuDNN Enabled:", tf.test.is_built_with_cuda() and tf.test.is_built_with_cuda()
    )
    train_size = int(1 * 12_200)
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    del dataset
    gc.collect()

    def reshape_sample(image, target_height=224 // 2, target_width=224 * 4 // 2):
        image = tf.image.resize(image, [target_height, target_width])
        return image

    train_dataset = train_dataset.map(lambda x, y: (reshape_sample(x), y))
    test_dataset = test_dataset.map(lambda x, y: (reshape_sample(x), y))

    def expand_dimensions(x, y):
        x = tf.expand_dims(x, axis=0)  # Add a batch dimension
        return x, y

    # Apply the mapping function to your train and test datasets
    # train_dataset = train_dataset.map(expand_dimensions)
    # test_dataset = test_dataset.map(expand_dimensions)

    policy = mixed_precision.Policy("mixed_float16")
    # mixed_precision.set_global_policy(policy)

    # num_shards = 4  # Example: dividing the dataset into 4 shards
    # shard_index = 0  # Example: this instance will process the first shard

    # # Apply sharding to the datasets
    # train_dataset = train_dataset.shard(num_shards=num_shards, index=shard_index)
    # test_dataset = test_dataset.shard(num_shards=num_shards, index=shard_index)

    def expand_image(image, label):
        image *= 255
        return image, label

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # train_dataset = train_dataset.map(expand_image)
    # test_dataset = test_dataset.map(expand_image)

    batch_size = 4

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        "model_weights_epoch_{epoch:02d}.h5",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

    input_shape = (224 // 2, 224 * 4 // 2, 3)
    count = 0
    # for i, j in test_dataset:
    #     print(i)
    train_dataset = train_dataset.shuffle(buffer_size=500)
    train_dataset = train_dataset
    train_dataset = train_dataset.batch(batch_size)

    test_dataset = test_dataset.batch(batch_size)

    train_ds = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    test_ds = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    del train_dataset
    del test_dataset
    gc.collect()
    # base_model = tf.keras.applications.VGG16(
    #     include_top=False, weights="imagenet", input_shape=input_shape
    # )
    # base_model.trainable = True
    # fine_tune_at = 15

    # inputs = tf.keras.Input(shape=input_shape)

    # x = data_augmentation()(inputs)
    # # x = tf.keras.applications.resnet_v2.preprocess_input(x)

    # x = base_model(inputs, training=True)
    # x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # x = tf.keras.layers.Flatten()(x)

    # x = tf.keras.layers.Dense(
    #     1000, kernel_regularizer=tf.keras.regularizers.l1_l2(0.01)
    # )(x)

    # x = tf.keras.layers.Activation("relu")(x)
    # x = tf.keras.layers.Dropout(0.3)(x)

    # x = tf.keras.layers.Dense(
    #     500, kernel_regularizer=tf.keras.regularizers.l1_l2(0.01)
    # )(x)

    # x = tf.keras.layers.Activation("relu")(x)
    # x = tf.keras.layers.Dropout(0.3)(x)

    # x = tf.keras.layers.Dense(50, kernel_regularizer=tf.keras.regularizers.l1_l2(0.01))(
    #     x
    # )
    # x = tf.keras.layers.Activation("relu")(x)

    # outputs = tf.keras.layers.Dense(2, activation="linear")(x)

    # model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # base_model = model.layers[1]

    # print("Number of layers in the base model: ", len(base_model.layers))

    # for layer in base_model.layers[:fine_tune_at]:
    #     print(layer)
    #     layer.trainable = False

    # # normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    # # train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    # # test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

    # # for i, j in test_dataset.unbatch().take(5):
    # #     plt.imshow(i)
    # #     plt.show()
    # model.summary()
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)
    # model.load_weights("model_weights_epoch_01.h5", by_name=True, skip_mismatch=True)

    model = create_transformer_regression_model(input_shape)
    model.summary()

    # model.compile(
    #     optimizer=optimizer,
    #     loss=custom_coordinate_loss,
    #     metrics=[mean_absolute_distance],
    # )

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=1,
        callbacks=[checkpoint_callback],
        batch_size=batch_size,
    )

    # model.save("model.model")

    distances = []

    # Iterate through the test dataset and calculate distances
    # for image, label in test_dataset.unbatch():
    #     prediction = model.predict(tf.expand_dims(image, axis=0))[0]
    #     distance = calculate_distance(prediction, label)
    #     distances.append(distance)

    # # Plot the distribution of distances as a histogram
    # plt.hist(distances, bins=30)  # Adjust the number of bins as needed
    # plt.xlabel("Distance")
    # plt.ylabel("Frequency")
    # plt.title("Distribution of Distances between Predicted and True Coordinates")
    # plt.show()
    # coords = read_labels("europe_labels.txt")
    # dataset = create_dataset("/home/karolito/DL/maps/street_view_images", coords)

    # dataset_save_path = "/home/karolito/DL/maps/europe.tfrecord"
    # tf.data.experimental.save(dataset, dataset_save_path)
    # def convert_to_float_16(feature, label):
    #     feature = tf.cast(feature, dtype=tf.float16)
    #     return feature, label

    # dataset = dataset.map(convert_to_float_16)
    # # for i, j in dataset:
    # #     print(i)
    # new_path = "/home/karolito/DL/maps/europe_2.tfrecord"

    # tf.data.experimental.save(dataset, new_path)


if __name__ == "__main__":
    main()
