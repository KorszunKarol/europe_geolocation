import tensorflow as tf
from tensorflow import keras
from maps import (
    model,
    preprocess_single_image,
    custom_coordinate_loss,
    mean_absolute_distance,
)
import os
from stitch_images import image_to_panorama
import cv2
import numpy as np
from matplotlib import pyplot as plt
from maps import TransformerBlock, create_transformer_regression_model
custom_model = tf.keras.applications.ResNet50V2(
    input_shape=(640, 120, 3), include_top=False, weights="imagenet"
)


def sort_images(image_name):
    int(image_name)


def visualize_cam(image, model):
    last_conv_layer_name = "conv5_block3_out"
    output = model(image)
    predicted_class_index = tf.argmax(output, axis=1)[0]

    grad_model = tf.keras.models.Model(
        model.inputs,
        [
            model.get_layer("resnet50v2").get_layer(last_conv_layer_name).output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(image)
        class_channel = preds[:, predicted_class_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[1]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(image[0], 0.6, heatmap, 0.4, 0)

    cv2.imshow("Original Image", image[0])
    cv2.imshow("Heatmap", heatmap)
    cv2.imshow("Superimposed Image", superimposed_img)
    cv2.waitKey(0)


def main():
    input_shape = (400, 1600, 3)

    folder_path = "/home/karolito/DL/maps/test_ims"
    im_files = []
    model = tf.keras.models.load_model(
        "model.model",
        custom_objects={
            "custom_coordinate_loss": custom_coordinate_loss,
            "mean_absolute_distance": mean_absolute_distance,
        },
    )

    for file in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file)
        image = cv2.imread(file_path)
        im_files.append(image)

    panorama = image_to_panorama(im_files[0], im_files[1], im_files[2], im_files[3])

    new_width = 640
    new_height = 120
    resized_image = cv2.resize(panorama, (new_width, new_height))
    resized_image = resized_image.astype(np.float32)

    resized_image = tf.convert_to_tensor(resized_image)
    resized_image = tf.keras.applications.resnet_v2.preprocess_input(resized_image)

    print(resized_image)
    resized_image = tf.expand_dims(resized_image, axis=0)
    prediction = model.predict(resized_image)
    print(prediction)
    # visualize_cam(resized_image, model)


if __name__ == "__main__":
    main()
