from model import ZeroDCE
from dataset import data_generator
from glob import glob
import keras
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from plot_results import plot_loss, plot_results
import config

zero_model = ZeroDCE()
zero_model.compile(learning_rate=1e-4)


def train(save_path=config.SAVE_WEIGHT_PATH, e=1):
    print("----------Loading training data----------")
    train_low_light_images = sorted(glob(config.TRAIN_PATH))
    val_low_light_images = sorted(glob(config.EVAL_PATH))

    train_dataset = data_generator(train_low_light_images)
    val_dataset = data_generator(val_low_light_images)

    print("Train Dataset:", train_dataset)
    print("Validation Dataset:", val_dataset)
    print("----------Training----------")
    history = zero_model.fit(train_dataset, validation_data=val_dataset, epochs=e)
    zero_model.save_weights(save_path)
    plot_loss(history, history.history)


def test(original_image):
    zero_model.load_weights(config.SAVE_WEIGHT_PATH)
    original_image = Image.open(original_image)
    image = keras.preprocessing.image.img_to_array(original_image)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    output_image = zero_model(image)
    output_image = tf.cast((output_image[0, :, :, :] * 255), dtype=np.uint8)
    output_image = Image.fromarray(output_image.numpy())
    # plot_results(
    #     [original_image, ImageOps.autocontrast(original_image), output_image],
    #     ["Original", "PIL Autocontrast", "Enhanced"],
    #     (8, 8),
    # )
    return output_image, original_image


if __name__ == "__main__":
    import os
    import cv2
    import matplotlib.pyplot as plt
    choice = 1
    if choice == 1:
        train()
    elif choice == 2:
        while True:
            img = np.random.choice(
                os.listdir("D:\\FPT Res Fes\\Code ref\\Zero DCE-Net\\dataset\\drive-download-20220819T064642Z-001\\trainA"))
            path = "D:\\FPT Res Fes\\Code ref\\Zero DCE-Net\\dataset\\drive-download-20220819T064642Z-001\\trainA\\" + img
            out, ori = test(path)
            img_cv2 = cv2.imread(path)
            img_cv2[:, :, 0] = cv2.equalizeHist(img_cv2[:, :, 0])
            img_cv2[:, :, 1] = cv2.equalizeHist(img_cv2[:, :, 1])
            img_cv2[:, :, 2] = cv2.equalizeHist(img_cv2[:, :, 2])

            plt.figure(figsize=(12, 12))
            plt.subplot(131)
            plt.imshow(ori)
            plt.title("input")

            plt.subplot(132)
            plt.imshow(out)
            plt.title("Zero")
            plt.subplot(133)
            plt.imshow(img_cv2[:, :, ::-1])
            plt.title("cv2")

            plt.show()
