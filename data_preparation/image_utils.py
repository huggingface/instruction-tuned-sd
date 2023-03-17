import cv2
import numpy as np
import requests
import tensorflow as tf
from PIL import Image


# Taken from
# https://github.com/SystemErrorWang/White-box-Cartoonization/blob/master/test_code/cartoonize.py#L11
def resize_crop(image: np.ndarray) -> np.ndarray:
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720 * h / w), 720
        else:
            h, w = 720, int(720 * w / h)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    h, w = (h // 8) * 8, (w // 8) * 8
    image = image[:h, :w, :]
    return image


def download_image(url: str) -> np.ndarray:
    image = Image.open(requests.get(url, stream=True).raw)
    image = image.convert("RGB")
    image = np.array(image)
    return image


def preprocess_image(image: np.ndarray) -> tf.Tensor:
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = resize_crop(image)
    image = image.astype(np.float32) / 127.5 - 1
    image = np.expand_dims(image, axis=0)
    image = tf.constant(image)
    return image


def postprocess_image(image: tf.Tensor) -> Image.Image:
    output = (image.numpy() + 1.0) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    output_image = Image.fromarray(output)
    return output_image
