import tensorflow_datasets as tfds 
import image_utils
import tensorflow as tf 
from huggingface_hub import snapshot_download
from typing import Callable
import numpy as np
from PIL import Image

def load_model(model_id="sayakpaul/whitebox-cartoonizer"):
    model_path = snapshot_download(model_id)
    loaded_model = tf.saved_model.load(model_path)
    concrete_func = loaded_model.signatures["serving_default"]
    return concrete_func

def perform_inference(concrete_fn: Callable, image: np.ndarray) -> Image.Image:
    preprocessed_image = image_utils.preprocess_image(image)
    result = concrete_fn(preprocessed_image)["final_output:0"]
    output_image = image_utils.postprocess_image(result)
    return output_image