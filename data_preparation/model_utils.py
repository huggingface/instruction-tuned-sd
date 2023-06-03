#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from typing import Callable

import numpy as np
import tensorflow as tf
from huggingface_hub import snapshot_download
from PIL import Image

import image_utils


def load_model(model_id="sayakpaul/whitebox-cartoonizer"):
    model_path = snapshot_download(model_id)
    loaded_model = tf.saved_model.load(model_path)
    concrete_func = loaded_model.signatures["serving_default"]
    return concrete_func


def perform_inference(concrete_fn: Callable) -> Callable:
    def fn(image: np.ndarray) -> Image.Image:
        preprocessed_image = image_utils.preprocess_image(image)
        result = concrete_fn(preprocessed_image)["final_output:0"]
        output_image = image_utils.postprocess_image(result)
        return output_image

    return fn
