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

import tensorflow as tf
import tensorflow_datasets as tfds

tf.keras.utils.set_random_seed(0)


def load_dataset(dataset_id: str, max_num_samples: int) -> tf.data.Dataset:
    dataset = tfds.load(dataset_id, split="validation")
    dataset = dataset.shuffle(max_num_samples if max_num_samples is not None else 128)
    if max_num_samples is not None:
        print(f"Dataset will be restricted to {max_num_samples} samples.")
        dataset = dataset.take(max_num_samples)
    return dataset
