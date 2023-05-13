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

import argparse
import os
from typing import List

import numpy as np
from datasets import Dataset, Features
from datasets import Image as ImageFeature
from datasets import Value

DS_NAME = "cartoonizer-dataset"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="cartoonizer-dataset")
    parser.add_argument("--instructions_path", type=str, default="instructions.txt")
    args = parser.parse_args()
    return args


def load_instructions(instructions_path: str) -> List[str]:
    with open(instructions_path, "r") as f:
        instructions = f.readlines()
    instructions = [i.strip() for i in instructions]
    return instructions


def generate_examples(data_paths: List[str], instructions: List[str]):
    def fn():
        for data_path in data_paths:
            yield {
                "original_image": {"path": data_path[0]},
                "edit_prompt": np.random.choice(instructions),
                "cartoonized_image": {"path": data_path[1]},
            }

    return fn


def main(args):
    instructions = load_instructions(args.instructions_path)

    data_paths = os.listdir(args.data_root)
    data_paths = [os.path.join(args.data_root, d) for d in data_paths]
    new_data_paths = []
    for data_path in data_paths:
        original_image = os.path.join(data_path, "original_image.png")
        cartoonized_image = os.path.join(data_path, "cartoonized_image.png")
        new_data_paths.append((original_image, cartoonized_image))

    generation_fn = generate_examples(new_data_paths, instructions)
    print("Creating dataset...")
    ds = Dataset.from_generator(
        generation_fn,
        features=Features(
            original_image=ImageFeature(),
            edit_prompt=Value("string"),
            cartoonized_image=ImageFeature(),
        ),
    )

    print("Pushing to the Hub...")
    ds.push_to_hub(DS_NAME)


if __name__ == "__main__":
    args = parse_args()
    main(args)
