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

import argparse
import hashlib
import os

import data_utils
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image

from data_preparation import model_utils

GEN = torch.manual_seed(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="sayakpaul/whitebox-cartoonizer",
        choices=[
            "sayakpaul/whitebox-cartoonizer",
            "instruction-tuning-vision/instruction-tuned-cartoonizer",
            "timbrooks/instruct-pix2pix",
        ],
    )
    parser.add_argument("--dataset_id", type=str, default="imagenette")
    parser.add_argument("--max_num_samples", type=int, default=10)
    parser.add_argument(
        "--prompt", type=str, default="Generate a cartoonized version of the image"
    )
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--image_guidance_scale", type=float, default=1.5)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    args = parser.parse_args()
    return args


def load_pipeline(model_id: str):
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, use_auth_token=True
    ).to("cuda")
    pipeline.enable_xformers_memory_efficient_attention()
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def main(args):
    data_root = os.path.join(f"comparison-{args.model_id}")

    print("Loading validation dataset and inference model...")
    dataset = data_utils.load_dataset(args.dataset_id, args.max_num_samples)
    using_tf = False
    if "sayakpaul" in args.model_id:
        inference = model_utils.load_model(args.model_id)
        using_tf = True
        print(
            "TensorFlow model detected for inference, Diffusion-specifc parameters won't be used."
        )
    else:
        inference = load_pipeline(args.model_id)

    num_samples_to_generate = (
        args.max_num_samples
        if args.max_num_samples is not None
        else dataset.cardinality()
    )
    print(f"Generating {num_samples_to_generate} images...")
    for sample in dataset.as_numpy_iterator():
        # Result dir creation.
        concept_path = os.path.join(data_root, str(sample["label"]))
        hash_image = hashlib.sha1(sample["image"].tobytes()).hexdigest()
        image_path = os.path.join(concept_path, hash_image)
        os.makedirs(image_path, exist_ok=True)

        # Perform inference and serialize the result.
        if using_tf:
            image = model_utils.perform_inference(inference)(sample["image"])
            Image.fromarray(sample["image"]).save(os.path.join(image_path, "original.png"))
            image.save(os.path.join(image_path, "tf_image.png"))
        else:
            image = inference(
                args.prompt,
                image=Image.fromarray(sample["image"]).convert("RGB"),
                num_inference_steps=args.num_inference_steps,
                image_guidance_scale=args.image_guidance_scale,
                guidance_scale=args.guidance_scale,
                generator=GEN,
            ).images[0]
            image_prefix = f"steps@{args.num_inference_steps}-igs@{args.image_guidance_scale}-gs@{args.guidance_scale}"
            Image.fromarray(sample["image"]).save(os.path.join(image_path, "original.png"))
            image.save(os.path.join(image_path, f"{image_prefix}.png"))


if __name__ == "__main__":
    args = parse_args()
    main(args)
