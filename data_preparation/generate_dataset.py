import argparse
import tensorflow_datasets as tfds
import tensorflow as tf
from datasets import Dataset, Features
from datasets import Image as ImageFeature
from datasets import Value

import model_utils
from typing import Callable, List
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a dataset for InstructPix2Pix style training."
    )
    parser.add_argument("--model_id", type=str, default="sayakpaul/whitebox-cartoonizer")
    parser.add_argument("--dataset_id", type=str, default="imagenette")
    parser.add_argument("--instructions_path", type=str, default="instructions.txt")
    parser.add_argument("--max_num_samples", type=int, default=5000)
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()
    return args

def load_dataset(max_num_samples=None):
    dataset = tfds.load("imagenette", split="train")
    dataset = dataset.shuffle(max_num_samples)
    if max_num_samples is not None: 
        dataset = dataset.take(max_num_samples)
    return dataset

def load_instructions(instructions_path: str) -> List[str]:
    with open(instructions_path, "r") as f:
        instructions = f.readlines()
    instructions = [i.strip() for i in instructions]
    return instructions



def gen_examples(dataset: tf.data.Dataset, instructions: List[str], concrete_fn: Callable):
    def fn():
        for sample in dataset.as_numpy_iterator():
            original_image = sample["image"]
            cartoonized_image = model_utils.perform_inference(concrete_fn, original_image)
            yield {
                "original_image": original_image,
                "edit_prompt": np.random.choice(instructions),
                "cartoonized_image": cartoonized_image,
            }

    return fn


def main(args):
    dataset = load_dataset(args.max_num_samples)
    instructions = load_instructions(args.instructions_path)
    concrete_fn = model_utils.load_model(args.model_id)
    generator_fn = gen_examples(dataset, instructions, concrete_fn)

    print("Creating dataset...")
    hub_ds = Dataset.from_generator(
        generator_fn,
        features=Features(
            original_image=ImageFeature(),
            edit_prompt=Value("string"),
            cartoonized_image=ImageFeature(),
        ),
    )

    if args.push_to_hub:
        print("Pushing to the Hub...")
        ds_name = f"cartoonizer-dataset"
        if args.max_num_samples is not None:
            num_samples = args.max_num_samples
            ds_name += f"{num_samples}-samples"
        hub_ds.push_to_hub(ds_name)


if __name__ == "__main__":
    args = parse_args()
    main(args)