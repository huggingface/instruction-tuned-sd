import argparse
import hashlib
import os

import model_utils
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a dataset for InstructPix2Pix style training."
    )
    parser.add_argument(
        "--model_id", type=str, default="sayakpaul/whitebox-cartoonizer"
    )
    parser.add_argument("--dataset_id", type=str, default="imagenette")
    parser.add_argument("--max_num_samples", type=int, default=5000)
    parser.add_argument("--data_root", type=str, default="cartoonizer-dataset")
    args = parser.parse_args()
    return args


def load_dataset(dataset_id: str, max_num_samples: int) -> tf.data.Dataset:
    dataset = tfds.load(dataset_id, split="train")
    dataset = dataset.shuffle(max_num_samples if max_num_samples is not None else 128)
    if max_num_samples is not None:
        print(f"Dataset will be restricted to {max_num_samples} samples.")
        dataset = dataset.take(max_num_samples)
    return dataset


def main(args):
    print("Loading initial dataset and the Cartoonizer model...")
    dataset = load_dataset(args.dataset_id, args.max_num_samples)
    concrete_fn = model_utils.load_model(args.model_id)
    inference_fn = model_utils.perform_inference(concrete_fn)

    print("Preparing the image pairs...")
    os.makedirs(args.data_root, exist_ok=True)
    for sample in tqdm(dataset.as_numpy_iterator()):
        original_image = sample["image"]
        cartoonized_image = inference_fn(original_image)

        hash_image = hashlib.sha1(original_image.tobytes()).hexdigest()
        sample_dir = os.path.join(args.data_root, hash_image)
        os.makedirs(sample_dir)

        original_image = Image.fromarray(original_image).convert("RGB")
        original_image.save(os.path.join(sample_dir, "original_image.png"))
        cartoonized_image.save(os.path.join(sample_dir, "cartoonized_image.png"))

    print(f"Total generated image-pairs: {len(os.listdir(args.data_root))}.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
