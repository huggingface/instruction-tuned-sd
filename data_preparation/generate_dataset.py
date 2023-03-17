import argparse
import hashlib
import os

import model_utils
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
    # parser.add_argument("--instructions_path", type=str, default="instructions.txt")
    parser.add_argument("--max_num_samples", type=int, default=5000)
    parser.add_argument("--data_root", type=str, default="cartoonizer-dataset")
    # parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()
    return args


def load_dataset(dataset_id, max_num_samples):
    dataset = tfds.load(dataset_id, split="train")
    dataset = dataset.shuffle(max_num_samples)
    if max_num_samples is not None:
        print(f"Dataset will be restricted to {max_num_samples} samples.")
        dataset = dataset.take(max_num_samples)
    return dataset


# def load_instructions(instructions_path: str) -> List[str]:
#     with open(instructions_path, "r") as f:
#         instructions = f.readlines()
#     instructions = [i.strip() for i in instructions]
#     return instructions


# def gen_examples(
#     dataset: tf.data.Dataset, instructions: List[str], concrete_fn: Callable
# ):
#     def fn():
#         for sample in dataset.as_numpy_iterator():
#             original_image = sample["image"]
#             cartoonized_image = model_utils.perform_inference(
#                 concrete_fn, original_image
#             )
#             yield {
#                 "original_image": original_image,
#                 "edit_prompt": np.random.choice(instructions),
#                 "cartoonized_image": cartoonized_image,
#             }

#     return fn


def main(args):
    print("Loading initial dataset and the Cartoonizer model...")
    dataset = load_dataset(args.dataset_id, args.max_num_samples)
    # instructions = load_instructions(args.instructions_path)
    concrete_fn = model_utils.load_model(args.model_id)
    inference_fn = model_utils.perform_inference(concrete_fn)
    # generator_fn = gen_examples(dataset, instructions, concrete_fn)

    print("Preparing the image pairs...")
    os.makedirs(args.data_root, exist_ok=True)
    for sample in tqdm(dataset.as_numpy_iterator()):
        original_image = sample["image"]
        cartoonized_image = inference_fn(original_image)

        hash_image = hashlib.sha1(original_image.tobytes()).hexdigest()
        sample_dir = os.path.join(args.data_root, hash_image)
        os.makedirs(sample_dir)

        original_image = Image.from_array(original_image).convert("RGB")
        original_image.save(os.path.join(sample_dir, "original_image.png"))
        cartoonized_image = Image.from_array(cartoonized_image).convert("RGB")
        cartoonized_image.save(os.path.join(sample_dir, "cartoonized_image.png"))

    print(f"Total generated image-pairs: {len(os.listdir(args.data_root))}.")
    # print("Creating dataset...")
    # hub_ds = Dataset.from_generator(
    #     generator_fn,
    #     features=Features(
    #         original_image=ImageFeature(),
    #         edit_prompt=Value("string"),
    #         cartoonized_image=ImageFeature(),
    #     ),
    # )

    # if args.push_to_hub:
    #     print("Pushing to the Hub...")
    #     ds_name = f"cartoonizer-dataset"
    #     if args.max_num_samples is not None:
    #         num_samples = args.max_num_samples
    #         ds_name += f"{num_samples}-samples"
    #     hub_ds.push_to_hub(ds_name)


if __name__ == "__main__":
    args = parse_args()
    main(args)
