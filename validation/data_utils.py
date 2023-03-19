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
