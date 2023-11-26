# Instruction-tuning Stable Diffusion

**TL;DR**: Motivated partly by [FLAN](https://arxiv.org/abs/2109.01652) and partly by [InstructPix2Pix](https://arxiv.org/abs/2211.09800), we explore a way to instruction-tune [Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release). This allows us to prompt our model using an input image and an “instruction”, such as - *Apply a cartoon filter to the natural image*.

You can read [our blog post](https://hf.co/blog/instruction-tuning-sd) to know more details. 

## Table of contents

🐶 [Motivation](#motivation) <br>
📷 [Data preparation](#data-preparation) <br>
💺 [Training](#training) <br>
🎛 [Models, datasets, demo](#models-datasets-demo) <br>
⭐️ [Inference](#inference) <br>
🧭 [Results](#results) <br>
🤝 [Acknowledgements](#acknowledgements) <br>

## Motivation 

Instruction-tuning is a supervised way of teaching language models to follow instructions to solve a task. It was introduced in [Fine-tuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652) (FLAN) by Google. From recent times, you might recall works like [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) and [FLAN V2](https://arxiv.org/abs/2210.11416), which are good examples of how beneficial instruction-tuning can be for various tasks. 

On the other hand, the idea of teaching Stable Diffusion to follow user instructions to perform edits on input images was introduced in [InstructPix2Pix: Learning to Follow Image Editing Instructions](https://arxiv.org/abs/2211.09800). 

Our motivation behind this work comes partly from the FLAN line of works and partly from InstructPix2Pix. We wanted to explore if it’s possible to prompt Stable Diffusion with specific instructions and input images to process them as per our needs. 

<p align="center">
<img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/instruction-tuning-sd.png" width=600/>
</p>

Our main idea is to first create an instruction prompted dataset (as described in [our blog](https://hf.co/blog/instruction-tuning-sd) and then conduct InstructPix2Pix style training. The end objective is to make Stable Diffusion better at following specific instructions that entail image transformation related operations.


## Data preparation

Our data preparation process is inspired by FLAN. Refer to the sections below for more details.

* **Cartoonization**: Refer to the `data_preparation` directory.
* **Low-level image processing**: Refer to the [dataset card](https://huggingface.co/datasets/instruction-tuning-sd/low-level-image-proc).

## Training

### Dev env setup

We recommend using a Python virtual environment for this. Feel free to use your favorite one here. 

We conducted our experiments with PyTorch 1.13.1 (CUDA 11.6) and a single A100 GPU. Since PyTorch installation can be hardware-dependent, we refer you to the [official docs](https://pytorch.org/) for installing PyTorch. 

Once PyTorch is installed, we can install the rest of the dependencies:

```bash 
pip install -r requirements.txt
```

Additionally, we recommend installing [xformers](https://github.com/facebookresearch/xformers) as well for enabling memory-efficient training.

> 💡 **Note**: If you're using PyTorch 2.0 then you don't need to additionally install xformers. This is because we default to a memory-efficient attention processor in Diffusers when PyTorch 2.0 is being used. 

### Launching training

Our training code leverages [🧨 diffusers](https://github.com/huggingface/diffusers), [🤗 accelerate](https://github.com/huggingface/accelerate), and [🤗 transformers](https://github.com/huggingface/transformers). In particular, we extend [this training example](https://github.com/huggingface/diffusers/blob/main/examples/instruct_pix2pix/train_instruct_pix2pix.py) to fit our needs. 

### Cartoonization

#### Training from scratch using the InstructPix2Pix methodology

```bash 
export MODEL_ID="runwayml/stable-diffusion-v1-5"
export DATASET_ID="instruction-tuning-sd/cartoonization"
export OUTPUT_DIR="cartoonization-scratch"

accelerate launch --mixed_precision="fp16" train_instruct_pix2pix.py \
  --pretrained_model_name_or_path=$MODEL_ID \
  --dataset_name=$DATASET_ID \
  --use_ema \
  --enable_xformers_memory_efficient_attention \
  --resolution=256 --random_flip \
  --train_batch_size=2 --gradient_accumulation_steps=4 --gradient_checkpointing \
  --max_train_steps=15000 \
  --checkpointing_steps=5000 --checkpoints_total_limit=1 \
  --learning_rate=5e-05 --lr_warmup_steps=0 \
  --mixed_precision=fp16 \
  --val_image_url="https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png" \
  --validation_prompt="Generate a cartoonized version of the natural image" \
  --seed=42 \
  --output_dir=$OUTPUT_DIR \
  --report_to=wandb \
  --push_to_hub
```

> 💡 **Note**: Following InstructPix2Pix, we train on the 256x256 resolution and that doesn't seem to affect the end quality too much when we perform inference with the 512x512 resolution.

Once the training successfully launched, the logs will be automatically tracked using Weights and Biases. Depending on how you specified the `checkpointing_steps` and the `max_train_steps`, there will be intermediate checkpoints too. At the end of training, you can expect a directory (namely `OUTPUT_DIR`) that contains the intermediate checkpoints and the final pipeline artifacts. 

If `--push_to_hub` is specified, the contents of `OUTPUT_DIR` will be pushed to a repository on the Hugging Face Hub. 

[Here](https://wandb.ai/sayakpaul/instruction-tuning-sd/runs/wszjpb1b) is an example run page on Weights and Biases. [Here](https://huggingface.co/instruction-tuning-sd/scratch-cartoonizer) is an example of how the pipeline repository would look like on the Hugging Face Hub. 

#### Fine-tuning from InstructPix2Pix

```bash 
export MODEL_ID="timbrooks/instruct-pix2pix"
export DATASET_ID="instruction-tuning-sd/cartoonization"
export OUTPUT_DIR="cartoonization-finetuned"

accelerate launch --mixed_precision="fp16" finetune_instruct_pix2pix.py \
  --pretrained_model_name_or_path=$MODEL_ID \
  --dataset_name=$DATASET_ID \
  --use_ema \
  --enable_xformers_memory_efficient_attention \
  --resolution=256 --random_flip \
  --train_batch_size=2 --gradient_accumulation_steps=4 --gradient_checkpointing \
  --max_train_steps=15000 \
  --checkpointing_steps=5000 --checkpoints_total_limit=1 \
  --learning_rate=5e-05 --lr_warmup_steps=0 \
  --mixed_precision=fp16 \
  --val_image_url="https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png" \
  --validation_prompt="Generate a cartoonized version of the natural image" \
  --seed=42 \
  --output_dir=$OUTPUT_DIR \
  --report_to=wandb \
  --push_to_hub
```

### Low-level image processing

#### Training from scratch using the InstructPix2Pix methodology

```bash 
export MODEL_ID="runwayml/stable-diffusion-v1-5"
export DATASET_ID="instruction-tuning-sd/low-level-image-proc"
export OUTPUT_DIR="low-level-img-proc-scratch"

accelerate launch --mixed_precision="fp16" train_instruct_pix2pix.py \
  --pretrained_model_name_or_path=$MODEL_ID \
  --dataset_name=$DATASET_ID \
  --original_image_column="input_image" \
  --edit_prompt_column="instruction" \
  --edited_image_column="ground_truth_image" \
  --use_ema \
  --enable_xformers_memory_efficient_attention \
  --resolution=256 --random_flip \
  --train_batch_size=2 --gradient_accumulation_steps=4 --gradient_checkpointing \
  --max_train_steps=15000 \
  --checkpointing_steps=5000 --checkpoints_total_limit=1 \
  --learning_rate=5e-05 --lr_warmup_steps=0 \
  --mixed_precision=fp16 \
  --val_image_url="https://hf.co/datasets/sayakpaul/sample-datasets/resolve/main/derain_the_image_1.png" \
  --validation_prompt="Derain the image" \
  --seed=42 \
  --output_dir=$OUTPUT_DIR \
  --report_to=wandb \
  --push_to_hub
```

#### Fine-tuning from InstructPix2Pix

```bash 
export MODEL_ID="timbrooks/instruct-pix2pix"
export DATASET_ID="instruction-tuning-sd/low-level-image-proc"
export OUTPUT_DIR="low-level-img-proc-finetuned"

accelerate launch --mixed_precision="fp16" finetune_instruct_pix2pix.py \
  --pretrained_model_name_or_path=$MODEL_ID \
  --dataset_name=$DATASET_ID \
  --original_image_column="input_image" \
  --edit_prompt_column="instruction" \
  --edited_image_column="ground_truth_image" \
  --use_ema \
  --enable_xformers_memory_efficient_attention \
  --resolution=256 --random_flip \
  --train_batch_size=2 --gradient_accumulation_steps=4 --gradient_checkpointing \
  --max_train_steps=15000 \
  --checkpointing_steps=5000 --checkpoints_total_limit=1 \
  --learning_rate=5e-05 --lr_warmup_steps=0 \
  --mixed_precision=fp16 \
  --val_image_url="https://hf.co/datasets/sayakpaul/sample-datasets/resolve/main/derain_the_image_1.png" \
  --validation_prompt="Derain the image" \
  --seed=42 \
  --output_dir=$OUTPUT_DIR \
  --report_to=wandb \
  --push_to_hub
```

## Models, datasets, demo

### **Models**:
  * [instruction-tuning-sd/scratch-low-level-img-proc](https://huggingface.co/instruction-tuning-sd/scratch-low-level-img-proc)
  * [instruction-tuning-sd/scratch-cartoonizer](https://huggingface.co/instruction-tuning-sd/scratch-cartoonizer)
  * [instruction-tuning-sd/cartoonizer](https://huggingface.co/instruction-tuning-sd/cartoonizer)
  * [instruction-tuning-sd/low-level-img-proc](https://huggingface.co/instruction-tuning-sd/low-level-img-proc)

### **Datasets**:
  * [Instruction-prompted cartoonization](https://huggingface.co/datasets/instruction-tuning-sd/cartoonization)
  * [Instruction-prompted low-level image processing](https://huggingface.co/datasets/instruction-tuning-sd/low-level-image-proc) 

### Demo on 🤗 Spaces

Try out the models interactively WITHOUT any setup: [Demo](https://huggingface.co/spaces/instruction-tuning-sd/instruction-tuned-sd)

## Inference

### Cartoonization

```python
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.utils import load_image

model_id = "instruction-tuning-sd/cartoonizer"
pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, use_auth_token=True
).to("cuda")

image_path = "https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"
image = load_image(image_path)

image = pipeline("Cartoonize the following image", image=image).images[0]
image.save("image.png")
```

### Low-level image processing 

```python 
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.utils import load_image

model_id = "instruction-tuning-sd/low-level-img-proc"
pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, use_auth_token=True
).to("cuda")

image_path = "https://hf.co/datasets/sayakpaul/sample-datasets/resolve/main/derain%20the%20image_1.png"
image = load_image(image_path)

image = pipeline("derain the image", image=image).images[0]
image.save("image.png")
```


> 💡 **Note**: Since the above pipelines are essentially of type `StableDiffusionInstructPix2PixPipeline`, you can customize several arguments that
the pipeline exposes. Refer to the [official docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/pix2pix) for more details.

## Results

### Cartoonization

<p align="center">
<img src="https://i.imgur.com/wOCjpdI.jpg"/>
</p>

---

<p align="center">
<img src="https://i.imgur.com/RhTG8Lf.jpg"/>
</p>

### Low-level image processing

<p align="center">
<img src="https://i.imgur.com/LOhcJLv.jpg"/>
</p>

---

<p align="center">
<img src="https://i.imgur.com/uhTqIpY.png"/>
</p>

Refer to our [blog post](https://hf.co/blog/instruction-tuning-sd) for more discussions on results and open questions.  


## Acknowledgements

Thanks to [Alara Dirik](https://www.linkedin.com/in/alaradirik/) and [Zhengzhong Tu](https://www.linkedin.com/in/zhengzhongtu) for the helpful discussions. 

## Citation

```bibtex
@article{
  Paul2023instruction-tuning-sd,
  author = {Paul, Sayak},
  title = {Instruction-tuning Stable Diffusion with InstructPix2Pix},
  journal = {Hugging Face Blog},
  year = {2023},
  note = {https://huggingface.co/blog/instruction-tuning-sd},
}
```

