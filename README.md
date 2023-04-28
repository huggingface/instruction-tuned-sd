# Instruction-tuned Cartoonization

Code for instruction-tuned cartoonization with Diffusion models.

## Data preparation

Refer to the `data_preparation` directory.

## Command for launching training

```bash
accelerate launch --mixed_precision="fp16" finetune_instruct_pix2pix.py \
    --pretrained_model_name_or_path=timbrooks/instruct-pix2pix \
    --dataset_name=sayakpaul/cartoonizer-dataset \
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
    --report_to=wandb 
```

## Inference

```bash
python cartoonize.py \
    --image_path https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png \
    --concept mountain
```

By default [`instruction-tuning-vision/instruction-tuned-cartoonizer`](https://hf.co/instruction-tuning-vision/instruction-tuned-cartoonizer) model will be used. You can also set `--model_id` to be `timbrooks/instruct-pix2pix` to use a pre-trained InstructPix2Pix model.

## Comparison across models

Refer to the `validation` directory.

## Results

<p align="center">
<img src="https://i.imgur.com/wOCjpdI.jpg"/>
</p>

<p align="center">
<img src="https://i.imgur.com/RhTG8Lf.jpg"/>
</p>

## Generalization to other types of image processing tasks

We can use the same recipe for other types of image processing tasks (such as deblurring)
as well. [instruction-tuning-vision/instruct-tuned-image-processing](https://huggingface.co/datasets/instruction-tuning-vision/instruct-tuned-image-processing) is a dataset
that consists of image-image pairs along with processing instructions like below:

<p align="center">
<img src="https://i.imgur.com/5dUbTLw.png"/>
</p>

To train a model on this dataset do:

```bash
accelerate launch --mixed_precision="fp16" finetune_instruct_pix2pix.py \
    --pretrained_model_name_or_path=timbrooks/instruct-pix2pix \
    --dataset_name=instruction-tuning-vision/instruct-tuned-image-processing \
    --original_image_column=input_image --edit_prompt_column=instruction \
    --edited_image_column=ground_truth_image \
    --use_ema \
    --enable_xformers_memory_efficient_attention \
    --resolution=256 --random_flip \
    --train_batch_size=2 --gradient_accumulation_steps=4 --gradient_checkpointing \
    --max_train_steps=15000 --checkpointing_steps=5000 --checkpoints_total_limit=1 \
    --learning_rate=5e-05 --lr_warmup_steps=0 --mixed_precision=fp16 \
    --val_image_url=https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/derain_the_image_1.png \
    --validation_prompt="Derain the image" \
    --seed=42 \
    --report_to=wandb \
    --output_dir=instruction-tuned-image-proc
```

Here are some results:

<p align="center">
<img src="https://i.imgur.com/XBkOBaX.png"/>
</p>

<p align="center">
<img src="https://i.imgur.com/lNR7mOD.png"/>
</p>


## Organization to keep track of the artifacts (datasets, models, etc.)

https://huggingface.co/instruction-tuning-vision
