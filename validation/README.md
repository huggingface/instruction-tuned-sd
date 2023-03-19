This directory provides utilities to visually compare the results of different models:

* [sayakpaul/whitebox-cartoonizer](https://hf.co/sayakpaul/whitebox-cartoonizer) (TensorFlow)
* [instruction-tuning-vision/instruction-tuned-cartoonizer](https://hf.co/sayakpaul/instruction-tuning-vision/instruction-tuned-cartoonizer)  (Diffusers)
* [timbrooks/instruct-pix2pix](https://hf.co/sayakpaul/timbrooks/instruct-pix2pix) (Diffusers)

We use the `validation` split of ImageNette for the validation purpose. Launch the following script to cartoonize 10 different samples with a specific model:

```bash
python compare_models.py --model_id sayakpaul/whitebox-cartoonizer --max_num_samples 10
```

For the Diffusers' compatible models, you can additionally specify the following options:

* prompt
* num_inference_steps
* image_guidance_scale
* guidance_scale

After the samples have been generated, they should be serialized in the following structure:

[TODO]