This directory provides utilities to create a Cartoonizer dataset for [InstructPix2Pix](https://arxiv.org/abs/2211.09800) like training. 

## Steps

We used 5000 randomly sampled images as the original images from the `train` set of [ImageNette](https://www.tensorflow.org/datasets/catalog/imagenette). To derive their
cartoonized renditions, we used the [Whitebox Cartoonizer model](https://huggingface.co/sayakpaul/whitebox-cartoonizer). For deriving the `instructions.txt` file, we used [ChatGPT](https://chat.openai.com/). In particular, we used the following prompt: 

> Provide al teast 50 synonymous sentences for the following instruction: "Cartoonize the following image."

Dataset preparation is divided into three steps:

### Step 0: Install dependencies

```bash
pip install -q requirements.txt
```

### Step 1: Obtain the image-cartoon pairs

```bash
python generate_dataset.py
```

If you want to use more than 5000 samples, specify the `--max_num_samples` option. One the image-cartoon pairs are generated, you should see a directory called `cartoonizer-dataset` directory (unless you specified a different one via `--data_root`): 

<p align="center">
<img src="https://i.imgur.com/jHaAPWa.png" width=500/>
</p>

### Step 2: Export the dataset to 🤗 Hub

For this step, you need to be authorized to access your Hugging Face account. Run the following command to do so:

```bash
huggingface-cli login
```

Then run:

```python
python export_to_hub.py
```

You can find a mini dataset [here](https://huggingface.co/datasets/instruction-tuning-vision/cartoonizer-dataset):

<p align="center">
<img src="https://i.imgur.com/QncO0BQ.png" width=500/>
</p>
