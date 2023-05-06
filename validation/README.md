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

```bash
├── comparison-sayakpaul
│   └── whitebox-cartoonizer
│       ├── 0 -- class label 
│       │   └── 55f8f5846192691faa2f603b0c92f27fd8599fc7 -- original image hash
│       │       └── tf_image.png -- cartoonized image
│       ├── 1
│       │   ├── b8bfb2ec1a9af348ade8f467ac99e0af0fa0e937
│       │   │   └── tf_image.png
│       │   └── d23da1e9d9c39b17dacb66ddb52f290049a774a5
│       │       └── tf_image.png
│       ├── 2
│       │   └── 7e25076bd693e10ad04e3c41aa29a3258e3d0ecd
│       │       └── tf_image.png
│       ├── 3
│       │   ├── 1c43c5c5f7350b59d0c0607fd9357ed9e1b55e46
│       │   │   └── tf_image.png
│       │   └── cd4ca63c3d7913b1473937618c157c1919465930
│       │       └── tf_image.png
│       ├── 6
│       │   ├── 220b6c136d47e81b186d337e0bdd064c67532e4e
│       │   │   └── tf_image.png
│       │   └── f80589219ae2b913677ea9417962d4ab75f08c2f
│       │       └── tf_image.png
│       └── 7
│           ├── 4f33183189589bb171ba9489b898e5edbac25dfe
│           │   └── tf_image.png
│           └── 519863ade478d26b467e08dc5fb4353a6316833c
│               └── tf_image.png
```

For you use a Diffusers' compatible model then it would look like so:

```bash
├── comparison-instruction-tuning-vision
│   └── instruction-tuned-cartoonizer
│       ├── 0
│       │   └── 55f8f5846192691faa2f603b0c92f27fd8599fc7
│       │       └── steps@20-igs@1.5-gs@7.0.png 
│       ├── 1
│       │   ├── b8bfb2ec1a9af348ade8f467ac99e0af0fa0e937
│       │   │   └── steps@20-igs@1.5-gs@7.0.png
│       │   └── d23da1e9d9c39b17dacb66ddb52f290049a774a5
│       │       └── steps@20-igs@1.5-gs@7.0.png
│       ├── 2
│       │   └── 7e25076bd693e10ad04e3c41aa29a3258e3d0ecd
│       │       └── steps@20-igs@1.5-gs@7.0.png
│       ├── 3
│       │   ├── 1c43c5c5f7350b59d0c0607fd9357ed9e1b55e46
│       │   │   └── steps@20-igs@1.5-gs@7.0.png
│       │   └── cd4ca63c3d7913b1473937618c157c1919465930
│       │       └── steps@20-igs@1.5-gs@7.0.png
│       ├── 6
│       │   ├── 220b6c136d47e81b186d337e0bdd064c67532e4e
│       │   │   └── steps@20-igs@1.5-gs@7.0.png
│       │   └── f80589219ae2b913677ea9417962d4ab75f08c2f
│       │       └── steps@20-igs@1.5-gs@7.0.png
│       └── 7
│           ├── 4f33183189589bb171ba9489b898e5edbac25dfe
│           │   └── steps@20-igs@1.5-gs@7.0.png
│           └── 519863ade478d26b467e08dc5fb4353a6316833c
│               └── steps@20-igs@1.5-gs@7.0.png
```
