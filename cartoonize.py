import argparse
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.utils import load_image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id", type=str, default="instruction-tuning-vision/instruction-tuned-cartoonizer"
    )
    parser.add_argument("--image_path", type=str, help="Can be a URL or a local path.")
    parser.add_argument("--prompt", type=str, default="Generate a cartoonized version of the image")
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--image_guidance_scale", type=float, default=1.5)
    parser.add_argument("--image_guidance_scale", type=float, default=7.)
    parser.add_argument("--num_images_per_prompt", type=float, default=3)
    args = parser.parse_args()
    return args

def load_pipeline(model_id):
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    pipeline.enable_xformers_memory_efficient_attention()
    return pipeline

def main(args):
    pipeline = load_pipeline(args.model_id)
    image = load_image(args.image_path)
    
    print("Generating images...")
    images = pipeline(args.prompt, image=image, num_inference_steps=args.num_inference_steps, image_guidance_scale=args.image_guidance_scale, guidance_scale=args.guidance_scale, num_images_per_prompt=args.num_images_per_prompt).images
    image_prefix = f"steps@{args.num_inference_steps}-igs@{args.image_guidance_scale}-gs@{args.guidance_scale}"
    for i, image in enumerate(images): 
        image_filename = image_prefix + "_{i}.png"
        image.save(image_filename)

