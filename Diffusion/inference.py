from diffusers import DiffusionPipeline
import torch
from PIL import Image
import os

prompts = [
    "Give me a design for a writing instrument that dispenses ink",
    "What pen should I gift my veteran grandfather?",
    "Design an executive pen",
    "How does an astronaut's pen look like?",
    "Show me a pen made of plastic",
    "Design a pen that is at least 50% steel",
    "Ballpoint pen",
    "Rollerball pen",
    "Gel pen",
    "Fountain pen",
    "Show me a pen that was used in the middle ages",
    "Design a pen fit for a king",
    "Generate a pen design for a class test",
    "Give me a pen that isn't too big between my fingers",
    "Show me a pen that is easy to hold and with a large diameter",
    "Design a pen that has a satisfying click action",
    "Show me a pen with a rubber grip for comfort",
    "Design a pen that has an arrowhead design for the clip",
    "Give me a pen that comes with an eraser"
]

prompts2 = [
    "luxury pen, great design, ergonomic, high quality, technology, future"
]

output_folder = 'output'

base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
base.to("cuda")

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

n_steps = 40
high_noise_frac = 0.8

# make sure the output directory exists
os.makedirs(output_folder, exist_ok=True)

for i, prompt in enumerate(prompts2):
    image = base(
        prompt=prompt,
        width=1920,
        height=1080,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]

    # save the image with filename as index of prompt + 1
    image.save(os.path.join(output_folder, f"21.png"))
