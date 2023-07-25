from diffusers import StableDiffusionPipeline
import torch


model_path = "./sd-pencil-model"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)

pipe.to("cuda")

image = pipe(prompt="5 pencils").images[0]

image.save("output.png")