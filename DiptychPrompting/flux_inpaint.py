import os
import sys
sys.path.append(os.path.abspath("/home_nfs/kkennethwu_nldap/ColegaAI/FoodQualityEnhancement/DiptychPrompting/FLUX-Controlnet-Inpainting"))

import torch
from diffusers.utils import load_image, check_min_version
from natsort import natsorted

from controlnet_flux import FluxControlNetModel
from transformer_flux import FluxTransformer2DModel
from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
check_min_version("0.30.2")



# Step1: read image
image_root = "/home_nfs/kkennethwu_nldap/ColegaAI/FoodQualityEnhancement/DiptychPrompting/data/test/ramen"


# Step2: Get object with Grounding Dino


# Step3: Generate input Diptych Prompt





# Set image path , mask path and prompt
# image_path='https://huggingface.co/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha/resolve/main/images/bucket.png'
# mask_path='https://huggingface.co/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha/resolve/main/images/bucket_mask.jpeg'
# prompt='a person wearing a white shoe, carrying a white bucket with text "FLUX" in black on it'

image_path='/home_nfs/kkennethwu_nldap/ColegaAI/FoodQualityEnhancement/DiptychPrompting/tmp/diptych_image.jpg'
mask_path='/home_nfs/kkennethwu_nldap/ColegaAI/FoodQualityEnhancement/DiptychPrompting/tmp/diptych_mask.jpg'
subject_name = "ramen"  
target_text_prompt="in the hand of a traditional Japn ramen chef"
prompt=f"A diptych with two side-by-side images of same {subject_name}. On the left, a photo of {subject_name}. On the right, replicate this {subject_name} but as {target_text_prompt}."

# Build pipeline
controlnet = FluxControlNetModel.from_pretrained("alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha", torch_dtype=torch.bfloat16)
transformer = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev", subfolder='transformer', torch_dtype=torch.bfloat16
    )
pipe = FluxControlNetInpaintingPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    controlnet=controlnet,
    transformer=transformer,
    torch_dtype=torch.bfloat16
).to("cuda")
pipe.transformer.to(torch.bfloat16)
pipe.controlnet.to(torch.bfloat16)

# adapter_id = "alimama-creative/FLUX.1-Turbo-Alpha"
# pipe.load_lora_weights(adapter_id)
# pipe.fuse_lora()

# Load image and mask
# size = (768, 768)
size = (1024, 384)
image = load_image(image_path).convert("RGB").resize(size)
mask = load_image(mask_path).convert("RGB").resize(size)
generator = torch.Generator(device="cuda").manual_seed(24)

# Inpaint
result = pipe(
    prompt=prompt,
    height=size[1],
    width=size[0],
    control_image=image,
    control_mask=mask,
    num_inference_steps=28,
    generator=generator,
    controlnet_conditioning_scale=0.9,
    guidance_scale=3.5,
    negative_prompt="",
    true_guidance_scale=1.0 # default: 3.5 for alpha and 1.0 for beta
).images[0]

result.save('output/diptych_inpaint.jpg')
print("Successfully inpaint image")


