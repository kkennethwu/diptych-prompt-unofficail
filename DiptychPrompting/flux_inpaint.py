import os
import sys
sys.path.append(os.path.abspath("/home_nfs/kkennethwu_nldap/ColegaAI/FoodQualityEnhancement/DiptychPrompting/FLUX-Controlnet-Inpainting"))

import torch
from diffusers.utils import load_image, check_min_version
from natsort import natsorted
from argparse import ArgumentParser

from controlnet_flux import FluxControlNetModel
from transformer_flux import FluxTransformer2DModel
from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
check_min_version("0.30.2")




def DiptychPromptingInpaint(image_path, mask_path, subject_name="SKS", target_text_prompt="", save_path=None):
    

    prompt=f"A diptych with two side-by-side images of same {subject_name}. On the left, a photo of {subject_name}. On the right, replicate this {subject_name} but as {target_text_prompt}."
    print(f"Prompt: {prompt}")
    
    image = load_image(image_path).convert("RGB")
    mask = load_image(mask_path).convert("RGB")
    original_size = image.size  
    size = (1536, 768)
    image = image.resize(size)
    mask = mask.resize(size)

    generator = torch.Generator(device="cuda").manual_seed(24)

    # Inpaint
    result = pipe(
        prompt=prompt,
        height=size[1],
        width=size[0],
        control_image=image,
        control_mask=mask,
        num_inference_steps=30,
        generator=generator,
        controlnet_conditioning_scale=0.95,
        guidance_scale=3.5,
        negative_prompt="",
        true_guidance_scale=3.5 # default: 3.5 for alpha and 1.0 for beta
    ).images[0]

    result = result.resize(original_size)
    result.save(save_path)
    print("Successfully inpaint image")



if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--cls", "-c", type=str, default='ramen', help="class name")
    arg_parser.add_argument("--image_path", "-i", type=str, help="image path", required=True)
    arg_parser.add_argument("--target_text_prompt", "-t", type=str, default="", help="target text prompt")
    arg_parser.add_argument("--scale_factor", "-s", type=float, default=1.0, help="scale factor for enhancing the reference attention") 
    args = arg_parser.parse_args()
    
    
    cls = args.cls.replace(" ", "_").lower()
    image_path = args.image_path
    target_text_prompt = args.target_text_prompt
    
    
    ##############################
    # Step1: Build pipeline
    ##############################
    controlnet = FluxControlNetModel.from_pretrained("alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha", torch_dtype=torch.bfloat16)
    transformer = FluxTransformer2DModel.from_pretrained(
            "black-forest-labs/FLUX.1-dev", subfolder='transformer', torch_dtype=torch.bfloat16, scale_facttor=args.scale_factor
        )
    pipe = FluxControlNetInpaintingPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        controlnet=controlnet,
        transformer=transformer,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe.transformer.to(torch.bfloat16)
    pipe.controlnet.to(torch.bfloat16)

    # TODO: change to the distilled version, (faster inference speed)
    # adapter_id = "alimama-creative/FLUX.1-Turbo-Alpha"
    # pipe.load_lora_weights(adapter_id)
    # pipe.fuse_lora()
    ##############################
    # Step1: Build pipeline
    ##############################
    
    
    
    # if image_path is a dir, process all images in the dir
    if os.path.isdir(image_path):
        image_files = natsorted([os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith('.jpg')])
        for img_path in image_files:
            mask_path = img_path.replace('data_diptych', 'data_mask')
            save_path = img_path.replace('data_diptych', 'output')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            DiptychPromptingInpaint(img_path, mask_path, cls, target_text_prompt, save_path=save_path)
    else:
        mask_path = image_path.replace('data_diptych', 'data_mask')
        save_path = image_path.replace('data_diptych', 'output')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        DiptychPromptingInpaint(image_path, mask_path, cls, target_text_prompt, save_path=save_path)
                    