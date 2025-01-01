from diffusers.pipelines import BlipDiffusionPipeline
from diffusers.utils import load_image
import torch
from PIL import Image
import os
from tqdm import tqdm
import numpy as np

blip_diffusion_pipe = BlipDiffusionPipeline.from_pretrained(
    "Salesforce/blipdiffusion", torch_dtype=torch.float16
).to("cuda")


image_root = "/home_nfs/kkennethwu_nldap/ColegaAI/FoodQualityEnhancement/DiptychPrompting/output/test"
    
classes = []
for cls in os.listdir(image_root):
    classes.append(cls)
    
guidance_scale = 7.5
num_inference_steps = 25
negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"

    
new_image_root = "/home_nfs/kkennethwu_nldap/ColegaAI/FoodQualityEnhancement/DiptychPrompting/data/test/"
for cls in tqdm(classes, desc=f"Processing classe {cls}"):
    
    cond_subject = cls
    tgt_subject = cls
    text_prompt_input = "on the table"
    
    image_dir = os.path.join(image_root, cls)
    output_dir = os.path.join("./output_blip/test", cls)
    os.makedirs(output_dir, exist_ok=True)    
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(new_image_root, cls, image_name)
        cond_image = Image.open(image_path).convert("RGB")
        original_size = cond_image.size
        
        output = blip_diffusion_pipe(
            text_prompt_input,
            cond_image.resize((512, 512)),
            cond_subject,
            tgt_subject,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            neg_prompt=negative_prompt,
            height=512,
            width=512,
        ).images
        # concat the output with the original image
        output_np = np.array(output[0].resize(original_size))
        cond_image_np = np.array(cond_image)
        try:
            concat_np = np.concatenate([cond_image_np, output_np], axis=1)
            Image.fromarray(concat_np).save(os.path.join(output_dir, image_name))
        except:
            print(f"Error processing {image_name}")
            breakpoint()