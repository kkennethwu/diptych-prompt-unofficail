import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import CLIPProcessor, CLIPModel
import os
import numpy as np
from tqdm import tqdm

def calculate_clip_score(image_path, prompt, model_name="openai/clip-vit-base-patch32"):
    """
    Calculate CLIP score between an image and a text prompt.
    
    Args:
        image_path (str): Path to the image file
        prompt (str): Text prompt to compare against
        model_name (str): Name of the CLIP model to use
    
    Returns:
        float: CLIP score (cosine similarity between image and text embeddings)
    """
    # Load the CLIP model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    # only take the right half of the image
    image = image.crop((image.width // 2, 0, image.width, image.height))
    
    
    # Prepare inputs
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
        padding=True
    ).to(device)
    
    # Calculate features
    with torch.no_grad():
        outputs = model(**inputs)
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        # Calculate similarity
        similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
        
    return similarity.item()

def calculate_batch_clip_scores(image_dir, prompts_dict):
    """
    Calculate CLIP scores for multiple images and their corresponding prompts.
    
    Args:
        image_dir (str): Directory containing images
        prompts_dict (dict): Dictionary mapping image filenames to their prompts
    
    Returns:
        dict: Dictionary containing CLIP scores for each image
    """
    scores = {}
    for image_name, prompt in prompts_dict.items():
        image_path = os.path.join(image_dir, image_name)
        if os.path.exists(image_path):
            score = calculate_clip_score(image_path, prompt)
            scores[image_name] = score
    
    # Calculate average score
    avg_score = np.mean(list(scores.values()))
    scores['average'] = avg_score
    
    return scores

# Example usage:
if __name__ == "__main__":
    # # Single image example
    # image_path = "path/to/your/generated/image.jpg"
    # prompt = "a donut on a table"
    # score = calculate_clip_score(image_path, prompt)
    # print(f"CLIP Score for single image: {score}")
    
    # Batch processing example
    image_root = "/home_nfs/kkennethwu_nldap/ColegaAI/FoodQualityEnhancement/DiptychPrompting/output/test"
    
    
    classes = []
    for cls in os.listdir(image_root):
        classes.append(cls)

    avg = {}
    for cls in tqdm(classes, desc=f"Processing classe {cls}"):
        image_dir = os.path.join(image_root, cls)
        prompts_dict = {}
        for image_name in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_name)
            image_name = os.path.basename(image_path)
            prompts_dict[image_name] = f"a {cls} on a table"
        
        scores = calculate_batch_clip_scores(image_dir, prompts_dict)
        print("\nBatch CLIP Scores:")
        
        for image_name, score in scores.items():
            print(f"{image_name}: {score}")
            
        # print(f"Average Score: {scores['average']}")
        avg[cls] = scores['average']
        
    for k, v in avg.items():
        print(f"{k}: {v}")