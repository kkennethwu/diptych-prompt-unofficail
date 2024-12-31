import os
import json
import random 
from PIL import Image
from tqdm import tqdm

classes = []
classes_text = "/home_nfs/kkennethwu_nldap/ColegaAI/FoodQualityEnhancement/food-101/meta/labels.txt"
with open(classes_text) as f:
    for line in f: 
        # replace ' ' with '_' and lowercase
        line = line.replace(" ", "_").lower()
        print(line)
        classes.append(line.strip())

train_json_path = "../food-101/meta/train.json"
test_json_path = "../food-101/meta/test.json"
image_root = "../food-101/images/"
output_root = "./data/"


for cls in tqdm(classes, desc="Processing classes"):
    os.makedirs(f"{output_root}/train/{cls}", exist_ok=True)
    os.makedirs(f"{output_root}/test/{cls}", exist_ok=True)
    
    with open(train_json_path) as f:
        train_datas = json.load(f)[cls]
        train_datas = random.sample(train_datas, 100)
        # read the image
        for train_name in train_datas:
            img = Image.open(f"{image_root}/{train_name}.jpg").convert("RGB")
            img.save(f"{output_root}/train/{train_name}.jpg")
    
    with open(test_json_path) as f:
        test_datas = json.load(f)[cls]
        test_datas = random.sample(test_datas, 50)
        # read the image
        for test_name in test_datas:
            img = Image.open(f"{image_root}/{test_name}.jpg").convert("RGB")
            img.save(f"{output_root}/test/{test_name}.jpg")

