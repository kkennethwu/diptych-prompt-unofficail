
## Intro
In this project we try to inplement [DiptychPrompting](https://diptychprompting.github.io/) for food quality enhancement.
I use SAM2 to segment the reference image, and enhance the reference attenstion in the  Flux-Controlnet-Inpainting.

## TO-DO
- [] manaully segmentation should be fixed with more powerful Nature Langauge Segmentation Model.
- [] Check if there is bug in the referenece attention enhancement. (the scale_factor can only be set to 1.0 for now, while the paper set it as 1.3.)
- [] Use lora-dsitilled version of Flux-Controlnet-Inpainting, which has faster inference speed.
- [] Try GPD dataset which has better visual quality in food images.

## Dataset
- [x]  Food101: https://paperswithcode.com/dataset/food-101

- [ ] GPD: https://github.com/Openning07/GPA


## Requirments
1. SAM2 
```
conda create --name same2 python=3.10
conda activate sam2
pip isntall sam2
```

2. Flux-Controlnet-Inpainting
```
conda create --name diptych python=3.10
conda activate diptych
pip install -r requirements.txt
```

I use A6000 (48G) to inference the Flux model, becasue my 4090 will run into OOM issue. 

## Starting
```
cd DiptychPrompting
# process data in food-101
python data_preprocessor.py

# generate diptych prompt
python diptych_generator.py --cls {food_cls}

# run food quality enhancement
python flux_inpaint.py -c {food_cls} -i {image_dir/image_path} -t "one the table"

```
