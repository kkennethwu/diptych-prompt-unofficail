import os
from argparse import ArgumentParser
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseEvent
from natsort import natsorted
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_checkpoint = "./sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "./configs/sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device='cuda')

predictor = SAM2ImagePredictor(sam2_model)

np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

def gen_seg_sam2(image_path, save_root):
    img = Image.open(image_path)
    img = np.array(img.convert("RGB"))
    img_name = os.path.basename(image_path)
    

    # Initialize the plot for displaying image and capturing clicks
    plt.figure(figsize=(9, 6))
    plt.title(f"Click to add points for frame {img_name}")
    plt.imshow(img)
    plt.axis('off')
    
    # set img predicter
    predictor.set_image(img)
    
    # Wait for user to finish clicking
    points = []
    labels = []

    def on_click(event: MouseEvent):
        if event.inaxes is None:
            return  # Ignore clicks outside of the axes
        
        # Get the coordinates of the click
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        print(f"Point clicked at x={x:.2f}, y={y:.2f}")

        # Ask for the label (positive click)
        label = 1
        labels.append(label)
        
        # Store the point and its label
        points.append([x, y])

        # Draw the point on the image
        plt.scatter(x, y, c='red' if label == 1 else 'blue', s=40, edgecolor='black')
        plt.draw()

    # Connect the click event to the on_click function
    plt.gcf().canvas.mpl_connect('button_press_event', on_click)
    
    # Keep asking for points until user decides to stop
    plt.show(block=True)  # block until the user closes the figure
    plt.close()
    
    if len(points) == 0:
        print("No points selected. Exiting...")
        return

    input_points = np.array(points, dtype=np.float32)
    input_labels = np.array(labels, dtype=np.int64)
     
    
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True,
    )
    
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]
    
    
    # Only show the mask with highest score
    show_masks(img, masks[0][None], scores, point_coords=input_points, input_labels=input_labels, borders=True)
    
    plt.show()
    plt.close()
    
    # segment the subject with the mask
    image_np = np.array(Image.open(image_path).convert('RGB')).astype(np.float32) / 255 # [384, 512, 3]
    mask = masks[0]
    
    image_object = np.zeros_like(image_np)
    image_object = image_np * (mask == 1).reshape(mask.shape[0], mask.shape[1], 1).astype(np.float32) 
    
    
    # save the diptych image
    img_save_path = os.path.join(save_root, img_name)
    mask_save_path = img_save_path.replace("diptych", "mask")
       
    # construct diptych prompt: left is object, right is empty
    diptych_image_np = np.concatenate([image_object, np.zeros(image_np.shape)], axis=1)
    diptych_mask_np = np.concatenate([np.zeros(mask.shape), np.ones(mask.shape)], axis=1)
    
    Image.fromarray((diptych_image_np * 255).astype(np.uint8)).convert("RGB").save(img_save_path)
    Image.fromarray((diptych_mask_np * 255).astype(np.uint8)).save(mask_save_path)
    
    breakpoint()
    


if __name__ == "__main__":
    classes = []
    classes_text = "../food-101/meta/labels.txt"
    with open(classes_text) as f:
        for line in f: 
            # replace ' ' with '_' and lowercase
            line = line.replace(" ", "_").lower()
            print(line)
            classes.append(line.strip())
    
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--cls", type=str, default="Ramen", help="class name")
    args = arg_parser.parse_args()    
    
    cls = args.cls.replace(" ", "_").lower()
    if cls not in classes:
        raise ValueError(f"Class {cls} not found in the dataset")
    
    
    image_root = f"/home_nfs/kkennethwu_nldap/ColegaAI/FoodQualityEnhancement/DiptychPrompting/data/test/{cls}/"
    save_root = f"/home_nfs/kkennethwu_nldap/ColegaAI/FoodQualityEnhancement/DiptychPrompting/data_diptych/test/{cls}/"
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(save_root.replace("diptych", "mask"), exist_ok=True)
    image_paths = natsorted(glob(os.path.join(image_root, "*.jpg")))
    for image_path in image_paths:
        gen_seg_sam2(image_path, save_root)
    