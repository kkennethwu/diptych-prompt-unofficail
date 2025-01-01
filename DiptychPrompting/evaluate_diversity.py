import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def extract_clip_features(image_path, model, processor, device):
    """Extract CLIP features for a single image."""
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        features = outputs / outputs.norm(dim=1, keepdim=True)
    
    return features.cpu().numpy()

def analyze_diversity(image_dir, cls_name, output_dir="diversity_analysis"):
    """Analyze diversity of generated images using CLIP features."""
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract features for all images
    features_list = []
    image_paths = []
    
    print(f"Extracting features for class: {cls_name}")
    for image_name in tqdm(os.listdir(image_dir)):
        if image_name.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(image_dir, image_name)
            features = extract_clip_features(image_path, model, processor, device)
            features_list.append(features.flatten())
            image_paths.append(image_path)
    
    # Convert to numpy array
    features_array = np.stack(features_list)
    n_samples = len(features_list)
    
    if n_samples < 3:
        print(f"Warning: Not enough samples ({n_samples}) for class {cls_name} to perform analysis")
        return None
        
    # Adjust t-SNE parameters based on sample size
    perplexity = min(n_samples - 1, 15)  # Ensure perplexity is less than n_samples
    
    # Perform t-SNE with adjusted perplexity
    print(f"Performing t-SNE with perplexity {perplexity}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    tsne_results = tsne.fit_transform(features_array)
    
    # Perform PCA
    print("Performing PCA...")
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(features_array)
    
    # Calculate diversity metrics
    def calculate_diversity_metrics(embeddings):
        # Average pairwise distance
        pairwise_distances = np.linalg.norm(embeddings[:, None] - embeddings, axis=2)
        avg_distance = np.mean(pairwise_distances[pairwise_distances > 0])
        
        # Standard deviation of pairwise distances
        std_distance = np.std(pairwise_distances[pairwise_distances > 0])
        
        # Coverage - use the area of the convex hull if we have enough points
        if len(embeddings) >= 3:
            from scipy.spatial import ConvexHull
            try:
                hull = ConvexHull(embeddings)
                coverage = hull.area
            except Exception as e:
                print(f"Warning: Could not compute convex hull: {e}")
                coverage = 0
        else:
            coverage = 0
            
        return {
            'average_distance': avg_distance,
            'std_distance': std_distance,
            'coverage': coverage
        }
    
    # Calculate metrics for both t-SNE and PCA results
    tsne_metrics = calculate_diversity_metrics(tsne_results)
    pca_metrics = calculate_diversity_metrics(pca_results)
    
    # Plotting
    def plot_embeddings(embeddings, title, metrics, filename):
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.5)
        
        # Add labels for each point
        for i, path in enumerate(image_paths):
            plt.annotate(os.path.basename(path), 
                        (embeddings[i, 0], embeddings[i, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8)
            
        plt.title(f"{title}\n" + 
                 f"Avg Distance: {metrics['average_distance']:.2f}\n" +
                 f"Std Distance: {metrics['std_distance']:.2f}\n" +
                 f"Coverage: {metrics['coverage']:.2f}\n" +
                 f"N Samples: {n_samples}")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        
        # Add colorbar
        plt.colorbar(scatter)
        
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    plot_embeddings(tsne_results, f"t-SNE Visualization - {cls_name}", 
                   tsne_metrics, f"{cls_name}_tsne.png")
    plot_embeddings(pca_results, f"PCA Visualization - {cls_name}", 
                   pca_metrics, f"{cls_name}_pca.png")
    
    # Calculate explained variance ratio for PCA
    explained_variance = pca.explained_variance_ratio_
    
    return {
        'tsne_metrics': tsne_metrics,
        'pca_metrics': pca_metrics,
        'explained_variance': explained_variance,
        'n_samples': n_samples
    }

if __name__ == "__main__":
    image_root = "/home_nfs/kkennethwu_nldap/ColegaAI/FoodQualityEnhancement/DiptychPrompting/output_blip/test"
    classes = []
    for cls in os.listdir(image_root):
        classes.append(cls)
    output_dir = "./diversity_analysis_blip"
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze diversity for each class
    diversity_results = {}
    for cls in tqdm(classes, desc="Processing classes"):
        image_dir = os.path.join(image_root, cls)
        if os.path.exists(image_dir):
            results = analyze_diversity(image_dir, cls, output_dir)
            if results is not None:
                diversity_results[cls] = results
    
    # Save overall results
    with open(os.path.join(output_dir, "diversity_metrics.txt"), "w") as f:
        f.write("Class-wise Diversity Metrics:\n\n")
        for cls, metrics in diversity_results.items():
            f.write(f"\n{cls}:\n")
            f.write(f"Number of samples: {metrics['n_samples']}\n")
            f.write("\nt-SNE Metrics:\n")
            for k, v in metrics['tsne_metrics'].items():
                f.write(f"{k}: {v:.4f}\n")
            f.write("\nPCA Metrics:\n")
            for k, v in metrics['pca_metrics'].items():
                f.write(f"{k}: {v:.4f}\n")
            f.write(f"PCA Explained variance ratio: {metrics['explained_variance']}\n")
            f.write("\n" + "="*50 + "\n")
            
