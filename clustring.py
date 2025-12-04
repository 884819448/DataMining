import os
import json
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm

def load_data(dataset_path):
    image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()  # Ensure consistent order
    return image_files

def extract_features(dataset_path, image_files, device):
    # Load pre-trained ResNet50
    model = models.resnet50(pretrained=True)
    # Remove the last fully connected layer to get feature vectors
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model = model.to(device)
    model.eval()

    # Define transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features = []
    
    print("Extracting features...")
    with torch.no_grad():
        for img_file in tqdm(image_files):
            img_path = os.path.join(dataset_path, img_file)
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = preprocess(img).unsqueeze(0).to(device)
                
                feature = model(img_tensor)
                # Flatten the feature
                feature = feature.cpu().numpy().flatten()
                features.append(feature)
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                features.append(np.zeros(2048)) # Fallback for error

    return np.array(features)

def perform_clustering(features, n_clusters=6):
    print(f"Clustering into {n_clusters} clusters...")
    
    # Optional: PCA for dimensionality reduction before K-Means
    # It often helps K-Means to work on lower dimensions
    pca = PCA(n_components=50, random_state=42)
    features_pca = pca.fit_transform(features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_pca)
    return clusters

def evaluate_results(image_files, clusters, labels_path):
    try:
        with open(labels_path, 'r') as f:
            ground_truth_map = json.load(f)
        
        true_labels = []
        pred_labels = []
        
        valid_files = 0
        for i, img_file in enumerate(image_files):
            if img_file in ground_truth_map:
                true_labels.append(ground_truth_map[img_file])
                pred_labels.append(clusters[i])
                valid_files += 1
        
        if valid_files == 0:
            print("No matching labels found for evaluation.")
            return

        # Calculate metrics
        ari = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        
        print("\nEvaluation Results:")
        print(f"Total images evaluated: {valid_files}")
        print(f"Adjusted Rand Index (ARI): {ari:.4f}")
        print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
        
        # Print cluster distribution
        print("\nCluster Distribution:")
        import pandas as pd
        df = pd.DataFrame({'True Label': true_labels, 'Cluster': pred_labels})
        cross_tab = pd.crosstab(df['True Label'], df['Cluster'])
        print(cross_tab)
        
    except Exception as e:
        print(f"Evaluation failed: {e}")

def main():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.abspath(os.path.join(base_dir, '../Cluster/dataset'))
    labels_path = os.path.abspath(os.path.join(base_dir, '../Cluster/cluster_labels.json'))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    image_files = load_data(dataset_path)
    print(f"Found {len(image_files)} images.")

    # 2. Extract Features
    features = extract_features(dataset_path, image_files, device)

    # 3. Cluster
    # We know there are 6 categories: transistor, leather, pill, bottle, tile, cable
    clusters = perform_clustering(features, n_clusters=6)

    # 4. Evaluate
    evaluate_results(image_files, clusters, labels_path)

    # Save results
    output_file = os.path.join(base_dir, 'clustering_results.json')
    results = {img: int(cluster) for img, cluster in zip(image_files, clusters)}
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nClustering assignments saved to {output_file}")

if __name__ == "__main__":
    main()
