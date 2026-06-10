import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

class CrackDetectorAI:
    def __init__(self):
        # 1. Load the "Brain" (Pre-trained ResNet18)
        # We use weights=ResNet18_Weights.DEFAULT for the latest version
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.eval()  # Set to 'Evaluation' mode (turns off dropout etc.)

        # 2. The "Glasses" (Transforms)
        # AI needs specific size (224x224) and Normalization based on ImageNet stats
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def analyze_image(self, image_path):
        # Load image using PIL (standard for PyTorch)
        img = Image.open(image_path).convert('RGB')
        img_t = self.transform(img)
        batch_t = torch.unsqueeze(img_t, 0) # Add a 'batch' dimension

        # 3. FORWARD PASS
        # We want to see the output of the first few layers (where edges are detected)
        with torch.no_grad():
            # Get features from the very first layer (Conv1)
            features = self.model.conv1(batch_t)
        
        return features, img

    # def visualize_features(self, features, original_img):
    #     # Convert features to a visible heatmap
    #     # We take the average of all 64 filters in the first layer
    #     feature_map = torch.mean(features[0], dim=0).cpu().numpy()
        
    #     # Normalize the map so we can see it
    #     feature_map = np.maximum(feature_map, 0)
    #     feature_map /= np.max(feature_map)

    #     # Plotting
    #     plt.figure(figsize=(10, 5))
    #     plt.subplot(121)
    #     plt.imshow(original_img)
    #     plt.title("Original (AI Input)")
        
    #     plt.subplot(122)
    #     plt.imshow(feature_map, cmap='magma')
    #     plt.title("AI Feature Map (Conv1 Layer)")
        
    #     print("[+] AI Analysis complete. Showing what the Neural Network 'sees'.")
    #     plt.show()
    def visualize_features(self, features, original_img):
        # 1. Processing the map (same as before)
        feature_map = torch.mean(features[0], dim=0).cpu().numpy()
        feature_map = np.maximum(feature_map, 0)
        feature_map /= np.max(feature_map)

        # 2. Setup Plotting
        fig = plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(original_img)
        plt.title("Original (AI Input)")
        
        plt.subplot(122)
        plt.imshow(feature_map, cmap='magma')
        plt.title("AI Feature Map (Conv1 Layer)")
        
        # 3. NEW: Saving Logic
        if not os.path.exists('results'):
            os.makedirs('results')
        
        save_path = 'results/ai_feature_map.png'
        plt.savefig(save_path, dpi=300) # Save high-res version
        print(f"[+] AI Analysis complete. Saved result to: {save_path}")
        
        plt.show()
if __name__ == "__main__":
    detector = CrackDetectorAI()
    # Path logic: assuming running from root
    path = 'assets/crack_img.jpeg'
    feat, original = detector.analyze_image(path)
    detector.visualize_features(feat, original)

