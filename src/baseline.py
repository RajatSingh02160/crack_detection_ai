import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans

def process_cracks(image_path):
    # 1. READ IMAGE
    img = cv2.imread(image_path)
    if img is None:
        print("[!] Error: Image not found.")
        return

    # Create 'results' directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')

    # 2. PROCESSING
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # K-Means logic
    pixel_values = gray.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixel_values, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()].reshape(gray.shape)

    # 3. SAVING INDIVIDUAL STEPS (High Quality)
    # We use cv2.imwrite for this. Note: cv2 uses BGR, so no need to convert for saving
    cv2.imwrite('results/output_gray.jpg', gray)
    cv2.imwrite('results/output_edges.jpg', edges)
    cv2.imwrite('results/output_kmeans.jpg', segmented_image)

    # 4. VISUALIZATION & SAVING THE SUMMARY
    plt.figure(figsize=(15, 5))
    plt.subplot(141), plt.imshow(img_rgb), plt.title('Original')
    plt.subplot(142), plt.imshow(gray, cmap='gray'), plt.title('Grayscale')
    plt.subplot(143), plt.imshow(edges, cmap='gray'), plt.title('Canny Edges')
    plt.subplot(144), plt.imshow(segmented_image, cmap='gray'), plt.title('K-Means')
    
    # Save the 4-plot summary for your report
    plt.savefig('results/full_comparison_plot.png', dpi=300)
    print("[+] Results saved in the 'results/' folder.")
    plt.show()

if __name__ == "__main__":
    process_cracks('assets/crack_img.jpeg')