import cv2
import numpy as np

def detect_freshness(image_path):
    img = cv2.imread(image_path)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Calculate average saturation
    avg_saturation = np.mean(hsv_img[:, :, 1])  # Saturation channel is index 1 in HSV
    
    # Set a threshold for freshness
    if avg_saturation > 100:
        return "Fresh"
    else:
        return "Stale or Rotten"

# Example usage
image_path = 'path_to_your_image.jpg'  # Replace with the path to your image
freshness_result = detect_freshness(image_path)
print(f'The freshness detection result is: {freshness_result}')
