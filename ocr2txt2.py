import cv2
import pytesseract
import numpy as np
import re
import datetime

# Preprocessing Function
def preprocess_image(image):
    # Resize image for uniformity (Optional)
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply contrast and brightness adjustments
    contrast_img = cv2.convertScaleAbs(gray, alpha=1.5, beta=50)  # Alpha: contrast, Beta: brightness
    
    # Reduce noise using a Gaussian filter
    denoised_img = cv2.GaussianBlur(contrast_img, (5, 5), 0)
    
    # Use adaptive thresholding
    thresh_image = cv2.adaptiveThreshold(denoised_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    
    return thresh_image

# Function to extract specific details from text
def extract_specific_details(text):
    # Define patterns to capture MRP, expiry date, and brand
    mrp_pattern = r'MRP[\s:₹]*([\d\.]+)'
    expiry_pattern = r'EXPIRY DATE[:\s]*([A-Za-z0-9\s/]+)'
    brand_pattern = r'Brand[:\s]*([A-Za-z0-9]+)'

    mrp = re.search(mrp_pattern, text)
    expiry_date = re.search(expiry_pattern, text)
    brand = re.search(brand_pattern, text)
    
    # Print extracted information
    if mrp:
        print("MRP: ₹", mrp.group(1))
    if expiry_date:
        print("Expiry Date:", expiry_date.group(1))
    if brand:
        print("Brand:", brand.group(1))

# Start capturing video from the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

# Initialize a variable to hold the extracted text
extracted_text = ""

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    preprocessed_image = preprocess_image(frame)

    # Extract text from preprocessed image
    text = pytesseract.image_to_string(preprocessed_image)

    # Print the extracted text
    print("Extracted Text: \n", text)

    # Update the accumulated extracted text
    extracted_text += text + "\n"  # Append the text with a newline

    # Call the specific detail extraction function
    extract_specific_details(text)

    # Display the preprocessed image (optional)
    cv2.imshow('Preprocessed Image', preprocessed_image)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # Press 's' to save the text
        # Open a file to save the extracted text
        filename = f'extracted_text_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(filename, 'w') as file:
            file.write(extracted_text)  # Write the accumulated text to the file
        print(f"Text saved to {filename}")

    elif key == ord('q'):  # Exit on 'q' key
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
