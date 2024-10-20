import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = tf.keras.models.load_model('fruit_vegetable_model_finetuned.h5')

# Define the mapping of class indices to labels
labels = {
    0: 'Rotten',
    1: 'Stale',
    2: 'Fresh',
    # Add additional labels here corresponding to your classes
}

# Function to predict freshness
def predict_freshness(image):
    img = cv2.resize(image, (224, 224))
    img = img.astype('float32') / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img)
    class_index = np.argmax(prediction, axis=1)
    predicted_label = labels.get(class_index[0], "Unknown")
    freshness_score = np.max(prediction)  # Getting the confidence score

    return predicted_label, freshness_score

# Function to process the video frame
def process_frame(frame, blur_value, threshold_value):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_value, blur_value), 0)
    _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

    return thresh

# Function to draw predictions on the frame
def draw_predictions(frame, contours):
    mask = np.zeros_like(frame)  # Create an empty mask same size as the frame
    
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small contours
            (x, y, w, h) = cv2.boundingRect(contour)
            roi = frame[y:y + h, x:x + w]  # Region of Interest for prediction
            
            # Predict only if there's a significant area detected
            predicted_label, freshness_score = predict_freshness(roi)

            # Draw rectangle around the detected object on the original frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Prepare text to display
            text = f"{predicted_label} - Freshness Score: {freshness_score:.2f}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Draw the same rectangle on the mask (for visualization)
            cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)  # White filled rectangle on mask

    return frame, mask

# Open the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

# Create a window for the trackbars
cv2.namedWindow('Trackbars')

# Create trackbars for adjusting GaussianBlur and threshold
cv2.createTrackbar('Blur', 'Trackbars', 1, 20, lambda x: None)  # Kernel size for blur (odd values)
cv2.createTrackbar('Threshold', 'Trackbars', 150, 255, lambda x: None)  # Threshold value

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Get current positions of trackbars
    blur_value = cv2.getTrackbarPos('Blur', 'Trackbars')
    threshold_value = cv2.getTrackbarPos('Threshold', 'Trackbars')

    # Ensure blur value is odd (required by GaussianBlur)
    if blur_value % 2 == 0:
        blur_value += 1

    # Process the frame
    thresh = process_frame(frame, blur_value, threshold_value)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw predictions on the frame
    frame, mask = draw_predictions(frame, contours)

    # Display the video frame with predictions
    cv2.imshow("Freshness Detection", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Mask", mask)  # Show the mask where the detected areas are highlighted

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
