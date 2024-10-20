import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

def create_model(num_classes):
    # Load the MobileNetV2 model pre-trained on ImageNet
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

    # Freeze the base model initially
    base_model.trainable = False

    # Create the model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Output layer for the number of classes
    ])

    # Compile the model with a lower learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # You can adjust the learning rate as needed
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def main():
    # Define paths to your dataset directories
    train_data_dir = '/Users/thetanmaymishra/Desktop/Flipkart Grid/freshness/Data/train'
    validation_data_dir = '/Users/thetanmaymishra/Desktop/Flipkart Grid/freshness/Data/validation'

    # Set parameters
    batch_size = 32
    img_height = 224
    img_width = 224

    # Data augmentation and preprocessing
    train_datagen = ImageDataGenerator(rescale=1.0/255.0)
    val_datagen = ImageDataGenerator(rescale=1.0/255.0)

    # Load training data
    train_data = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    # Load validation data
    val_data = val_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # Step 1: Determine the number of unique classes in the training data
    unique_classes_train = np.unique(train_data.classes)  # Get unique classes from the training data
    num_classes_train = len(unique_classes_train)

    # Ensure the validation data has the same unique classes as the training data
    unique_classes_val = np.unique(val_data.classes)  # Get unique classes from the validation data
    num_classes_val = len(unique_classes_val)

    # Print the unique classes and their counts
    print(f"Unique classes in training data: {unique_classes_train}")
    print(f"Unique classes in validation data: {unique_classes_val}")
    print(f"Number of classes in training data: {num_classes_train}")

    # Check if training and validation classes match
    if set(unique_classes_train) != set(unique_classes_val):
        print("Warning: Training and validation datasets have different classes.")
    
    # Step 2: Ensure that num_classes passed to create_model matches the training data
    model = create_model(num_classes_train)  # Use num_classes_train

    # Now proceed to train your model
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=20
    )

if __name__ == '__main__':
    main()
