import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, Conv2DTranspose
from tensorflow.keras.callbacks import EarlyStopping

# Function to load images and their corresponding segmentation masks
def load_images_and_masks(image_path, mask_path, n):
    images = []
    masks = []
    for filename in os.listdir(image_path):
        if filename.endswith(".jpg"):
            # Load the image
            image = cv2.imread(os.path.join(image_path, filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (n, n))  # Resize images to a common size
            images.append(image)

            # Load the corresponding mask image
            mask_filename = filename.split('.')[0] + '.png'  # Assuming mask images have _mask.png extension
            mask_img = cv2.imread(os.path.join(mask_path, mask_filename), cv2.IMREAD_GRAYSCALE)
            mask_img = cv2.resize(mask_img, (n, n))  # Resize mask images
            masks.append(mask_img)

    return np.array(images), np.array(masks)

# Define paths to your dataset images and labels
train_image_path = 'train_img'
train_mask_path = 'train_lab'
test_image_path = 'test_img'
test_mask_path = 'test_lab'

n=256

# Load training images and segmentation masks
X_train, y_train = load_images_and_masks(train_image_path, train_mask_path, n)

# Load testing images and segmentation masks
X_test, y_test = load_images_and_masks(test_image_path, test_mask_path, n)

# Normalize and preprocess the images
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Expand dimensions for single channel (grayscale)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Normalize masks (divide by 255 to obtain values between 0 and 1)
y_train = y_train.astype('float32') / 255.0
y_test = y_test.astype('float32') / 255.0

# Expand dimensions for single channel (grayscale)
y_train = np.expand_dims(y_train, axis=-1)
y_test = np.expand_dims(y_test, axis=-1)

# Define the model for segmentation
input_shape = (n, n, 1)
inputs = Input(input_shape)
conv1 = Conv2D(n/4, (3, 3), activation='relu', padding='same')(inputs)
conv1 = Conv2D(n/4, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(n/2, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Conv2D(n/2, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(n, (3, 3), activation='relu', padding='same')(pool2)
conv3 = Conv2D(n, (3, 3), activation='relu', padding='same')(conv3)

up1 = Conv2DTranspose(n/2, (2, 2), strides=(2, 2), padding='same')(conv3)
up1 = Conv2D(n/2, (3, 3), activation='relu', padding='same')(up1)

up2 = Conv2DTranspose(n/4, (2, 2), strides=(2, 2), padding='same')(up1)
up2 = Conv2D(n/4, (3, 3), activation='relu', padding='same')(up2)

outputs = Conv2D(1, (1, 1), activation='sigmoid')(up2)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate the model
test_loss, test_acc, test_mse = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")

# Save the trained model
model.save('crack_segmentation_model.h5')
