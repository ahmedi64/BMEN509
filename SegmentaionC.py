import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from sklearn.model_selection import train_test_split

#Load Data 
def load_data(data_dir, img_size=(128, 128)):
    images, masks = [], []
    categories = ['Glioma', 'Meningioma', 'Pituitary tumor']
    
    for category in categories:
        folder = os.path.join(data_dir, category)
        for file in os.listdir(folder):
            if '_mask' not in file:
                img_path = os.path.join(folder, file)
                mask_path = os.path.join(folder, file.split('.')[0] + '_mask.png')
                
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None or mask is None:
                    continue
                
                img = cv2.resize(img, img_size) / 255.0
                mask = cv2.resize(mask, img_size) / 255.0
                
                images.append(img)
                masks.append(mask)
    
    images = np.expand_dims(np.array(images), axis=-1)
    masks = np.expand_dims(np.array(masks), axis=-1)
    return images, masks

# Simple U-Net Model

def unet_model(input_size=(128, 128, 1)):
    inputs = Input(input_size)

    # Encoder
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D()(conv1)

    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)

    # Decoder
    up1 = Conv2DTranspose(32, 2, strides=2, padding='same')(conv2)
    merge1 = concatenate([up1, conv1])
    conv3 = Conv2D(1, 1, activation='sigmoid')(merge1)

    model = Model(inputs=inputs, outputs=conv3)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

#  Dice Score Function 

def dice_coef(pred, truth):
    pred = pred.astype(np.float32).flatten()
    truth = truth.astype(np.float32).flatten()
    intersection = np.sum(pred * truth)
    return (2. * intersection + 1e-6) / (np.sum(pred) + np.sum(truth) + 1e-6)
# Load and Prepare Data 
X, Y = load_data('/Users/kenzyhamed/Downloads/data')  # Make sure the path and folder structure match
print(f"Loaded {len(X)} MRI scans.")

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Build and Train Model 
model = unet_model()
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=20, batch_size=4)

# Predict and Evaluate 
i = 0
mri = X_val[i]
true_mask = Y_val[i]
pred_mask = model.predict(np.expand_dims(mri, axis=0))[0]
binary_pred = (pred_mask > 0.5).astype(np.float32)

dice = dice_coef(binary_pred, true_mask)
print(f"Dice Score (Sample {i}): {dice:.4f}")

#Visualize 
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(mri.squeeze(), cmap='gray')
plt.title("Original MRI")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(true_mask.squeeze(), cmap='Reds')
plt.title("Ground Truth Mask")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(mri.squeeze(), cmap='gray')
plt.imshow(binary_pred.squeeze(), cmap='Reds', alpha=0.4)
plt.title("Predicted Mask Overlay")
plt.axis('off')

plt.tight_layout()
plt.show()
