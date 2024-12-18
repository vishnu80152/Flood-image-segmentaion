
# Flood Image Segmentation with U-Net

This project implements a deep learning-based solution for segmenting flood-affected regions in satellite or aerial images using the U-Net architecture.

## Overview

The goal of this project is to train a model capable of segmenting flood-affected areas from satellite or aerial images. The model uses the **U-Net** architecture, which is particularly effective for image segmentation tasks. The images are paired with corresponding masks that highlight the flood areas, and the model is trained to predict these masks based on the input images.

## Requirements

To run the project, you need the following Python libraries:

- Python 3.x
- TensorFlow 2.x
- Keras
- Matplotlib
- NumPy
- Pillow (PIL)
- Scikit-learn

You can install all the required libraries using pip:

```bash
pip install matplotlib numpy tensorflow scikit-learn
```

## Dataset

The dataset for this project consists of pairs of images and corresponding masks:

- **Images**: Satellite or aerial images of flood-affected areas.
- **Masks**: Binary masks highlighting the flood areas (1 for flooded areas, 0 for non-flooded areas).

### Dataset Organization

The dataset should be organized as follows:

```bash
data/
  ├── Image/
  └── Mask/
```

Where:
- `Image/` contains the image files (e.g., .jpg or .png).
- `Mask/` contains the corresponding mask files (e.g., .png).

## Project Structure

- `flood_image_segmenter.py`: Main script that trains the model and performs predictions.
- `README.md`: This file.

## Functions

### 1. `visualize_images_and_masks(num_images=5)`

Visualizes a specified number of sample images along with their corresponding masks.

**Parameters:**
- `num_images`: The number of images to visualize (default is 5).

### 2. `overlay_mask_on_image(num_images=5, alpha=0.5)`

Visualizes the overlay of the mask on the image, with the mask areas highlighted in red.

**Parameters:**
- `num_images`: The number of images to visualize (default is 5).
- `alpha`: The blending factor between the image and the mask overlay (default is 0.5).

### 3. `load_images_and_masks()`

Loads the images and masks from the disk, resizes them to the target shape, and normalizes the pixel values.

**Returns:**
- `images`: A numpy array of the normalized images.
- `masks`: A numpy array of the normalized masks.

### 4. `unet_model(input_size=(224, 224, 3))`

Defines the U-Net model architecture for semantic segmentation.

**Parameters:**
- `input_size`: The shape of the input image (default is (224, 224, 3)).

**Returns:**
- A compiled U-Net model.

### 5. `train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)`

Trains the U-Net model on the image-mask pairs using binary cross-entropy loss and accuracy as the metric.

**Parameters:**
- `X_train`: Training images.
- `y_train`: Training masks.
- `X_val`: Validation images.
- `y_val`: Validation masks.
- `epochs`: The number of epochs to train the model (default is 50).
- `batch_size`: The batch size to use for training (default is 32).

### 6. `visualize_predictions(num_images=5)`

Visualizes the model's predictions on a subset of validation images, comparing the predicted masks to the ground truth masks.

**Parameters:**
- `num_images`: The number of images to visualize (default is 5).

### 7. `predict_mask(image_path)`

Takes a new image as input, resizes it, and predicts the corresponding mask using the trained model.

**Parameters:**
- `image_path`: The path to the image file.

**Returns:**
- The predicted mask for the input image.

## Steps to Run the Project

### 1. Prepare the Dataset

Organize the dataset in the following structure:

```bash
data/
  ├── Image/
  └── Mask/
```

### 2. Train the Model

Run the script to load the images and masks, train the U-Net model, and visualize the training progress.

```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=BATCH_SIZE
)
```

### 3. Visualize Predictions

After training, you can visualize the model's predictions on a few validation images:

```python
visualize_predictions()
```

### 4. Save and Load the Model

After training, you can save the model for future use:

```python
model.save("flood_image_masker.keras")
```

To make predictions with the saved model, load the model and use the `predict_mask()` function:

```python
loaded_model = load_model("flood_image_masker.keras")
pred_mask = predict_mask(image_path)
```

### 5. Make Predictions on New Images

Use the `predict_mask()` function to predict the mask for a new image:

```python
pred_mask = predict_mask(image_path)
plt.imshow(pred_mask, cmap='gray')
plt.show()
```

## Model Architecture

The model is based on the U-Net architecture, which consists of the following parts:

- **Encoder**: A series of convolutional and pooling layers that extract features from the input image.
- **Bottleneck**: A deep convolutional layer that combines features extracted at various levels.
- **Decoder**: Upsampling layers combined with skip connections to reconstruct the segmentation mask from the features.

## Results

After training, the model should be able to generate segmentation masks that highlight flooded areas in unseen images.

## Example Image and Mask Visualization

- **Original Image**: A satellite or aerial image of a flood-affected area.
- **Ground Truth Mask**: The binary mask that highlights the flooded areas.
- **Predicted Mask**: The mask predicted by the trained model.

## Conclusion

This project demonstrates the application of deep learning, specifically U-Net, for flood image segmentation. The trained model can be used for real-time flood detection from satellite or aerial images.
