# Bird Species Classification

This project aims to solve a multi-class classification problem involving bird species identification. A pre-trained **MobileNetV2** model is fine-tuned for this task, and several techniques like data augmentation, class weight balancing, and regularization are applied to achieve optimal performance.


## Table of Contents
- Overview
- Dataset
- Model Architecture
- Preprocessing Techniques
- Methodology
- Training and Evaluation
- Results
- Conclusion

## Overview
This project aims to solve a multi-class classification problem involving bird species identification. A pre-trained MobileNetV2 model is fine-tuned for this task, and several techniques like data augmentation, class weight balancing, and regularization are applied to achieve optimal performance.

### *Key Features*
- `Fine-tuned MobileNetV2 model for bird species classification.`
- `Data augmentation for robust learning.`
- `Handling class imbalance with class weights.`
- `EarlyStopping and ReduceLROnPlateau for training optimization.`
- `Final model saved and exported for real-world deployment.`


## Dataset
The dataset can be downloaded from [https://drive.google.com/drive/folders/13yCz3Hc7zaBAsne1jffq6IKL4EEX41lv?usp=sharing](#) and should be unzipped into the project directory.

- **Train images:** `images/train`
- **Test images:** `images/test`
- **CSV Files:** `train.csv`, `test.csv`

## Model Architecture

We used **MobileNetV2**, a lightweight and efficient convolutional neural network, pre-trained on **ImageNet**. The last 20 layers of the model are fine-tuned for domain-specific learning, specifically for bird species classification. Custom layers are added for feature extraction and classification.

- **Base Model:** MobileNetV2 (pre-trained on ImageNet)
- **Custom Layers:**
  - `GlobalAveragePooling2D`
  - `Dense` (1024 units, ReLU, L2 Regularization)
  - `Dropout` (0.5)
  - Output Layer: `Softmax` for multi-class classification

## Preprocessing Techniques

### 1. Data Augmentation:
We applied the following augmentations using `ImageDataGenerator` to improve model generalization:
- Rotation
- Zoom
- Width and height shifts
- Shearing
- Horizontal flipping

### 2. Image Rescaling:

Pixel values were rescaled to the range `[0, 1]` using a factor of `1./255`.

### 3. Handling Class Imbalance:

Class weights were computed based on the distribution of bird species using `sklearn`'s `compute_class_weight()` function to ensure balanced learning across the 200 classes.

## Methodology
1. **Data Augmentation:** Applied techniques like rotation, zoom, shifting, and horizontal flipping to increase dataset variability.
2. **Model Architecture:**
   - Pre-trained **MobileNetV2** (on ImageNet) with fine-tuned last 20 layers.
   - Added custom layers: GlobalAveragePooling, Dense, Dropout, and Softmax for classification.
3. **Class Imbalance Handling:** Computed and applied class weights using `sklearn`.
4. **Training Optimization:**
   - Used **Adam optimizer** with early stopping, learning rate reduction, and model checkpoint callbacks for optimal performance.


## Training and Evaluation

- **Optimizer:** Adam with a learning rate of `0.0001`
- **Loss Function:** Categorical Cross-Entropy
- **Metrics:** Accuracy

### *Callbacks*:
- **EarlyStopping:** Monitors validation loss and halts training early if no improvement is observed.
- **ModelCheckpoint:** Saves the best model based on validation loss.
- **ReduceLROnPlateau:** Reduces the learning rate when validation loss plateaus.

The model was trained for **25 epochs** with data augmentation and class weights to handle class imbalance.

## Results

The fine-tuned model achieved **94% accuracy** in distinguishing bird species across the 200 classes. The final model was saved as the  `best_model.keras` based on the validation loss during training.
At the end of the model evaluation, we generate a `predictions.csv` file that contains the following columns:

- **path** (`str`): The image path (kept the same as in `test.csv`).
- **predicted_label** (`int`): The predicted class label for each image.
- **confidence_score** (`float`): The confidence score associated with the predicted class.

This file helps to review the model's performance on the test set, making it easy to analyze the predictions and evaluate the model in a real-world scenario.

