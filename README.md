# A multivariate correlation approach for detection and classification of leaf diseases in greenhouse tomato crops using deep learning and machine learning models
This study implements a neural network model for the detection and classification of diseases in tomato leaves, using labeled images and deep learning techniques.

<img width="1400" height="1000" alt="Figura 3" src="https://github.com/user-attachments/assets/a10a8c30-553e-4ceb-819a-17c71c03f267" />


# Confusion Matrix
The confusion matrix allows evaluating the model's performance for each class by showing correct predictions and classification errors.

<img width="1200" height="1000" alt="Figura 7" src="https://github.com/user-attachments/assets/0ed893c6-6f2b-4dd4-b338-e860bbce5d86" />

A high number of correct predictions is observed along the main diagonal.
The best-performing classes include:
- Late blight
- Healthy
- Tomato mosaic virus
Errors are minimal and mainly occur between diseases with similar visual characteristics.

 # Model Accuracy
 Evolution of accuracy during training and validation:

<img width="9648" height="6840" alt="Figure_9a" src="https://github.com/user-attachments/assets/a1886d77-e090-47c6-b8b1-0647b7a91813" />


Accuracy reaches values close to 99%.
Training and validation curves are very close, indicating:
- Good generalization
- Low overfitting

# Loss Function

<img width="6432" height="4560" alt="Figure_9b" src="https://github.com/user-attachments/assets/669b8ca4-772c-47c8-a738-8d5d530beb4f" />


Loss decreases rapidly in the early epochs.
It stabilizes near zero, indicating:

- Effective learning
- Model convergence

# Model Fine-Tuning (EfficientNetB0)
The base model used corresponds to a pre-trained EfficientNetB0 network for classifying 11 tomato leaf disease classes.

## Objective
The fine-tuning process focused on improving performance in three specific classes:

- Healthy
- Early blight
- Late blight

This was done without significantly affecting knowledge in the remaining classes.

# Implemented Strategy
To achieve this, a controlled fine-tuning strategy was applied:
## Layer Freezing
Most layers of the model were frozen:
```python
for layer in model.layers[:-40]:
    layer.trainable = False
```
This allows:
- Preserving prior knowledge
- Avoiding catastrophic forgetting
- Reducing overfitting risk

# Partial Training
Training was performed using only images from 3 classes.
The following were NOT modified:
- Model architecture
- Output layer (still 11 classes)

This means the model remains multiclass, but with refinement in specific classes.

# Preprocessing
Image normalization:
```python
Rescaling(1./255)
```
No data augmentation was applied, making the adjustment more controlled and directly based on original data.

# Training Configuration
- Low learning rate: 1e-5
- Batch size: 32
- Epochs: 20

This allows fine adjustments without drastically altering learned weights.

# Callbacks Used
To optimize training:
ReduceLROnPlateau
- Reduces learning rate when loss stagnates
EarlyStopping
- Stops training when no improvement is observed
- Restores the best weights

## Tomato model (.h5) link: https://n9.cl/3gux2

# Random Forest Model
This study also implements a Random Forest model to predict crop status based on soil and environmental variables.

# Confusion Matrix

<img width="700" height="600" alt="Figure 10" src="https://github.com/user-attachments/assets/ba96f979-2c42-41ce-9f70-45a08d561bf6" />

The confusion matrix allows visualization of model performance by comparing predictions against actual values.

- Each row represents the actual class
- Each column represents the predicted class
- Diagonal values indicate correct predictions

This helps identify which classes perform better and where misclassifications occur.

# Reporte de clasificación

<img width="927" height="232" alt="Screenshot 2026-04-29 101106" src="https://github.com/user-attachments/assets/86367c1e-aa80-4b97-a341-77d1ad4ad644" />

The classification report summarizes key model metrics:

- Precision: how reliable positive predictions are
- Recall (Sensitivity): proportion of actual cases correctly detected
- F1-score: balance between precision and recall
- Support: number of samples per class

This enables detailed evaluation for each class.

# Feature Importance (Gini Coefficient)

<img width="989" height="490" alt="Figura 11" src="https://github.com/user-attachments/assets/219110dd-0aa6-4035-a9f2-afac50ecaa9d" />

Feature importance is calculated using the Gini coefficient, which measures how much each variable contributes to impurity reduction in the model's trees.

- Higher values indicate greater importance
- Helps identify the most influential variables
- Improves model interpretability

# Correlation Matrix (Soil and Environmental Variables)

<img width="935" height="790" alt="Correlacion" src="https://github.com/user-attachments/assets/1c511545-f7b4-4d1e-aef6-0a237e992afa" />

The correlation matrix shows the linear relationship between dataset variables using the Pearson correlation coefficient.

- Values close to 1 → strong positive correlation
- Values close to -1 → strong negative correlation
- Values close to 0 → little or no relationship

This analysis helps:

- Identify highly correlated variables
- Detect multicollinearity issues
- Better understand data structure before modeling

# Hardware Specifications
Model training was performed on an HP Victus 15-fb3019la laptop, capable of handling intensive computational tasks efficiently.

## Main Features
Processor (CPU):
- AMD Ryzen 7 7445HS (6 cores / 12 threads, up to 4.7 GHz)

Graphics Card (GPU):
- NVIDIA GeForce RTX 3050 (6 GB GDDR6 dedicated)

RAM:
- 16 GB DDR4
  
Storage:
- 512 GB NVMe SSD
  
Display:
- 15.6" Full HD (1920 × 1080)
  
Operating System:
- Windows 11 and Linux
  
Architecture:
- 64-bits

# Application hardware

## Main Features

Processor (CPU):

- Broadcom BCM2712 Quad-Core ARM Cortex-A76 (4 cores, up to 2.4 GHz)


Graphics Processor (GPU):

- VideoCore VII GPU with OpenGL ES 3.1 and Vulkan 1.2 support


RAM:

- 8 GB LPDDR4X SDRAM 


Storage:

- High-speed microSD card and external SSD support via USB 3.0


Connectivity:

- Gigabit Ethernet
- Dual-band Wi-Fi (2.4 GHz / 5 GHz)
- Bluetooth 5.0 BLE


Interfaces:

- 40-pin GPIO header
- USB 3.0 and USB 2.0 ports
- CSI camera interface
- DSI display interface


Operating System:

- Raspberry Pi OS (64-bit)


Architecture:

- 64-bit ARM Architecture
