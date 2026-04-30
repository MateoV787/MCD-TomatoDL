"""
=====================================================================
PARTIAL FINE-TUNING OF EfficientNetB0 (11-CLASS MODEL)
=====================================================================

Author: Mateo Valencia
Location: Universidad del Quindío, Armenia - Colombia

DESCRIPTION:
This script performs fine-tuning on a pretrained model based on
EfficientNetB0 that originally classifies 11 classes.

OBJECTIVE:
Refine the model performance on 3 specific classes:
- healthy
- Early_blight
- Late_blight

WITHOUT:
- Modifying the architecture
- Changing the output layer (it remains 11 classes)
- Applying data augmentation

STRATEGY:
- Training ONLY with images from 3 classes
- Most layers are frozen
- Final layers are adjusted with a low learning rate
- Minimizing the risk of "catastrophic forgetting"

"""

# =========================================================
#IMPORT LIBRARIES
# =========================================================
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# =========================================================
# GENERAL CONFIGURATION
# =========================================================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

TRAIN_DIR = r"C:\Users\teori\OneDrive\Documentos\2_Personal\ProyectoPAI\Dataset_UQ\train"   
VAL_DIR = r"C:\Users\teori\OneDrive\Documentos\2_Personal\ProyectoPAI\Dataset_UQ\valid"

MODEL_PATH = r"C:\Users\teori\OneDrive\Documentos\2_Personal\ProyectoPAI\tomato_model.h5"
OUTPUT_MODEL = r"C:\Users\teori\OneDrive\Documentos\2_Personal\ProyectoPAI\modelo_refinado.h5"

# =========================================================
# DATASET LOADING
# =========================================================

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# =========================================================
#PREPROCESSING
# =========================================================

normalization = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization(x), y))
val_ds = val_ds.map(lambda x, y: (normalization(x), y))

# Performance improvement
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# =========================================================
#LOAD PRETRAINED MODEL
# =========================================================


model = tf.keras.models.load_model(MODEL_PATH)

model.summary()

# =========================================================
# STRATEGIC FREEZING
# =========================================================


"""
Most layers are frozen to:
- Preserve knowledge of the other 8 classes
- Avoid overfitting
- Maintain model stability

Only the final layers are trained
"""

for layer in model.layers[:-40]:  # To avoid catastrophic forgetting
    layer.trainable = False

# =========================================================
# COMPILATION
# =========================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================================================
#CALLBACKS
# =========================================================

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# =========================================================
#TRAINING
# =========================================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[lr_scheduler, early_stop]
)

# =========================================================
# MODEL SAVING
# =========================================================

model.save(OUTPUT_MODEL)
