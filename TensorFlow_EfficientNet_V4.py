"""
===========================================================
PROYECTO: Clasificación de enfermedades en tomate
MODELO: EfficientNetB0
AUTOR: Mateo Valencia
DESCRIPCIÓN:
Este script carga imágenes desde carpetas, genera dataframes,
entrena un modelo CNN usando EfficientNet y evalúa resultados.
===========================================================
"""

# =========================
#IMPORTACIONES
# =========================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.preprocessing import image

import seaborn as sns

# =========================
# CONFIGURACIÓN
# =========================
TRAIN_DIR = r"C:\Users\teori\OneDrive\Documentos\2_Personal\ProyectoPAI\Dataset\train"   
VAL_DIR = r"C:\Users\teori\OneDrive\Documentos\2_Personal\ProyectoPAI\Dataset\valid"

IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 11

# =========================
# CREACIÓN DE DATAFRAMES
# =========================
def create_dataframe(path):
    """
    Recorre carpetas y crea un DataFrame con:
    - Ruta de imagen
    - Etiqueta (clase)
    """
    data = []
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        for file in os.listdir(folder_path):
            label = folder.split('-')[-1]
            filepath = os.path.join(folder_path, file)
            data.append([filepath, label])

    return pd.DataFrame(data, columns=['filename', 'label'])


df = create_dataframe(TRAIN_PATH)
val_df = create_dataframe(VAL_PATH)

# =========================
# ANÁLISIS EXPLORATORIO
# =========================
def plot_distribution(dataframe, title):
    dataframe['label'].value_counts().plot(kind='bar')
    plt.title(title)
    plt.show()

plot_distribution(df, "Distribución - Train")
plot_distribution(val_df, "Distribución - Validation")

# =========================
# SPLIT TRAIN / TEST
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    df['filename'], df['label'], test_size=0.2, random_state=42
)

train_df = pd.DataFrame({'filename': X_train, 'label': y_train})
test_df  = pd.DataFrame({'filename': X_test,  'label': y_test})

# =========================
# DATA AUGMENTATION
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

test_datagen = ImageDataGenerator(rescale=1./255)
val_datagen  = ImageDataGenerator(rescale=1./255)

# =========================
# GENERADORES
# =========================
train_generator = train_datagen.flow_from_dataframe(
    train_df, x_col='filename', y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_dataframe(
    val_df, x_col='filename', y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_datagen.flow_from_dataframe(
    test_df, x_col='filename', y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# =========================
# VISUALIZACIÓN
# =========================
def show_images(generator):
    images, labels = next(generator)
    class_names = list(generator.class_indices.keys())

    plt.figure(figsize=(12, 12))
    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i])
        plt.title(class_names[np.argmax(labels[i])])
        plt.axis('off')
    plt.show()

show_images(train_generator)

# =========================
# MODELO (TRANSFER LEARNING)
# =========================
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling='max'
)

x = base_model.output
x = BatchNormalization()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.45)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adamax(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# ENTRENAMIENTO
# =========================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# =========================
# EVALUACIÓN
# =========================
train_loss, train_acc = model.evaluate(train_generator)
test_loss, test_acc = model.evaluate(test_generator)

print(f"Train Accuracy: {train_acc:.2%}")
print(f"Test Accuracy: {test_acc:.2%}")

# =========================
# MATRIZ DE CONFUSIÓN
# =========================
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
true_classes = test_generator.classes

cm = confusion_matrix(true_classes, y_pred_classes)

plt.figure(figsize=(8,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# =========================
# MÉTRICAS
# =========================
precision = precision_score(true_classes, y_pred_classes, average='weighted')
recall    = recall_score(true_classes, y_pred_classes, average='weighted')
f1        = f1_score(true_classes, y_pred_classes, average='weighted')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# =========================
# GUARDAR MODELO
# =========================
model.save("tomato_model.h5")

# =========================
# PREDICCIÓN DE UNA IMAGEN
# =========================
def predict_image(model, img_path, class_labels):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)

    return class_labels[class_idx]



class_labels = list(train_generator.class_indices.keys())

resultado = predict_image(model, "ejemplo.jpg", class_labels)
print("Predicción:", resultado)