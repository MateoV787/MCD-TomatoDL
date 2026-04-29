"""
=====================================================================
FINE-TUNING PARCIAL DE EfficientNetB0 (MODELO DE 11 CLASES)
=====================================================================

Autor: Mateo Valencia
Ubicación: Universidad del Quindío, Armenia - Colombia

DESCRIPCIÓN:
Este script realiza fine-tuning sobre un modelo preentrenado basado en
EfficientNetB0 que originalmente clasifica 11 clases.

OBJETIVO:
Refinar el desempeño del modelo en 3 clases específicas:
- healthy
- Early_blight
- Late_blight

SIN:
- Modificar la arquitectura
- Cambiar la capa de salida (sigue siendo de 11 clases)
- Aplicar data augmentation

ESTRATEGIA:
- Se entrena SOLO con imágenes de 3 clases
- Se congelan la mayoría de capas
- Se ajustan capas finales con learning rate bajo
- Se minimiza el riesgo de "catastrophic forgetting"

"""

# =========================================================
#IMPORTACIÓN DE LIBRERÍAS
# =========================================================
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

TRAIN_DIR = r"C:\Users\teori\OneDrive\Documentos\2_Personal\ProyectoPAI\Dataset_UQ\train"   
VAL_DIR = r"C:\Users\teori\OneDrive\Documentos\2_Personal\ProyectoPAI\Dataset_UQ\valid"

MODEL_PATH = r"C:\Users\teori\OneDrive\Documentos\2_Personal\ProyectoPAI\tomato_model.h5"
OUTPUT_MODEL = r"C:\Users\teori\OneDrive\Documentos\2_Personal\ProyectoPAI\modelo_refinado.h5"

# =========================================================
# CARGA DEL DATASET
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
# PREPROCESAMIENTO
# =========================================================

normalization = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization(x), y))
val_ds = val_ds.map(lambda x, y: (normalization(x), y))

# Mejora de rendimiento
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# =========================================================
# CARGA DEL MODELO PREENTRENADO
# =========================================================


model = tf.keras.models.load_model(MODEL_PATH)

model.summary()

# =========================================================
# FREEZING ESTRATÉGICO
# =========================================================


"""
Se congelan la mayoría de capas para:
-Preservar conocimiento de las otras 8 clases
-Evitar sobreajuste
-Mantener estabilidad del modelo

Solo se entrenan las capas finales
"""

for layer in model.layers[:-40]:  # Para evitar Catastrophic forgetting
    layer.trainable = False

# =========================================================
# COMPILACIÓN
# =========================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================================================
# CALLBACKS
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
# ENTRENAMIENTO
# =========================================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[lr_scheduler, early_stop]
)

# =========================================================
# GUARDADO DEL MODELO
# =========================================================

model.save(OUTPUT_MODEL)
