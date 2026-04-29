# A multivariate correlation approach for detection and classification of leaf diseases in greenhouse tomato crops using deep learning and machine learning models
Este estudio implementa un modelo de red neuronal para la detección y clasificación de enfermedades en hojas de tomate, utilizando imágenes etiquetadas y técnicas de aprendizaje profundo.

<img width="1400" height="1000" alt="Figura 3" src="https://github.com/user-attachments/assets/a10a8c30-553e-4ceb-819a-17c71c03f267" />


# Matriz de confusion
La matriz de confusión permite evaluar el desempeño del modelo en cada una de las clases, mostrando las predicciones correctas y los errores de clasificación.

<img width="1200" height="1000" alt="Figura 7" src="https://github.com/user-attachments/assets/0ed893c6-6f2b-4dd4-b338-e860bbce5d86" />

Se observa un alto número de predicciones correctas en la diagonal principal.
Las clases con mejor desempeño incluyen:
- Late blight
- Healthy
- Tomato mosaic virus
Los errores son mínimos y se presentan principalmente entre enfermedades con características visuales similares.
 # Precision del Modelo
 Evolución de la precisión durante el entrenamiento y validación:

<img width="9648" height="6840" alt="Figure_9a" src="https://github.com/user-attachments/assets/a1886d77-e090-47c6-b8b1-0647b7a91813" />


La precisión alcanza valores cercanos a 99%.
Las curvas de entrenamiento y validación son muy cercanas, lo que indica:
- Buena generalización
- Bajo sobreajuste

# Funcion de perdida

<img width="6432" height="4560" alt="Figure_9b" src="https://github.com/user-attachments/assets/669b8ca4-772c-47c8-a738-8d5d530beb4f" />


La pérdida disminuye rápidamente en las primeras épocas.
Se estabiliza en valores cercanos a 0, indicando:
- Aprendizaje efectivo
- Convergencia del modelo

# Fine-Tuning del Modelo (EfficientNetB0)
El modelo base utilizado corresponde a una red EfficientNetB0 previamente entrenada para la clasificación de 11 clases de enfermedades en hojas de tomate.

Objetivo
El proceso de fine-tuning se enfocó en mejorar el desempeño del modelo en tres clases específicas:
- Healthy
- Early blight
- Late blight
Esto se realizó sin afectar significativamente el conocimiento adquirido en las demás clases.

# Estrategia Implementada
Para lograr este objetivo, se aplicó una estrategia de ajuste fino controlado:
Congelamiento de capas (Layer Freezing)
Se congelaron la mayoría de las capas del modelo:
```python
for layer in model.layers[:-40]:
    layer.trainable = False
```
Esto permite:
- Preservar el conocimiento previo del modelo
- Evitar el catastrophic forgetting
- Reducir el riesgo de sobreajuste
# Entrenamiento parcial
El entrenamiento se realizó únicamente con imágenes de 3 clases.
No se modificó:
- La arquitectura del modelo
- La capa de salida (sigue siendo de 11 clases)
Esto significa que el modelo sigue siendo multiclase, pero con refinamiento en clases específicas.
# Preprocesamiento
Normalización de imágenes:
```python
Rescaling(1./255)
```
No se aplicó data augmentation, lo cual hace que el ajuste sea más controlado y directo sobre los datos originales.

# Configuración de entrenamiento
- Learning rate bajo: 1e-5
- Batch size: 32
- Épocas: 20

Esto permite ajustes finos sin alterar drásticamente los pesos aprendidos.

# Callbacks utilizados
Para optimizar el entrenamiento:
ReduceLROnPlateau
- Reduce la tasa de aprendizaje cuando la pérdida se estanca
EarlyStopping
- Detiene el entrenamiento cuando no hay mejora
- Restaura los mejores pesos
## Modelo de tomate (.h5) link: https://n9.cl/3gux2

# Modelo Random Forest
Este estudio implementa un modelo de Random Forest para predecir el estado del cultivo a partir de variables edáficas y ambientales. A continuación se presentan los principales resultados del modelo

# Matriz de confusión

<img width="700" height="600" alt="Figure 10" src="https://github.com/user-attachments/assets/ba96f979-2c42-41ce-9f70-45a08d561bf6" />

La matriz de confusión permite visualizar el desempeño del modelo comparando las predicciones frente a los valores reales.

- Cada fila representa la clase real
- Cada columna representa la clase predicha
- Los valores en la diagonal indican aciertos del modelo

Esto permite identificar en qué clases el modelo tiene mayor precisión y en cuáles presenta errores de clasificación.

# Reporte de clasificación

<img width="927" height="232" alt="Screenshot 2026-04-29 101106" src="https://github.com/user-attachments/assets/86367c1e-aa80-4b97-a341-77d1ad4ad644" />

El reporte de clasificación resume métricas clave del modelo:
- Precisión (Precision): qué tan confiables son las predicciones positivas
- Recall (Sensibilidad): qué proporción de casos reales se detecta correctamente
- F1-score: balance entre precisión y recall
- Soporte (Support): número de muestras por clase

Este reporte permite evaluar el rendimiento del modelo de manera detallada para cada clase.
# Importancia de variables (Coeficiente de Gini)

<img width="989" height="490" alt="Figura 11" src="https://github.com/user-attachments/assets/219110dd-0aa6-4035-a9f2-afac50ecaa9d" />

La importancia de variables se calcula utilizando el coeficiente de Gini, el cual mide cuánto contribuye cada variable a la reducción de la impureza en los árboles del modelo.

- Valores más altos indican mayor relevancia en la predicción
- Permite identificar las variables más influyentes en el estado del cultivo
- Facilita la interpretación del modelo

# Matriz de correlación (Variables edáficas y ambientales)

<img width="935" height="790" alt="Correlacion" src="https://github.com/user-attachments/assets/1c511545-f7b4-4d1e-aef6-0a237e992afa" />

La matriz de correlación muestra el grado de relación lineal entre las variables del dataset, utilizando el coeficiente de correlación de Pearson.

- Valores cercanos a 1 indican correlación positiva fuerte
- Valores cercanos a -1 indican correlación negativa fuerte
- Valores cercanos a 0 indican poca o ninguna relación

Este análisis permite:

- Identificar variables altamente relacionadas
- Detectar posibles problemas de multicolinealidad
- Comprender mejor la estructura de los datos antes del modelado

# Especificaciones técnicas del equipo
El entrenamiento de los modelos se realizó en un portátil HP Victus 15-fb3019la, cuyas especificaciones permiten ejecutar tareas de cómputo intensivo de manera eficiente.

## Características principales
Procesador (CPU):
- AMD Ryzen 7 7445HS (6 núcleos / 12 hilos, hasta 4.7 GHz)

Tarjeta gráfica (GPU):
- NVIDIA GeForce RTX 3050 (6 GB GDDR6 dedicada)

Memoria RAM:
- 16 GB DDR4
  
Almacenamiento:
- SSD NVMe de 512 GB
  
Pantalla:
- 15.6" Full HD (1920 × 1080)
  
Sistema operativo:
- Windows 11 y Linux
  
Arquitectura:
- 64 bits
