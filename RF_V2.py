"""
=====================================================================
MODELO RANDOM FOREST PARA VARIABLES EDÁFICAS Y AMBIENTALES
=====================================================================

Autor: Mateo Valencia
Ubicación: Universidad del Quindío, Armenia - Colombia

DESCRIPCIÓN:
Este script implementa un modelo de clasificación utilizando Random Forest
para predecir el estado de un cultivo a partir de variables edáficas y
ambientales.

VARIABLE OBJETIVO:
- 'estado' (variable categórica)

VARIABLES DE ENTRADA:
- Propiedades del suelo: pH, Conductividad, N, P, K
- Condiciones del suelo: Humedad y Temperatura
- Condiciones ambientales: Humedad, Temperatura, Radiación

FLUJO DEL SCRIPT:
1. Carga y exploración de datos
2. Preprocesamiento
3. División de datos (train/test)
4. Entrenamiento del modelo
5. Evaluación del modelo
6. Importancia de variables
7. Análisis de correlación
=====================================================================
"""

# =========================
# LIBRERÍAS
# =========================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# =========================
# CARGA DE DATOS
# =========================
ruta = r"C:\Users\teori\OneDrive\Documentos\2_Personal\ProyectoPAI\rf.csv"
df = pd.read_csv(ruta, sep=",")

print("\n=== PREVISUALIZACIÓN DEL DATASET ===")
print(df.head())

print("\n=== VALORES NULOS ===")
print(df.isnull().sum())


# =========================
# SELECCIÓN DE VARIABLES
# =========================
X = df[['pH', 'Conductividad', 'Nitrogeno', 'Fosforo', 'Potasio',
        'HumedadRelativaSuelo', 'TemperaturaSuelo',
        'HumedadRelativaAmbiente', 'TemperaturaAmbiente', 'Radiacion']]

y = df['estado']


# =========================
# DIVISIÓN DE DATOS
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# MODELO BASE
# =========================
modelo_base = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    random_state=42
)

modelo_base.fit(X_train, y_train)


# =========================
# EVALUACIÓN MODELO BASE
# =========================
y_pred_base = modelo_base.predict(X_test)

print("\n=== MODELO BASE ===")
print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred_base))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred_base))
print("\nPrecisión:", accuracy_score(y_test, y_pred_base))


# =========================
# VALIDACIÓN CRUZADA
# =========================
print("\n=== VALIDACIÓN CRUZADA (5-FOLD) ===")

cv_scores = cross_val_score(
    modelo_base,
    X,
    y,
    cv=5,
    scoring='accuracy'
)

print("Precisión por fold:", cv_scores)
print("Promedio:", cv_scores.mean())
print("Desviación estándar:", cv_scores.std())


# =========================
# GRID SEARCH (OPTIMIZACIÓN)
# =========================
print("\n=== GRID SEARCH ===")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\nMejores parámetros encontrados:")
print(grid_search.best_params_)

# Modelo optimizado
modelo_opt = grid_search.best_estimator_


# =========================
# EVALUACIÓN MODELO OPTIMIZADO
# =========================
y_pred_opt = modelo_opt.predict(X_test)

print("\n=== MODELO OPTIMIZADO ===")
print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred_opt))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred_opt))
print("\nPrecisión:", accuracy_score(y_test, y_pred_opt))


# =========================
# COMPARACIÓN DE MODELOS
# =========================
acc_base = accuracy_score(y_test, y_pred_base)
acc_opt = accuracy_score(y_test, y_pred_opt)

print("\n=== COMPARACIÓN ===")
print(f"Modelo Base: {acc_base:.4f}")
print(f"Modelo Optimizado: {acc_opt:.4f}")


# =========================
# IMPORTANCIA DE VARIABLES
# =========================
nombres_simplificados = {
    'pH': 'pH',
    'Conductividad': 'CE',
    'Nitrogeno': 'N',
    'Fosforo': 'P',
    'Potasio': 'K',
    'HumedadRelativaSuelo': 'SL_HR',
    'TemperaturaSuelo': 'SL_Temp',
    'HumedadRelativaAmbiente': 'Amb_HR',
    'TemperaturaAmbiente': 'Amb_Temp',
    'Radiacion': 'Rad'
}

importancias = pd.DataFrame({
    'Variable': X.columns.map(nombres_simplificados),
    'Importancia': modelo_opt.feature_importances_ * 100
}).sort_values('Importancia', ascending=False)

plt.figure(figsize=(10, 5))
barras = plt.bar(importancias['Variable'], importancias['Importancia'])

for barra in barras:
    altura = barra.get_height()
    plt.text(
        barra.get_x() + barra.get_width() / 2,
        altura,
        f"{altura:.1f}%",
        ha='center',
        va='bottom',
        fontsize=9
    )

plt.xticks(rotation=45)
plt.ylabel('Importancia (%)')
plt.xlabel('Variables')
plt.title('Importancia de Variables (Modelo Optimizado)')
plt.tight_layout()
plt.show()


# =========================
# MATRIZ DE CORRELACIÓN
# =========================
corr_matrix = X.corr(method='pearson')

print("\n=== MATRIZ DE CORRELACIÓN ===")
print(corr_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix,
            annot=True,
            cmap='coolwarm',
            fmt=".2f",
            linewidths=0.5)

plt.title("Matriz de Correlación - Variables Edáficas y Ambientales")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()