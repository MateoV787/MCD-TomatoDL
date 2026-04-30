"""
=====================================================================
RANDOM FOREST MODEL FOR SOIL AND ENVIRONMENTAL VARIABLES
=====================================================================

Author: Mateo Valencia
Location: Universidad del Quindío, Armenia - Colombia

DESCRIPTION:
This script implements a classification model using Random Forest
to predict the state of a crop based on soil and environmental variables.

TARGET VARIABLE:
- 'estado' (categorical variable)

INPUT VARIABLES:
- Soil properties: pH, Conductivity, N, P, K
- Soil conditions: Humidity and Temperature
- Environmental conditions: Humidity, Temperature, Radiation

SCRIPT FLOW:
1. Data loading and exploration
2. Preprocessing
3. Data splitting (train/test)
4. Model training
5. Model evaluation
6. Feature importance
7. Correlation analysis
=====================================================================
"""

# =========================
# LIBRARIES
# =========================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# =========================
# DATA LOADING
# =========================
ruta = r"C:\Users\teori\OneDrive\Documentos\2_Personal\ProyectoPAI\rf.csv"
df = pd.read_csv(ruta, sep=",")

print("\n=== DATASET PREVIEW ===")
print(df.head())

print("\n=== NULL VALUES ===")
print(df.isnull().sum())


# =========================
# VARIABLE SELECTION
# =========================
X = df[['pH', 'Conductividad', 'Nitrogeno', 'Fosforo', 'Potasio',
        'HumedadRelativaSuelo', 'TemperaturaSuelo',
        'HumedadRelativaAmbiente', 'TemperaturaAmbiente', 'Radiacion']]

y = df['estado']


# =========================
# DATA SPLITTING
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# BASE MODEL
# =========================
modelo_base = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    random_state=42
)

modelo_base.fit(X_train, y_train)


# =========================
# BASE MODEL EVALUATION
# =========================
y_pred_base = modelo_base.predict(X_test)

print("\n===BASE MODELE ===")
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred_base))
print("\nClassification report:\n", classification_report(y_test, y_pred_base))
print("\nAccuracy:", accuracy_score(y_test, y_pred_base))

# =========================
# CROSS-VALIDATION
# =========================
print("\n=== CROSS-VALIDATION (5-FOLD) ===")

cv_scores = cross_val_score(
    modelo_base,
    X,
    y,
    cv=5,
    scoring='accuracy'
)

print("Accuracy per fold:", cv_scores)
print("Mean:", cv_scores.mean())
print("Standard deviation:", cv_scores.std())


# =========================
# GRID SEARCH (OPTIMIZATION)
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

print("\nBest parameters found:")
print(grid_search.best_params_)

# Optimized model
modelo_opt = grid_search.best_estimator_

# =========================
# OPTIMIZED MODEL EVALUATION
# =========================
y_pred_opt = modelo_opt.predict(X_test)

print("\n=== OPTIMIZED MODEL ===")
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred_opt))
print("\nClassification report:\n", classification_report(y_test, y_pred_opt))
print("\nAccuracy:", accuracy_score(y_test, y_pred_opt))

# =========================
# MODEL COMPARISON
# =========================
acc_base = accuracy_score(y_test, y_pred_base)
acc_opt = accuracy_score(y_test, y_pred_opt)

print("\n=== COMPARISON ===")
print(f"Modelo Base: {acc_base:.4f}")
print(f"Modelo Optimizado: {acc_opt:.4f}")

# =========================
# FEATURE IMPORTANCE
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
plt.ylabel('Importance (%)')
plt.xlabel('Variables')
plt.title('Feature Importance (Optimized Model)')
plt.tight_layout()
plt.show()


# =========================
# CORRELATION MATRIX
# =========================
corr_matrix = X.corr(method='pearson')
print("\n=== CORRELATION MATRIX ===")
print(corr_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix,
            annot=True,
            cmap='coolwarm',
            fmt=".2f",
            linewidths=0.5)

plt.title("Correlation Matrix - Soil and Environmental Variables")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
