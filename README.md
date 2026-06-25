# Actividad 5 - Predicción de Carga Energética con MLflow

## Descripción

Este proyecto implementa un flujo completo de Machine Learning para la predicción de la carga de calefacción (Heating Load) utilizando el dataset **Energy Efficiency** del UCI Machine Learning Repository.

El proyecto incluye:

- Preparación y limpieza de datos
- Entrenamiento de modelos de regresión
- Optimización de hiperparámetros mediante GridSearchCV
- Evaluación de desempeño
- Registro de experimentos utilizando MLflow
- Versionamiento mediante Git y GitHub

---

## Dataset

**Nombre**

Energy Efficiency Dataset

**Fuente**

UCI Machine Learning Repository

https://archive.ics.uci.edu/dataset/242/energy+efficiency

**Variable objetivo**

Heating Load (Y1)

**Variables predictoras**

- Relative Compactness
- Surface Area
- Wall Area
- Roof Area
- Overall Height
- Orientation
- Glazing Area
- Glazing Area Distribution

---

## Modelos implementados

Se entrenaron dos modelos de regresión:

1. Regresión Lineal
2. XGBoost Regressor optimizado mediante GridSearchCV

---

## Métricas utilizadas

Para evaluar el desempeño de los modelos se utilizaron:

- MAE
- RMSE
- R²
- Validación Cruzada (5-Fold Cross Validation)

---

## Resultados obtenidos

| Modelo | MAE | RMSE | R² |
|---------|------|------|------|
| Regresión Lineal | 2.059 | 2.872 | 0.921 |
| XGBoost + GridSearchCV | 0.270 | 0.389 | 0.999 |

El modelo XGBoost obtuvo el mejor desempeño en todas las métricas evaluadas.

---

## Estructura del proyecto

```
Actividad5/
│
├── datos/
│   ├── datos_ini/
│   └── datos_limp/
│
├── figuras/
│
├── fuentes/
│   ├── entrena.ipynb
│   ├── datos_prep.py
│   └── train.py
│
├── README.md
├── CHANGELOG.md
├── requirements.txt
└── .gitignore
```

---

## Requisitos

Python 3.11

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

## Ejecución

Preparación de datos

```bash
python fuentes/datos_prep.py
```

Entrenamiento

```bash
python fuentes/train.py
```

Visualización de experimentos

```bash
mlflow ui
```

Abrir:

http://127.0.0.1:5000

---

## Herramientas utilizadas

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- MLflow
- Matplotlib
- Git
- GitHub

---

## Autor

Jesús Manuel Núñez López

Universidad Tecmilenio

Actividad 5 - Gestión de Proyectos de Inteligencia Artificial
