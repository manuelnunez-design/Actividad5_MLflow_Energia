# CHANGELOG

Todas las modificaciones relevantes realizadas durante el desarrollo del proyecto.

---

## Versión 1.0 (Junio 2026)

### Preparación del proyecto

- Creación de la estructura de carpetas.
- Configuración del entorno virtual.
- Instalación de dependencias.
- Configuración de MLflow.

---

### Preparación de datos

- Descarga del dataset Energy Efficiency.
- Limpieza de datos.
- Verificación de valores nulos.
- Separación de variables predictoras y objetivo.
- Escalamiento de variables numéricas.
- Almacenamiento del dataset limpio.

---

### Modelado

Se implementaron dos modelos:

- Regresión Lineal
- XGBoost Regressor

Posteriormente se optimizó XGBoost mediante GridSearchCV.

---

### Evaluación

Se calcularon las siguientes métricas:

- MAE
- RMSE
- R²

Además se aplicó Validación Cruzada de 5 particiones.

---

### MLflow

Se registraron automáticamente:

- Parámetros
- Métricas
- Modelos entrenados
- Artefactos
- Figuras generadas

---

### Versionamiento

El proyecto fue versionado utilizando Git y publicado en GitHub.

Repositorio:

https://github.com/manuelnunez-design/Actividad5_MLflow_Energia

---

## Estado final

Proyecto terminado y funcional.
