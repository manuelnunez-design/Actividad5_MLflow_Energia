import os
import sys
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

sys.path.append(os.path.dirname(__file__))
from datos_prep import limpiar_datos, preparar_datos


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUTA_DATOS = os.path.join(BASE_DIR, "datos", "datos_ini", "energy_efficiency_original.csv")
RUTA_LIMPIO = os.path.join(BASE_DIR, "datos", "datos_limp", "energy_efficiency_clean.csv")
RUTA_FIGURAS = os.path.join(BASE_DIR, "figuras")

os.makedirs(RUTA_FIGURAS, exist_ok=True)

mlflow.set_tracking_uri("file:" + os.path.join(BASE_DIR, "mlruns"))
mlflow.set_experiment("Prediccion_Carga_Energetica")


def entrenar_modelo(nombre_modelo, modelo, X_train, X_test, y_train, y_test, parametros=None):
    with mlflow.start_run(run_name=nombre_modelo):

        if parametros:
            mlflow.log_params(parametros)

        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="r2")

        mlflow.log_param("modelo", nombre_modelo)
        mlflow.log_param("target", "Heating_Load")
        mlflow.log_param("cv_folds", 5)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("CV_R2_mean", cv_scores.mean())
        mlflow.log_metric("CV_R2_std", cv_scores.std())

        grafica = os.path.join(RUTA_FIGURAS, f"{nombre_modelo}.png")

        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.xlabel("Valor real")
        plt.ylabel("Valor predicho")
        plt.title(f"Real vs Predicho - {nombre_modelo}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(grafica, dpi=300)
        plt.close()

        mlflow.log_artifact(grafica)
        mlflow.sklearn.log_model(modelo, nombre_modelo)

        return {
            "Modelo": nombre_modelo,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "CV_R2_mean": cv_scores.mean(),
            "CV_R2_std": cv_scores.std()
        }


def main():
    df = pd.read_csv(RUTA_DATOS)

    df_limpio = limpiar_datos(df)
    df_limpio.to_csv(RUTA_LIMPIO, index=False)

    X_train, X_test, y_train, y_test = preparar_datos(df_limpio)

    resultados = []

    modelo_lr = LinearRegression()

    resultados.append(
        entrenar_modelo(
            "Regresion_Lineal",
            modelo_lr,
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    xgb_base = XGBRegressor(
        objective="reg:squarederror",
        random_state=42
    )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [2, 3, 5],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0]
    }

    grid_xgb = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    grid_xgb.fit(X_train, y_train)

    resultados.append(
        entrenar_modelo(
            "XGBoost_GridSearch",
            grid_xgb.best_estimator_,
            X_train,
            X_test,
            y_train,
            y_test,
            parametros=grid_xgb.best_params_
        )
    )

    resultados_df = pd.DataFrame(resultados)
    resultados_df.to_csv(
        os.path.join(RUTA_FIGURAS, "comparacion_modelos.csv"),
        index=False
    )

    print(resultados_df)


if __name__ == "__main__":
    main()