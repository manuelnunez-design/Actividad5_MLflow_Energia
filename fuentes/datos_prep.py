import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def limpiar_datos(df):
    df_limpio = df.copy()
    df_limpio = df_limpio.drop_duplicates()
    df_limpio = df_limpio.dropna()

    df_limpio["Orientation"] = df_limpio["Orientation"].astype("category")
    df_limpio["Glazing_Area_Distribution"] = (
        df_limpio["Glazing_Area_Distribution"].astype("category")
    )

    return df_limpio


def preparar_datos(df, target="Heating_Load"):
    X = df.drop(columns=["Heating_Load", "Cooling_Load"])
    y = df[target]

    X = pd.get_dummies(
        X,
        columns=["Orientation", "Glazing_Area_Distribution"],
        drop_first=True
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test