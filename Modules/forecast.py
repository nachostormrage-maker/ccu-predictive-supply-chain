import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def generar_forecast(df):
    """
    Genera un pronóstico de demanda para la
    Cadena de Suministro Predictiva de CCU.

    El modelo utiliza una regresión lineal simple
    sobre la demanda histórica.

    Retorna:
        DataFrame con fecha, demanda, forecast e inventario.
    """

    df = df.copy()

    # ========================================================
    # VALIDACIÓN
    # ========================================================

    if df.empty:
        return df

    # ========================================================
    # ORDEN TEMPORAL
    # ========================================================

    if "fecha" in df.columns:

        df["fecha"] = pd.to_datetime(
            df["fecha"],
            errors="coerce"
        )

        df = (
            df.dropna(subset=["fecha"])
            .sort_values("fecha")
            .reset_index(drop=True)
        )

    # ========================================================
    # FORECAST
    # ========================================================

    if "demanda" not in df.columns:

        df["forecast"] = np.nan

    else:

        demanda = pd.to_numeric(
            df["demanda"],
            errors="coerce"
        )

        demanda = demanda.fillna(
            demanda.mean()
        )

        # ----------------------------------------------------
        # Si existen suficientes datos:
        # regresión lineal
        # ----------------------------------------------------

        if len(df) >= 14:

            df["t"] = np.arange(
                len(df)
            )

            X = df[["t"]]
            y = demanda

            try:

                modelo = LinearRegression()

                modelo.fit(
                    X,
                    y
                )

                df["forecast"] = modelo.predict(X)

            except Exception:

                df["forecast"] = (
                    demanda
                    .rolling(
                        window=7,
                        min_periods=1
                    )
                    .mean()
                )

        else:

            # ------------------------------------------------
            # Fallback para datasets pequeños
            # ------------------------------------------------

            df["forecast"] = (
                demanda
                .rolling(
                    window=7,
                    min_periods=1
                )
                .mean()
            )

    # ========================================================
    # INVENTARIO
    # ========================================================

    if "inventario" not in df.columns:

        df["inventario"] = 0

    # ========================================================
    # SUAVIZADO DEL FORECAST
    # ========================================================

    if "forecast" in df.columns:

        df["forecast"] = (
            pd.to_numeric(
                df["forecast"],
                errors="coerce"
            )
            .fillna(
                df["demanda"].mean()
                if "demanda" in df.columns
                else 0
            )
        )

        df["forecast"] = df["forecast"].clip(
            lower=0
        )

    # ========================================================
    # COLUMNAS FINALES
    # ========================================================

    columnas = [
        "fecha",
        "demanda",
        "forecast",
        "inventario"
    ]

    columnas = [
        col
        for col in columnas
        if col in df.columns
    ]

    return df[columnas]
