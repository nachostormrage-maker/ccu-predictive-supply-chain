import pandas as pd
import numpy as np


def calcular_kpis(df):
    """
    Calcula KPIs principales de la cadena de suministro de CCU.

    Retorna:
        fill_rate
        mae
        desviacion_demanda
        inventario_prom
        cobertura_inventario
    """

    res = {
        "fill_rate": 0.0,
        "mae": 0.0,
        "desviacion_demanda": 0.0,
        "inventario_prom": 0.0,
        "cobertura_inventario": 0.0
    }

    if df is None or df.empty:
        return res

    # ========================================================
    # DEMANDA Y VENTAS
    # ========================================================

    if "demanda" in df.columns:

        demanda = pd.to_numeric(
            df["demanda"],
            errors="coerce"
        ).fillna(0)

    else:

        demanda = pd.Series(
            0,
            index=df.index
        )

    if "ventas" in df.columns:

        ventas = pd.to_numeric(
            df["ventas"],
            errors="coerce"
        ).fillna(0)

    else:

        ventas = pd.Series(
            0,
            index=df.index
        )

    # ========================================================
    # FILL RATE
    # ========================================================

    mask = demanda > 0

    if mask.any():

        fill_rate = (
            ventas[mask] /
            demanda[mask]
        ).mean()

        fill_rate = np.clip(
            fill_rate,
            0,
            1
        )

        res["fill_rate"] = float(
            fill_rate
        )

    # ========================================================
    # ERROR DE DEMANDA
    # ========================================================

    error = (
        demanda -
        ventas
    ).abs()

    res["mae"] = float(
        error.mean()
    )

    res["desviacion_demanda"] = float(
        error.mean()
    )

    # ========================================================
    # INVENTARIO PROMEDIO
    # ========================================================

    if "inventario" in df.columns:

        inventario = pd.to_numeric(
            df["inventario"],
            errors="coerce"
        ).fillna(0)

        res["inventario_prom"] = float(
            inventario.mean()
        )

        # ----------------------------------------------------
        # Cobertura aproximada en días
        # ----------------------------------------------------

        demanda_diaria = demanda.mean()

        if demanda_diaria > 0:

            res["cobertura_inventario"] = float(
                inventario.mean() /
                demanda_diaria
            )

    return res


def clasificacion_abc(df):
    """
    Clasificación ABC de productos según demanda acumulada.

    A = hasta 80% del valor acumulado
    B = 80% a 95%
    C = sobre 95%

    Si no existe SKU, utiliza categoría.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    # ========================================================
    # IDENTIFICAR PRODUCTO
    # ========================================================

    if "sku" in df.columns:

        columna_producto = "sku"

    elif "categoria" in df.columns:

        columna_producto = "categoria"

    else:

        return pd.DataFrame()

    # ========================================================
    # DEMANDA
    # ========================================================

    if "demanda" not in df.columns:
        return pd.DataFrame()

    resumen = (
        df.groupby(columna_producto)["demanda"]
        .sum()
        .reset_index()
    )

    resumen = resumen.rename(
        columns={
            columna_producto: "Producto",
            "demanda": "Demanda Total"
        }
    )

    # ========================================================
    # ORDENAR
    # ========================================================

    resumen = resumen.sort_values(
        "Demanda Total",
        ascending=False
    ).reset_index(
        drop=True
    )

    total = resumen["Demanda Total"].sum()

    if total <= 0:

        resumen["Participación"] = 0.0
        resumen["Acumulado"] = 0.0
        resumen["Clasificación"] = "C"

        return resumen

    # ========================================================
    # PARTICIPACIÓN
    # ========================================================

    resumen["Participación"] = (
        resumen["Demanda Total"] /
        total
    )

    resumen["Acumulado"] = (
        resumen["Participación"]
        .cumsum()
    )

    # ========================================================
    # CLASIFICACIÓN ABC
    # ========================================================

    def clasificar(valor):

        if valor <= 0.80:
            return "A"

        elif valor <= 0.95:
            return "B"

        return "C"

    resumen["Clasificación"] = (
        resumen["Acumulado"]
        .apply(clasificar)
    )

    # ========================================================
    # FORMATO
    # ========================================================

    resumen["Participación"] = (
        resumen["Participación"]
        .round(4)
    )

    resumen["Acumulado"] = (
        resumen["Acumulado"]
        .round(4)
    )

    return resumen
