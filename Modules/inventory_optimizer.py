import numpy as np
import pandas as pd


def optimizar_inventario(df=None):
    """
    Calcula parámetros básicos de reposición para la
    Cadena de Suministro Predictiva de CCU.

    Modelo:
    - Demanda diaria
    - Lead Time
    - Stock de seguridad
    - Punto de reorden (ROP)
    - EOQ
    - Pedido sugerido

    Los parámetros económicos son referenciales
    y pueden reemplazarse posteriormente por datos reales.
    """

    # ========================================================
    # VALORES BASE
    # ========================================================

    costo_pedir = 45000
    costo_mantener = 3300
    dias_trabajo = 300

    # ========================================================
    # DATOS DEL DATASET
    # ========================================================

    if df is not None and not df.empty:

        # ----------------------------------------------------
        # Demanda anual estimada
        # ----------------------------------------------------

        if "demanda" in df.columns:

            demanda_diaria = float(
                pd.to_numeric(
                    df["demanda"],
                    errors="coerce"
                )
                .fillna(0)
                .mean()
            )

        else:

            demanda_diaria = 0.0

        demanda_anual = (
            demanda_diaria *
            dias_trabajo
        )

        # ----------------------------------------------------
        # Lead Time
        # ----------------------------------------------------

        if "lead_time" in df.columns:

            lead_time = float(
                pd.to_numeric(
                    df["lead_time"],
                    errors="coerce"
                )
                .fillna(5)
                .mean()
            )

        else:

            lead_time = 5.0

        # ----------------------------------------------------
        # Inventario actual
        # ----------------------------------------------------

        if "inventario" in df.columns:

            inventario_actual = float(
                pd.to_numeric(
                    df["inventario"],
                    errors="coerce"
                )
                .fillna(0)
                .iloc[-1]
            )

        else:

            inventario_actual = 0.0

        # ----------------------------------------------------
        # Variabilidad de demanda
        # ----------------------------------------------------

        if "demanda" in df.columns:

            demanda_std = float(
                pd.to_numeric(
                    df["demanda"],
                    errors="coerce"
                )
                .fillna(0)
                .std()
            )

        else:

            demanda_std = 0.0

    else:

        demanda_diaria = 40.0
        demanda_anual = demanda_diaria * dias_trabajo
        lead_time = 5.0
        inventario_actual = 150.0
        demanda_std = 10.0

    # ========================================================
    # PROTECCIÓN CONTRA VALORES INVÁLIDOS
    # ========================================================

    demanda_diaria = max(
        demanda_diaria,
        0
    )

    demanda_anual = max(
        demanda_anual,
        1
    )

    lead_time = max(
        lead_time,
        1
    )

    demanda_std = max(
        demanda_std,
        1
    )

    # ========================================================
    # STOCK DE SEGURIDAD
    # ========================================================

    nivel_servicio = 1.65

    stock_seguridad = (
        nivel_servicio *
        demanda_std *
        np.sqrt(lead_time)
    )

    # ========================================================
    # PUNTO DE REORDEN
    # ========================================================

    demanda_lead_time = (
        demanda_diaria *
        lead_time
    )

    reorder_point = (
        demanda_lead_time +
        stock_seguridad
    )

    # ========================================================
    # EOQ
    # ========================================================

    eoq = np.sqrt(
        (
            2 *
            demanda_anual *
            costo_pedir
        )
        /
        costo_mantener
    )

    # ========================================================
    # PEDIDO SUGERIDO
    # ========================================================

    if inventario_actual < reorder_point:

        suggested_order = max(
            0,
            round(
                max(
                    eoq,
                    reorder_point -
                    inventario_actual
                )
            )
        )

    else:

        suggested_order = 0

    # ========================================================
    # COBERTURA ACTUAL
    # ========================================================

    if demanda_diaria > 0:

        cobertura_actual = (
            inventario_actual /
            demanda_diaria
        )

    else:

        cobertura_actual = 0

    # ========================================================
    # NIVEL DE RIESGO
    # ========================================================

    if inventario_actual < (
        demanda_lead_time
    ):

        riesgo = "ALTO"

    elif inventario_actual < reorder_point:

        riesgo = "MEDIO"

    else:

        riesgo = "BAJO"

    # ========================================================
    # RECOMENDACIÓN
    # ========================================================

    if riesgo == "ALTO":

        recomendacion = (
            "Reposición prioritaria: "
            "el inventario se encuentra "
            "por debajo de la demanda "
            "esperada durante el lead time."
        )

    elif riesgo == "MEDIO":

        recomendacion = (
            "Monitorear y programar reposición: "
            "el inventario se aproxima al "
            "punto de reorden."
        )

    else:

        recomendacion = (
            "Operación estable: "
            "el inventario se encuentra "
            "por sobre el punto de reorden."
        )

    # ========================================================
    # RETORNO
    # ========================================================

    return {

        "eoq": float(
            round(eoq, 1)
        ),

        "reorder_point": float(
            round(reorder_point, 1)
        ),

        "stock_seguridad": float(
            round(stock_seguridad, 1)
        ),

        "suggested_order": int(
            suggested_order
        ),

        "inventario_actual": float(
            round(inventario_actual, 1)
        ),

        "demanda_diaria": float(
            round(demanda_diaria, 1)
        ),

        "lead_time": float(
            round(lead_time, 1)
        ),

        "cobertura_actual": float(
            round(cobertura_actual, 1)
        ),

        "riesgo": riesgo,

        "recomendacion": recomendacion
    }
