import pandas as pd
import numpy as np


def generar_dataset_ccu(dias=180):
    """
    Genera un dataset demostrativo para la
    Cadena de Suministro Predictiva de CCU.

    Los datos son simulados y tienen fines académicos.
    """

    np.random.seed(42)

    fechas = pd.date_range(
        start="2026-01-01",
        periods=dias,
        freq="D"
    )

    # ========================================================
    # DEMANDA
    # ========================================================

    tendencia = np.linspace(0, 15, dias)

    estacionalidad = (
        12 * np.sin(
            np.arange(dias) * 2 * np.pi / 30
        )
    )

    demanda = (
        150
        + tendencia
        + estacionalidad
        + np.random.normal(0, 18, dias)
    )

    demanda = np.clip(
        demanda,
        60,
        None
    )

    demanda = demanda.round().astype(int)

    # ========================================================
    # VENTAS
    # ========================================================

    ventas = (
        demanda
        + np.random.normal(0, 10, dias)
    )

    ventas = np.clip(
        ventas,
        40,
        demanda
    )

    ventas = ventas.round().astype(int)

    # ========================================================
    # INVENTARIO
    # ========================================================

    inventario = []

    stock = 2200

    for i in range(dias):

        entradas = np.random.randint(
            100,
            350
        )

        salidas = ventas[i]

        stock = stock + entradas - salidas

        stock = max(
            400,
            min(stock, 3000)
        )

        inventario.append(stock)

    # ========================================================
    # PRODUCTOS
    # ========================================================

    categorias = [
        "Cervezas",
        "Bebidas gaseosas",
        "Aguas",
        "Néctares",
        "Bebidas isotónicas",
        "Bebidas energéticas"
    ]

    categoria = np.random.choice(
        categorias,
        dias
    )

    # ========================================================
    # VARIABLES LOGÍSTICAS
    # ========================================================

    lead_time = np.random.randint(
        2,
        8,
        dias
    )

    pedidos = np.random.randint(
        20,
        80,
        dias
    )

    costo_unitario = np.random.randint(
        500,
        1800,
        dias
    )

    # ========================================================
    # DATAFRAME
    # ========================================================

    df = pd.DataFrame({

        "fecha": fechas,

        "categoria": categoria,

        "demanda": demanda,

        "ventas": ventas,

        "inventario": inventario,

        "lead_time": lead_time,

        "pedidos": pedidos,

        "costo_unitario": costo_unitario

    })

    # ========================================================
    # FILL RATE
    # ========================================================

    df["fill_rate"] = np.where(
        df["demanda"] > 0,
        df["ventas"] / df["demanda"],
        0
    )

    # ========================================================
    # VALIDACIÓN
    # ========================================================

    if df.empty:
        raise ValueError(
            "El dataset CCU no pudo ser generado."
        )

    return df
