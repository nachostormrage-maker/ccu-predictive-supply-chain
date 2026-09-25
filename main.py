import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
from datetime import datetime

from sklearn.linear_model import LinearRegression


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="CCU | Predictive Supply Chain",
    page_icon="🍺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# ESTILO
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #f4f7fb;
}

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #123b5d;
    margin-bottom: 2px;
}

.subtitle {
    color: #64748b;
    font-size: 15px;
    margin-bottom: 18px;
}

.section {
    font-size: 22px;
    font-weight: 750;
    color: #123b5d;
    margin-top: 24px;
    margin-bottom: 12px;
}

.metric-box {
    background: white;
    border-radius: 14px;
    padding: 17px;
    border: 1px solid #dfe7ef;
    min-height: 112px;
    box-shadow: 0 2px 8px rgba(15,23,42,.04);
}

.metric-title {
    color: #64748b;
    font-size: 13px;
}

.metric-value {
    color: #123b5d;
    font-size: 27px;
    font-weight: 800;
}

.metric-delta {
    color: #64748b;
    font-size: 12px;
    margin-top: 4px;
}

.executive {
    background: white;
    border-left: 5px solid #176b9c;
    border-radius: 10px;
    padding: 18px;
    border-top: 1px solid #e2e8f0;
    border-right: 1px solid #e2e8f0;
    border-bottom: 1px solid #e2e8f0;
}

.realtime {
    background: linear-gradient(
        90deg,
        #ecfdf5,
        #f0fdfa
    );
    border: 1px solid #a7f3d0;
    color: #065f46;
    border-radius: 10px;
    padding: 9px 15px;
    margin-bottom: 18px;
    font-size: 13px;
    font-weight: 600;
}

.signal-green {
    color: #16a34a;
    font-weight: 800;
}

.signal-yellow {
    color: #d97706;
    font-weight: 800;
}

.signal-red {
    color: #dc2626;
    font-weight: 800;
}

.small-note {
    color: #64748b;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# FUNCIONES DEL SISTEMA
# ============================================================

def generar_dataset():

    np.random.seed(42)

    dias = 180

    fechas = pd.date_range(
        start="2026-01-01",
        periods=dias,
        freq="D"
    )

    demanda = np.clip(
        np.random.normal(150, 28, dias),
        70,
        None
    )

    ventas = np.clip(
        demanda * np.random.normal(
            0.96,
            0.035,
            dias
        ),
        50,
        None
    )

    inventario = np.clip(
        1900 +
        np.cumsum(
            np.random.normal(0, 25, dias)
        ),
        800,
        3000
    )

    return pd.DataFrame({
        "fecha": fechas,
        "demanda": demanda,
        "ventas": ventas,
        "inventario": inventario,
        "sku": np.random.choice(
            [
                "Cerveza",
                "Bebida",
                "Agua",
                "Néctar"
            ],
            dias
        ),
        "lead_time": np.random.randint(
            2,
            8,
            dias
        ),
        "costo_unitario": np.random.randint(
            800,
            1500,
            dias
        )
    })


# ============================================================
# FORECAST
# ============================================================

def generar_forecast(df):

    df = df.copy()

    df["fecha"] = pd.to_datetime(
        df["fecha"],
        errors="coerce"
    )

    df = (
        df
        .dropna(subset=["fecha"])
        .sort_values("fecha")
        .reset_index(drop=True)
    )

    demanda = pd.to_numeric(
        df["demanda"],
        errors="coerce"
    ).fillna(0)

    if len(df) >= 14:

        df["t"] = np.arange(len(df))

        modelo = LinearRegression()

        try:

            modelo.fit(
                df[["t"]],
                demanda
            )

            df["forecast"] = modelo.predict(
                df[["t"]]
            )

        except Exception:

            df["forecast"] = (
                demanda
                .rolling(
                    7,
                    min_periods=1
                )
                .mean()
            )

    else:

        df["forecast"] = (
            demanda
            .rolling(
                7,
                min_periods=1
            )
            .mean()
        )

    df["forecast"] = (
        pd.to_numeric(
            df["forecast"],
            errors="coerce"
        )
        .fillna(demanda.mean())
        .clip(lower=0)
    )

    return df


# ============================================================
# KPIs
# ============================================================

def calcular_kpis(df):

    demanda = pd.to_numeric(
        df["demanda"],
        errors="coerce"
    )

    ventas = pd.to_numeric(
        df["ventas"],
        errors="coerce"
    )

    mask = demanda > 0

    if mask.any():

        fill_rate = (
            ventas[mask] /
            demanda[mask]
        ).mean()

    else:

        fill_rate = 0

    mae = (
        demanda -
        ventas
    ).abs().mean()

    inventario_prom = (
        df["inventario"]
        .mean()
    )

    return {
        "fill_rate": float(fill_rate),
        "mae": float(mae),
        "inventario_prom": float(
            inventario_prom
        )
    }


# ============================================================
# OPTIMIZACIÓN
# ============================================================

def optimizar_inventario(
    demanda_diaria,
    lead_time,
    servicio
):

    demanda_anual = (
        demanda_diaria *
        300
    )

    costo_pedir = 45000
    costo_mantener = 3300

    eoq = np.sqrt(
        (
            2 *
            demanda_anual *
            costo_pedir
        )
        /
        costo_mantener
    )

    desviacion = (
        demanda_diaria *
        0.20
    )

    z = 1.65

    if servicio >= 97:
        z = 1.88

    elif servicio >= 95:
        z = 1.65

    elif servicio >= 90:
        z = 1.28

    else:
        z = 1.04

    stock_seguridad = (
        z *
        desviacion *
        np.sqrt(lead_time)
    )

    reorder_point = (
        demanda_diaria *
        lead_time
        +
        stock_seguridad
    )

    return {
        "eoq": float(eoq),
        "stock_seguridad": float(
            stock_seguridad
        ),
        "reorder_point": float(
            reorder_point
        )
    }


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 🍺 CCU")

st.sidebar.caption(
    "Predictive Supply Chain Control Tower"
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    "### Parámetros de simulación"
)

demanda_factor = st.sidebar.slider(
    "Variación de demanda (%)",
    -30,
    50,
    0,
    5
)

lead_time = st.sidebar.slider(
    "Lead Time (días)",
    1,
    15,
    5
)

capacidad = st.sidebar.slider(
    "Capacidad logística (%)",
    50,
    120,
    100,
    5
)

nivel_servicio_obj = st.sidebar.slider(
    "Nivel de servicio objetivo (%)",
    85,
    99,
    95
)

escenario = st.sidebar.selectbox(
    "Escenario",
    [
        "Base",
        "Aumento de demanda",
        "Restricción logística",
        "Alta demanda + restricción"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Mueve los parámetros para observar "
    "cómo cambia la cadena de suministro."
)


# ============================================================
# DATA BASE
# ============================================================

df_base = generar_dataset()


# ============================================================
# ESCENARIO
# ============================================================

demanda_extra = demanda_factor / 100

capacidad_real = capacidad
lead_time_real = lead_time


if escenario == "Aumento de demanda":

    demanda_extra += 0.20


elif escenario == "Restricción logística":

    capacidad_real *= 0.70
    lead_time_real += 3


elif escenario == "Alta demanda + restricción":

    demanda_extra += 0.25
    capacidad_real *= 0.70
    lead_time_real += 3


# ============================================================
# SIMULACIÓN
# ============================================================

df = df_base.copy()

df["demanda"] = (
    df["demanda"] *
    (1 + demanda_extra)
)

df["ventas"] = np.minimum(
    df["ventas"] *
    (
        1 +
        demanda_extra *
        0.45
    ),
    df["demanda"] *
    (
        capacidad_real /
        100
    )
)

consumo = np.maximum(
    0,
    df["demanda"] -
    df["ventas"]
)

df["inventario"] = (
    df_base["inventario"]
    -
    consumo.cumsum()
    *
    0.45
    *
    (
        lead_time_real /
        5
    )
)

df["inventario"] = (
    df["inventario"]
    .clip(
        lower=100
    )
)


# ============================================================
# FORECAST
# ============================================================

df_fc = generar_forecast(df)


# ============================================================
# KPIs
# ============================================================

kpis = calcular_kpis(df)

fill_rate = kpis["fill_rate"]

inventario_actual = (
    df["inventario"].iloc[-1]
)

demanda_promedio = (
    df["demanda"].mean()
)

cobertura = (
    inventario_actual /
    max(
        demanda_promedio,
        1
    )
)

utilizacion = min(
    100,
    (
        demanda_promedio /
        (
            demanda_promedio *
            capacidad_real /
            100
        )
    ) *
    100
)

riesgo_quiebre = max(
    0,
    min(
        100,
        100 -
        cobertura * 7
        +
        max(
            0,
            lead_time_real - 5
        ) * 2
    )
)


# ============================================================
# ESTADO GENERAL
# ============================================================

# IMPORTANTE:
# El estado ahora considera TODOS los indicadores.
# No puede quedar "ESTABLE" si hay señales críticas.

if (
    fill_rate < nivel_servicio_obj / 100
    and riesgo_quiebre >= 30
):

    estado = "RIESGO CRÍTICO"
    color_estado = "#dc2626"

elif (
    fill_rate < nivel_servicio_obj / 100
    or riesgo_quiebre >= 30
    or cobertura < 5
    or utilizacion >= 100
):

    estado = "RIESGO"
    color_estado = "#dc2626"

elif (
    riesgo_quiebre >= 15
    or fill_rate < 0.90
    or cobertura < 10
):

    estado = "MONITOREAR"
    color_estado = "#f59e0b"

else:

    estado = "ESTABLE"
    color_estado = "#16a34a"


# ============================================================
# COLORES DINÁMICOS KPI
# ============================================================

def color_servicio(valor, objetivo):

    if valor >= objetivo:
        return "#16a34a"

    elif valor >= objetivo - 0.05:
        return "#f59e0b"

    return "#dc2626"


def color_riesgo(valor):

    if valor < 15:
        return "#16a34a"

    elif valor < 30:
        return "#f59e0b"

    return "#dc2626"


def color_cobertura(valor):

    if valor >= 10:
        return "#16a34a"

    elif valor >= 5:
        return "#f59e0b"

    return "#dc2626"


color_servicio_actual = color_servicio(
    fill_rate,
    nivel_servicio_obj / 100
)

color_riesgo_actual = color_riesgo(
    riesgo_quiebre
)

color_cobertura_actual = color_cobertura(
    cobertura
)

color_utilizacion = (
    "#dc2626"
    if utilizacion >= 100
    else
    "#f59e0b"
    if utilizacion >= 90
    else
    "#16a34a"
)


# ============================================================
# OPTIMIZACIÓN
# ============================================================

optim = optimizar_inventario(
    demanda_promedio,
    lead_time_real,
    nivel_servicio_obj
)


# ============================================================
# FECHA / HORA DE ACTUALIZACIÓN
# ============================================================

ahora = datetime.now()

hora_actualizacion = ahora.strftime(
    "%d-%m-%Y %H:%M:%S"
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
    CCU | Predictive Supply Chain Control Tower
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Sistema ejecutivo de monitoreo, predicción y simulación
    de la cadena de suministro.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="realtime">
    🟢 MOTOR PREDICTIVO ACTIVO
    &nbsp; | &nbsp;
    Última actualización: {hora_actualizacion}
    &nbsp; | &nbsp;
    Parámetros recalculados en tiempo real
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div style="
        background:white;
        padding:12px 18px;
        border-radius:12px;
        border:1px solid #e2e8f0;
        margin-bottom:20px;
    ">

    Estado de la cadena:

    <span style="
        background:{color_estado};
        color:white;
        padding:6px 14px;
        border-radius:20px;
        font-weight:bold;
        margin-left:8px;
    ">
    {estado}
    </span>

    <span style="
        float:right;
        color:#64748b;
    ">
    Escenario:
    <b>{escenario}</b>
    </span>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

st.markdown(
    '<div class="section">Executive Overview</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4, c5 = st.columns(5)


with c1:

    st.markdown(
        f"""
        <div class="metric-box"
        style="border-top:4px solid {color_servicio_actual};">

        <div class="metric-title">
        Nivel de servicio
        </div>

        <div class="metric-value"
        style="color:{color_servicio_actual};">
        {fill_rate:.1%}
        </div>

        <div class="metric-delta">
        Objetivo: {nivel_servicio_obj}%
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c2:

    st.markdown(
        f"""
        <div class="metric-box"
        style="border-top:4px solid #176b9c;">

        <div class="metric-title">
        Demanda promedio
        </div>

        <div class="metric-value">
        {demanda_promedio:,.0f}
        </div>

        <div class="metric-delta">
        unidades / día
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c3:

    st.markdown(
        f"""
        <div class="metric-box"
        style="border-top:4px solid {color_cobertura_actual};">

        <div class="metric-title">
        Inventario
        </div>

        <div class="metric-value"
        style="color:{color_cobertura_actual};">
        {inventario_actual:,.0f}
        </div>

        <div class="metric-delta">
        {cobertura:.1f} días de cobertura
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c4:

    st.markdown(
        f"""
        <div class="metric-box"
        style="border-top:4px solid {color_utilizacion};">

        <div class="metric-title">
        Utilización logística
        </div>

        <div class="metric-value"
        style="color:{color_utilizacion};">
        {utilizacion:.0f}%
        </div>

        <div class="metric-delta">
        capacidad utilizada
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c5:

    st.markdown(
        f"""
        <div class="metric-box"
        style="border-top:4px solid {color_riesgo_actual};">

        <div class="metric-title">
        Riesgo de quiebre
        </div>

        <div class="metric-value"
        style="color:{color_riesgo_actual};">
        {riesgo_quiebre:.1f}%
        </div>

        <div class="metric-delta">
        proyección
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CADENA INTEGRAL
# ============================================================

st.markdown(
    '<div class="section">Cadena de suministro integral</div>',
    unsafe_allow_html=True
)

nodos = [
    "Proveedores",
    "Abastecimiento",
    "Producción",
    "CD",
    "Inventario",
    "Transporte",
    "Puntos de venta",
    "Consumidor"
]

valores = [
    95,
    92,
    min(100, capacidad_real),
    min(100, capacidad_real),
    min(100, cobertura * 8),
    min(100, utilizacion),
    fill_rate * 100,
    fill_rate * 100
]

colores = []

for valor in valores:

    if valor >= 90:
        colores.append("#16a34a")

    elif valor >= 70:
        colores.append("#f59e0b")

    else:
        colores.append("#dc2626")


fig_chain = go.Figure()


for i in range(
    len(nodos) - 1
):

    fig_chain.add_trace(
        go.Scatter(
            x=[i, i + 1],
            y=[0, 0],
            mode="lines",
            line=dict(
                color="#b8c7d9",
                width=9
            ),
            hoverinfo="skip",
            showlegend=False
        )
    )


fig_chain.add_trace(
    go.Scatter(
        x=list(
            range(len(nodos))
        ),
        y=[0] * len(nodos),
        mode="markers+text",
        marker=dict(
            size=40,
            color=colores,
            line=dict(
                color="white",
                width=4
            )
        ),
        text=nodos,
        textposition="bottom center",
        textfont=dict(
            size=11,
            color="#123b5d"
        ),
        customdata=np.array(
            valores
        ),
        hovertemplate=
        "<b>%{text}</b><br>"
        "Índice operacional: "
        "%{customdata:.1f}%"
        "<extra></extra>",
        showlegend=False
    )
)


fig_chain.update_layout(
    height=230,
    margin=dict(
        l=20,
        r=20,
        t=20,
        b=75
    ),
    xaxis=dict(
        visible=False,
        range=[
            -0.5,
            len(nodos) - 0.5
        ]
    ),
    yaxis=dict(
        visible=False,
        range=[
            -0.25,
            0.25
        ]
    ),
    plot_bgcolor="white",
    paper_bgcolor="white"
)


st.plotly_chart(
    fig_chain,
    use_container_width=True,
    config={
        "displayModeBar": False
    }
)


# ============================================================
# FORECAST
# ============================================================

st.markdown(
    '<div class="section">Predicción de demanda</div>',
    unsafe_allow_html=True
)

fig_forecast = go.Figure()


fig_forecast.add_trace(
    go.Scatter(
        x=df_fc["fecha"],
        y=df_fc["demanda"],
        name="Demanda",
        line=dict(
            color="#123b5d",
            width=2
        )
    )
)


fig_forecast.add_trace(
    go.Scatter(
        x=df_fc["fecha"],
        y=df_fc["forecast"],
        name="Forecast",
        line=dict(
            color="#16a3a5",
            width=3,
            dash="dash"
        )
    )
)


fig_forecast.update_layout(
    height=350,
    hovermode="x unified",
    margin=dict(
        l=20,
        r=20,
        t=20,
        b=20
    ),
    plot_bgcolor="white",
    paper_bgcolor="white"
)


st.plotly_chart(
    fig_forecast,
    use_container_width=True
)


# ============================================================
# INVENTARIO
# ============================================================

col1, col2 = st.columns(2)


with col1:

    st.markdown(
        '<div class="section">Inventario proyectado</div>',
        unsafe_allow_html=True
    )

    fig_inv = go.Figure()

    fig_inv.add_trace(
        go.Scatter(
            x=df["fecha"],
            y=df["inventario"],
            fill="tozeroy",
            name="Inventario",
            line=dict(
                color="#176b9c",
                width=3
            )
        )
    )

    fig_inv.add_hline(
        y=optim["reorder_point"],
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text="Punto de reorden"
    )

    fig_inv.update_layout(
        height=300,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        ),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(
        fig_inv,
        use_container_width=True
    )


with col2:

    st.markdown(
        '<div class="section">Presión de la cadena</div>',
        unsafe_allow_html=True
    )

    presion = pd.DataFrame({
        "Factor": [
            "Demanda",
            "Lead Time",
            "Transporte",
            "Inventario",
            "Servicio"
        ],
        "Presión": [
            min(
                100,
                50 +
                demanda_extra *
                100
            ),
            min(
                100,
                lead_time_real *
                7
            ),
            utilizacion,
            min(
                100,
                100 -
                cobertura * 5
            ),
            max(
                0,
                100 -
                fill_rate * 100
            )
        ]
    })

    fig_pressure = go.Figure(
        go.Bar(
            x=presion["Presión"],
            y=presion["Factor"],
            orientation="h",
            marker_color=[
                "#176b9c",
                "#f59e0b",
                "#7c3aed",
                "#dc2626",
                "#16a34a"
            ]
        )
    )

    fig_pressure.update_layout(
        height=300,
        xaxis=dict(
            range=[0, 100]
        ),
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        ),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(
        fig_pressure,
        use_container_width=True
    )


# ============================================================
# DECISIÓN PREDICTIVA
# ============================================================

st.markdown(
    '<div class="section">Lectura ejecutiva del modelo</div>',
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# ANÁLISIS REAL DE CADA INDICADOR
# ------------------------------------------------------------

problemas = []
señales = []

if fill_rate < nivel_servicio_obj / 100:

    problemas.append(
        f"el nivel de servicio está por debajo del objetivo "
        f"({fill_rate:.1%} vs {nivel_servicio_obj}%)"
    )

    señales.append("servicio")


if cobertura < 5:

    problemas.append(
        f"la cobertura es crítica ({cobertura:.1f} días)"
    )

    señales.append("inventario")

elif cobertura < 10:

    problemas.append(
        f"la cobertura es reducida ({cobertura:.1f} días)"
    )

    señales.append("inventario")


if utilizacion >= 100:

    problemas.append(
        "la capacidad logística está completamente utilizada"
    )

    señales.append("logística")

elif utilizacion >= 90:

    problemas.append(
        f"la utilización logística es elevada ({utilizacion:.0f}%)"
    )

    señales.append("logística")


if riesgo_quiebre >= 50:

    problemas.append(
        f"el riesgo proyectado es crítico ({riesgo_quiebre:.1f}%)"
    )

    señales.append("riesgo")

elif riesgo_quiebre >= 30:

    problemas.append(
        f"el riesgo proyectado es elevado ({riesgo_quiebre:.1f}%)"
    )

    señales.append("riesgo")


# ------------------------------------------------------------
# TENDENCIA DEL FORECAST
# ------------------------------------------------------------

forecast_promedio = (
    df_fc["forecast"]
    .tail(30)
    .mean()
)

variacion_forecast = (
    (
        forecast_promedio -
        demanda_promedio
    )
    /
    max(
        demanda_promedio,
        1
    )
) * 100


if variacion_forecast > 5:

    lectura_forecast = (
        "El modelo detecta una tendencia creciente "
        "de demanda."
    )

elif variacion_forecast < -5:

    lectura_forecast = (
        "El modelo detecta una tendencia decreciente "
        "de demanda."
    )

else:

    lectura_forecast = (
        "El modelo proyecta una demanda relativamente estable."
    )


# ------------------------------------------------------------
# MENSAJE EJECUTIVO DINÁMICO
# ------------------------------------------------------------

if len(problemas) >= 3:

    titulo_lectura = (
        "La simulación identifica múltiples presiones "
        "operacionales."
    )

    color_lectura = "#dc2626"

elif len(problemas) >= 1:

    titulo_lectura = (
        "La simulación identifica señales que requieren "
        "monitoreo."
    )

    color_lectura = "#f59e0b"

else:

    titulo_lectura = (
        "La simulación se mantiene dentro de los "
        "parámetros definidos."
    )

    color_lectura = "#16a34a"


detalle_problemas = ""

if problemas:

    detalle_problemas = (
        "<br><br><b>Principales señales:</b><br>"
        +
        "<br>".join(
            [
                f"• {p}"
                for p in problemas
            ]
        )
    )

else:

    detalle_problemas = (
        "<br><br>Los principales indicadores "
        "se encuentran dentro de los rangos definidos."
    )


st.markdown(
    f"""
    <div class="executive"
    style="border-left-color:{color_lectura};">

    <div style="
        color:{color_lectura};
        font-size:18px;
        font-weight:800;
    ">
    {titulo_lectura}
    </div>

    <br>

    <b>Demanda:</b>
    {lectura_forecast}

    {detalle_problemas}

    <br><br>

    <b>Lectura del modelo:</b>
    La combinación entre demanda, inventario,
    capacidad logística, lead time y nivel de servicio
    determina la presión operacional proyectada.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# EXPLICACIÓN AUTOMÁTICA
# ============================================================

st.markdown(
    '<div class="section">¿Qué está diciendo el sistema?</div>',
    unsafe_allow_html=True
)


# Demanda

if variacion_forecast > 5:

    demanda_texto = (
        f"La demanda presenta una señal creciente. "
        f"El forecast se encuentra aproximadamente "
        f"{variacion_forecast:.1f}% sobre la demanda promedio."
    )

elif variacion_forecast < -5:

    demanda_texto = (
        f"La demanda presenta una señal decreciente. "
        f"El forecast se encuentra aproximadamente "
        f"{abs(variacion_forecast):.1f}% bajo la demanda promedio."
    )

else:

    demanda_texto = (
        "La demanda proyectada mantiene una trayectoria "
        "relativamente estable."
    )


# Inventario

if cobertura < 5:

    inventario_texto = (
        f"El inventario tiene solo {cobertura:.1f} días "
        "de cobertura, señal de presión sobre disponibilidad."
    )

elif cobertura < 10:

    inventario_texto = (
        f"El inventario tiene {cobertura:.1f} días "
        "de cobertura y requiere seguimiento."
    )

else:

    inventario_texto = (
        f"El inventario mantiene {cobertura:.1f} días "
        "de cobertura."
    )


# Servicio

if fill_rate < nivel_servicio_obj / 100:

    servicio_texto = (
        f"El nivel de servicio ({fill_rate:.1%}) está "
        f"por debajo del objetivo ({nivel_servicio_obj}%)."
    )

else:

    servicio_texto = (
        f"El nivel de servicio ({fill_rate:.1%}) "
        "cumple el objetivo configurado."
    )


# Logística

if utilizacion >= 100:

    logistica_texto = (
        "La capacidad logística está en el límite "
        "operacional."
    )

elif utilizacion >= 90:

    logistica_texto = (
        f"La utilización logística es elevada ({utilizacion:.0f}%)."
    )

else:

    logistica_texto = (
        f"La utilización logística se encuentra "
        f"en {utilizacion:.0f}%."
    )


# Riesgo

if riesgo_quiebre >= 50:

    riesgo_texto = (
        f"El riesgo proyectado es crítico ({riesgo_quiebre:.1f}%). "
        "La disponibilidad futura requiere atención prioritaria."
    )

elif riesgo_quiebre >= 30:

    riesgo_texto = (
        f"El riesgo proyectado es elevado ({riesgo_quiebre:.1f}%). "
        "El sistema detecta presión sobre la disponibilidad."
    )

elif riesgo_quiebre >= 15:

    riesgo_texto = (
        f"El riesgo proyectado es moderado ({riesgo_quiebre:.1f}%)."
    )

else:

    riesgo_texto = (
        f"El riesgo proyectado se mantiene bajo "
        f"({riesgo_quiebre:.1f}%)."
    )


st.info(
    f"""
    **Demanda:** {demanda_texto}

    **Inventario:** {inventario_texto}

    **Servicio:** {servicio_texto}

    **Logística:** {logistica_texto}

    **Riesgo:** {riesgo_texto}

    **Acción analítica:** el modelo combina estas señales
    para anticipar necesidades de reposición, capacidad
    y nivel de servicio.
    """
)


# ============================================================
# OPTIMIZACIÓN
# ============================================================

st.markdown(
    '<div class="section">Decisión de inventario</div>',
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)


with c1:

    st.metric(
        "Punto de reorden",
        f"{optim['reorder_point']:,.0f}"
    )


with c2:

    st.metric(
        "Stock de seguridad",
        f"{optim['stock_seguridad']:,.0f}"
    )


with c3:

    st.metric(
        "EOQ",
        f"{optim['eoq']:,.0f}"
    )


# ============================================================
# INFORME PDF
# ============================================================

st.markdown(
    '<div class="section">Informe ejecutivo</div>',
    unsafe_allow_html=True
)

st.write(
    "Genera un informe del escenario actualmente seleccionado."
)


def generar_pdf():

    try:

        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle
        )
        from reportlab.lib.styles import (
            getSampleStyleSheet,
            ParagraphStyle
        )
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER

        buffer = io.BytesIO()

        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=35,
            leftMargin=35,
            topMargin=35,
            bottomMargin=35
        )

        styles = getSampleStyleSheet()

        titulo = ParagraphStyle(
            "TituloCCU",
            parent=styles["Title"],
            alignment=TA_CENTER,
            textColor=colors.HexColor(
                "#123b5d"
            ),
            fontSize=22
        )

        subtitulo = ParagraphStyle(
            "SubtituloCCU",
            parent=styles["Heading2"],
            alignment=TA_CENTER,
            textColor=colors.HexColor(
                "#176b9c"
            ),
            fontSize=13
        )

        contenido = []

        # ----------------------------------------------------
        # PORTADA
        # ----------------------------------------------------

        contenido.append(
            Paragraph(
                "CCU | PREDICTIVE SUPPLY CHAIN",
                titulo
            )
        )

        contenido.append(
            Paragraph(
                "Informe Ejecutivo de Cadena de Suministro",
                subtitulo
            )
        )

        contenido.append(
            Spacer(1, 12)
        )

        contenido.append(
            Paragraph(
                f"""
                <b>Fecha de generación:</b>
                {hora_actualizacion}<br/>

                <b>Escenario:</b>
                {escenario}<br/>

                <b>Estado del sistema:</b>
                {estado}<br/>

                <b>Motor:</b>
                Modelo predictivo + simulación operacional
                """,
                styles["BodyText"]
            )
        )

        contenido.append(
            Spacer(1, 18)
        )

        # ----------------------------------------------------
        # PROYECTO
        # ----------------------------------------------------

        contenido.append(
            Paragraph(
                "1. Descripción del proyecto",
                styles["Heading2"]
            )
        )

        contenido.append(
            Paragraph(
                """
                CCU Predictive Supply Chain es un sistema
                académico de soporte a decisiones orientado
                al monitoreo integral de la cadena de suministro.

                El proyecto integra simulación de demanda,
                forecasting, análisis de inventario, capacidad
                logística, nivel de servicio y optimización
                de reposición.

                <br/><br/>

                <b>Duración estimada del proyecto:</b>
                1 a 2 semanas de desarrollo iterativo.

                <br/>

                <b>Etapa:</b>
                Prototipo funcional / demostrativo.

                <br/>

                <b>Arquitectura:</b>
                Python + Streamlit + Pandas + NumPy +
                Scikit-learn + Plotly + ReportLab.
                """,
                styles["BodyText"]
            )
        )

        contenido.append(
            Spacer(1, 15)
        )

        # ----------------------------------------------------
        # KPI
        # ----------------------------------------------------

        contenido.append(
            Paragraph(
                "2. Indicadores ejecutivos",
                styles["Heading2"]
            )
        )

        tabla_kpi = Table([
            [
                "Indicador",
                "Resultado",
                "Referencia"
            ],
            [
                "Nivel de servicio",
                f"{fill_rate:.1%}",
                f"Objetivo {nivel_servicio_obj}%"
            ],
            [
                "Demanda promedio",
                f"{demanda_promedio:,.0f}",
                "unidades/día"
            ],
            [
                "Inventario",
                f"{inventario_actual:,.0f}",
                f"{cobertura:.1f} días"
            ],
            [
                "Utilización logística",
                f"{utilizacion:.0f}%",
                "capacidad"
            ],
            [
                "Riesgo de quiebre",
                f"{riesgo_quiebre:.1f}%",
                "proyección"
            ]
        ])

        tabla_kpi.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#123b5d"
                    )
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor(
                        "#cbd5e1"
                    )
                ),
                (
                    "BACKGROUND",
                    (0, 1),
                    (-1, -1),
                    colors.HexColor(
                        "#f8fafc"
                    )
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    8
                )
            ])
        )

        contenido.append(
            tabla_kpi
        )

        contenido.append(
            Spacer(1, 18)
        )

        # ----------------------------------------------------
        # PARAMETROS
        # ----------------------------------------------------

        contenido.append(
            Paragraph(
                "3. Parámetros de simulación",
                styles["Heading2"]
            )
        )

        contenido.append(
            Paragraph(
                f"""
                <b>Variación de demanda:</b>
                {demanda_factor}%<br/>

                <b>Lead Time:</b>
                {lead_time_real} días<br/>

                <b>Capacidad logística:</b>
                {capacidad_real:.0f}%<br/>

                <b>Nivel de servicio objetivo:</b>
                {nivel_servicio_obj}%<br/>

                <b>Escenario:</b>
                {escenario}
                """,
                styles["BodyText"]
            )
        )

        contenido.append(
            Spacer(1, 18)
        )

        # ----------------------------------------------------
        # OPTIMIZACIÓN
        # ----------------------------------------------------

        contenido.append(
            Paragraph(
                "4. Decisión de inventario",
                styles["Heading2"]
            )
        )

        tabla = Table([
            [
                "Indicador",
                "Resultado"
            ],
            [
                "Punto de reorden",
                f"{optim['reorder_point']:,.0f}"
            ],
            [
                "Stock de seguridad",
                f"{optim['stock_seguridad']:,.0f}"
            ],
            [
                "EOQ",
                f"{optim['eoq']:,.0f}"
            ]
        ])

        tabla.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#176b9c"
                    )
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    8
                )
            ])
        )

        contenido.append(
            tabla
        )

        contenido.append(
            Spacer(1, 18)
        )

        # ----------------------------------------------------
        # LECTURA EJECUTIVA
        # ----------------------------------------------------

        contenido.append(
            Paragraph(
                "5. Lectura ejecutiva",
                styles["Heading2"]
            )
        )

        contenido.append(
            Paragraph(
                f"""
                <b>Demanda:</b>
                {demanda_texto}<br/><br/>

                <b>Inventario:</b>
                {inventario_texto}<br/><br/>

                <b>Servicio:</b>
                {servicio_texto}<br/><br/>

                <b>Logística:</b>
                {logistica_texto}<br/><br/>

                <b>Riesgo:</b>
                {riesgo_texto}
                """,
                styles["BodyText"]
            )
        )

        contenido.append(
            Spacer(1, 18)
        )

        # ----------------------------------------------------
        # CONCLUSIÓN
        # ----------------------------------------------------

        contenido.append(
            Paragraph(
                "6. Conclusión del modelo",
                styles["Heading2"]
            )
        )

        contenido.append(
            Paragraph(
                f"""
                El escenario analizado presenta un estado
                <b>{estado}</b>.

                El sistema utiliza la interacción entre
                demanda, inventario, capacidad logística,
                lead time y nivel de servicio para generar
                señales anticipadas sobre la cadena.

                <br/><br/>

                Este sistema corresponde a un modelo académico
                demostrativo. Los datos simulados y las
                estimaciones deben ser reemplazados o
                contrastados con información operacional real
                antes de utilizarse para decisiones empresariales.
                """,
                styles["BodyText"]
            )
        )

        contenido.append(
            Spacer(1, 25)
        )

        contenido.append(
            Paragraph(
                "CCU | Predictive Supply Chain Control Tower",
                titulo
            )
        )

        contenido.append(
            Paragraph(
                "Informe generado automáticamente por el sistema.",
                subtitulo
            )
        )

        doc.build(
            contenido
        )

        buffer.seek(0)

        return buffer.getvalue()

    except Exception as e:

        st.error(
            f"Error generando PDF: {e}"
        )

        return None


if st.button(
    "📄 Generar informe ejecutivo CCU",
    type="primary"
):

    pdf = generar_pdf()

    if pdf:

        st.download_button(
            "⬇️ Descargar PDF",
            data=pdf,
            file_name=(
                "CCU_Predictive_Supply_Chain.pdf"
            ),
            mime="application/pdf"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "CCU Predictive Supply Chain | "
    "Modelo académico demostrativo | "
    "Los datos utilizados pueden ser simulados "
    "y deben validarse con información operacional "
    "real antes de utilizarse para decisiones empresariales."
)
