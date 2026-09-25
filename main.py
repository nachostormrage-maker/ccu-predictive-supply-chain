import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

from modules.forecast import generar_forecast
from modules.inventory import calcular_kpis
from modules.inventory_optimizer import optimizar_inventario
from modules.pdf_report import generar_pdf_bytes


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
    background: #f5f8fb;
}

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #123b5d;
    margin-bottom: 0;
}

.subtitle {
    color: #64748b;
    font-size: 15px;
    margin-bottom: 20px;
}

.section {
    font-size: 22px;
    font-weight: 750;
    color: #123b5d;
    margin-top: 25px;
    margin-bottom: 12px;
}

.metric-box {
    background: white;
    border-radius: 14px;
    padding: 18px;
    border: 1px solid #e2e8f0;
    min-height: 115px;
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
    font-size: 13px;
    margin-top: 5px;
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

.status {
    padding: 8px 15px;
    border-radius: 20px;
    font-weight: 700;
    display: inline-block;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# DATOS BASE
# ============================================================

np.random.seed(42)

dias = 180

fechas = pd.date_range(
    start="2026-01-01",
    periods=dias,
    freq="D"
)

demanda_base = np.clip(
    np.random.normal(150, 28, dias),
    70,
    None
)

ventas_base = np.clip(
    demanda_base * np.random.normal(0.96, 0.035, dias),
    50,
    None
)

inventario_base = np.clip(
    1900 + np.cumsum(
        np.random.normal(0, 25, dias)
    ),
    800,
    3000
)

df_base = pd.DataFrame({
    "fecha": fechas,
    "demanda": demanda_base,
    "ventas": ventas_base,
    "inventario": inventario_base,
    "sku": np.random.choice(
        ["Cerveza", "Bebida", "Agua", "Néctar"],
        dias
    ),
    "lead_time": np.random.randint(2, 8, dias),
    "costo_unitario": np.random.randint(800, 1500, dias)
})


# ============================================================
# SIDEBAR — SIMULADOR
# ============================================================

st.sidebar.markdown("## 🍺 CCU")

st.sidebar.caption(
    "Predictive Supply Chain Control Tower"
)

st.sidebar.markdown("---")

st.sidebar.markdown("### Simulación")

demanda_factor = st.sidebar.slider(
    "Variación de demanda",
    -30,
    50,
    0,
    5
)

lead_time_factor = st.sidebar.slider(
    "Lead Time",
    1,
    15,
    5
)

capacidad = st.sidebar.slider(
    "Capacidad logística",
    50,
    120,
    100,
    5
)

nivel_servicio_obj = st.sidebar.slider(
    "Nivel de servicio objetivo",
    85,
    99,
    95
)

st.sidebar.markdown("---")

escenario = st.sidebar.selectbox(
    "Escenario",
    [
        "Base",
        "Aumento de demanda",
        "Restricción logística",
        "Alta demanda + restricción"
    ]
)

# ============================================================
# ESCENARIOS
# ============================================================

demanda_extra = demanda_factor / 100

if escenario == "Aumento de demanda":
    demanda_extra += 0.20

elif escenario == "Restricción logística":
    capacidad *= 0.70
    lead_time_factor += 3

elif escenario == "Alta demanda + restricción":
    demanda_extra += 0.25
    capacidad *= 0.70
    lead_time_factor += 3


# ============================================================
# SIMULACIÓN
# ============================================================

df = df_base.copy()

df["demanda"] = df["demanda"] * (1 + demanda_extra)

df["ventas"] = np.minimum(
    df["ventas"] * (1 + demanda_extra * 0.5),
    df["demanda"]
)

# Impacto del lead time
factor_lead = lead_time_factor / 5

df["inventario"] = (
    df_base["inventario"]
    - np.maximum(
        0,
        df["demanda"] - df["ventas"]
    ).cumsum() * 0.45 * factor_lead
)

df["inventario"] = df["inventario"].clip(
    lower=100
)


# ============================================================
# KPIs
# ============================================================

kpis = calcular_kpis(df)

fill_rate = kpis["fill_rate"]

inventario = df["inventario"].iloc[-1]

demanda_promedio = df["demanda"].mean()

cobertura = (
    inventario / demanda_promedio
)

utilizacion = min(
    100,
    (demanda_promedio / (demanda_promedio * capacidad / 100)) * 100
)

riesgo_quiebre = max(
    0,
    min(
        100,
        100 - cobertura * 7
    )
)


# ============================================================
# ESTADO GENERAL
# ============================================================

if riesgo_quiebre < 15 and fill_rate >= nivel_servicio_obj / 100:
    estado = "ESTABLE"
    color_estado = "#16a34a"

elif riesgo_quiebre < 30:
    estado = "MONITOREAR"
    color_estado = "#f59e0b"

else:
    estado = "RIESGO"
    color_estado = "#dc2626"


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">CCU | Predictive Supply Chain Control Tower</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Visión integral y predictiva de la cadena de suministro'
    '</div>',
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
    ">{estado}</span>

    <span style="float:right;color:#64748b;">
    Escenario: <b>{escenario}</b>
    </span>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# KPIs GERENCIALES
# ============================================================

st.markdown(
    '<div class="section">Executive Overview</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(
        f"""
        <div class="metric-box">
        <div class="metric-title">Nivel de servicio</div>
        <div class="metric-value">{fill_rate:.1%}</div>
        <div class="metric-delta">Objetivo: {nivel_servicio_obj}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f"""
        <div class="metric-box">
        <div class="metric-title">Demanda promedio</div>
        <div class="metric-value">{demanda_promedio:,.0f}</div>
        <div class="metric-delta">unidades / día</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        f"""
        <div class="metric-box">
        <div class="metric-title">Inventario</div>
        <div class="metric-value">{inventario:,.0f}</div>
        <div class="metric-delta">{cobertura:.1f} días cobertura</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c4:
    st.markdown(
        f"""
        <div class="metric-box">
        <div class="metric-title">Utilización logística</div>
        <div class="metric-value">{utilizacion:.0f}%</div>
        <div class="metric-delta">capacidad utilizada</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c5:
    st.markdown(
        f"""
        <div class="metric-box">
        <div class="metric-title">Riesgo de quiebre</div>
        <div class="metric-value">{riesgo_quiebre:.1f}%</div>
        <div class="metric-delta">proyección</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CADENA DE SUMINISTRO
# ============================================================

st.markdown(
    '<div class="section">Supply Chain Flow</div>',
    unsafe_allow_html=True
)

nodos = [
    "PROVEEDORES",
    "ABASTECIMIENTO",
    "PRODUCCIÓN",
    "CENTRO DISTRIBUCIÓN",
    "INVENTARIO",
    "TRANSPORTE",
    "PUNTOS DE VENTA",
    "CONSUMIDOR"
]

valores = [
    95,
    92,
    min(100, capacidad),
    min(100, capacidad),
    min(100, cobertura * 8),
    min(100, utilizacion),
    fill_rate * 100,
    fill_rate * 100
]

colores = []

for v in valores:
    if v >= 90:
        colores.append("#16a34a")
    elif v >= 70:
        colores.append("#f59e0b")
    else:
        colores.append("#dc2626")


fig = go.Figure()

# líneas
for i in range(len(nodos) - 1):

    fig.add_trace(
        go.Scatter(
            x=[i, i + 1],
            y=[0, 0],
            mode="lines",
            line=dict(
                color="#b8c7d9",
                width=8
            ),
            hoverinfo="skip",
            showlegend=False
        )
    )

# nodos
fig.add_trace(
    go.Scatter(
        x=list(range(len(nodos))),
        y=[0] * len(nodos),
        mode="markers+text",
        marker=dict(
            size=38,
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
        customdata=np.array(valores),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Estado: %{customdata:.1f}%<extra></extra>"
        ),
        showlegend=False
    )
)

fig.update_layout(
    height=230,
    margin=dict(
        l=20,
        r=20,
        t=20,
        b=70
    ),
    xaxis=dict(
        visible=False,
        range=[-0.5, len(nodos) - 0.5]
    ),
    yaxis=dict(
        visible=False,
        range=[-0.25, 0.25]
    ),
    plot_bgcolor="white",
    paper_bgcolor="white"
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displayModeBar": False
    }
)


# ============================================================
# PREDICCIÓN
# ============================================================

st.markdown(
    '<div class="section">Predictive Demand</div>',
    unsafe_allow_html=True
)

df_fc = generar_forecast(df)

if isinstance(df_fc, tuple):
    df_fc = df_fc[0]

if "forecast" not in df_fc.columns:
    df_fc["forecast"] = (
        df_fc["demanda"]
        .rolling(7, min_periods=1)
        .mean()
    )

fig_forecast = go.Figure()

fig_forecast.add_trace(
    go.Scatter(
        x=df_fc["fecha"],
        y=df_fc["demanda"],
        name="Demanda real",
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
    margin=dict(
        l=20,
        r=20,
        t=20,
        b=20
    ),
    hovermode="x unified",
    plot_bgcolor="white",
    paper_bgcolor="white"
)

st.plotly_chart(
    fig_forecast,
    use_container_width=True
)


# ============================================================
# INVENTARIO + RIESGO
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

    fig_inv.update_layout(
        height=300,
        margin=dict(
            l=20,
            r=20,
            t=10,
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
            min(100, 50 + demanda_extra * 100),
            min(100, lead_time_factor * 8),
            utilizacion,
            min(100, 100 - cobertura * 5),
            100 - fill_rate * 100
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
            t=10,
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
    '<div class="section">Lectura Ejecutiva</div>',
    unsafe_allow_html=True
)

if riesgo_quiebre >= 30:

    mensaje = """
    <b>La cadena presenta presión operacional.</b><br><br>
    El escenario actual proyecta una disminución de la cobertura
    y una mayor probabilidad de quiebre. La principal presión se
    concentra en la relación entre demanda, inventario y capacidad logística.
    """

elif cobertura < 10:

    mensaje = """
    <b>La cadena requiere monitoreo.</b><br><br>
    La cobertura proyectada se encuentra en un nivel reducido.
    Se recomienda revisar anticipadamente reposición, capacidad
    de transporte y planificación de demanda.
    """

else:

    mensaje = """
    <b>La cadena opera dentro de parámetros controlados.</b><br><br>
    El modelo mantiene una cobertura suficiente y el nivel de servicio
    se encuentra próximo al objetivo definido. La simulación no identifica
    actualmente una presión crítica.
    """

st.markdown(
    f"""
    <div class="executive">
    {mensaje}
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# RECOMENDACIÓN
# ============================================================

st.markdown(
    '<div class="section">Decisión sugerida por el modelo</div>',
    unsafe_allow_html=True
)

optim = optimizar_inventario(df)

c1, c2, c3 = st.columns(3)

c1.metric(
    "Punto de reorden",
    f"{optim.get('reorder_point', 0):,.0f}"
)

c2.metric(
    "Stock de seguridad",
    f"{optim.get('stock_seguridad', 0):,.0f}"
)

c3.metric(
    "EOQ",
    f"{optim.get('eoq', 0):,.0f}"
)


# ============================================================
# PDF
# ============================================================

st.markdown(
    '<div class="section">Informe ejecutivo</div>',
    unsafe_allow_html=True
)

st.write(
    "Genera un informe con los resultados del escenario actualmente seleccionado."
)

if st.button(
    "📄 Generar informe ejecutivo CCU",
    type="primary"
):

    try:

        pdf = generar_pdf_bytes(
            df,
            kpis
        )

        st.download_button(
            label="⬇️ Descargar PDF",
            data=pdf,
            file_name="CCU_Predictive_Supply_Chain.pdf",
            mime="application/pdf"
        )

        st.success(
            "Informe generado correctamente."
        )

    except Exception as e:

        st.error(
            f"No fue posible generar el PDF: {e}"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "CCU Predictive Supply Chain | Modelo académico demostrativo | "
    "Los datos utilizados pueden ser simulados y deben validarse "
    "con información operacional real antes de utilizarse para decisiones empresariales."
)
