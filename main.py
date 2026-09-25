import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="CCU | Cadena de Suministro Predictiva",
    page_icon="🍺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# ESTILO
# ============================================================

st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }

    .header {
        background: linear-gradient(90deg, #123b5d, #176b9c);
        padding: 25px;
        border-radius: 14px;
        color: white;
        margin-bottom: 25px;
    }

    .header h1 {
        margin: 0;
        font-size: 32px;
    }

    .header p {
        margin: 5px 0 0 0;
        font-size: 16px;
        color: #dbeafe;
    }

    .kpi {
        background: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        text-align: center;
    }

    .kpi-title {
        color: #64748b;
        font-size: 14px;
        font-weight: 600;
    }

    .kpi-value {
        color: #123b5d;
        font-size: 27px;
        font-weight: 800;
        margin-top: 5px;
    }

    .section {
        color: #123b5d;
        font-size: 22px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .executive {
        background: #eff6ff;
        border-left: 5px solid #2563eb;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 15px;
    }

    .success {
        background: #f0fdf4;
        border-left: 5px solid #22c55e;
        padding: 16px;
        border-radius: 8px;
    }

    .warning {
        background: #fff7ed;
        border-left: 5px solid #f97316;
        padding: 16px;
        border-radius: 8px;
    }

    .danger {
        background: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 16px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATOS SIMULADOS CCU
# ============================================================

@st.cache_data
def generar_datos(dias=180):

    np.random.seed(42)

    fechas = pd.date_range(
        start="2026-03-01",
        periods=dias,
        freq="D"
    )

    categorias = [
        "Cervezas",
        "Bebidas gaseosas",
        "Aguas",
        "Bebidas energéticas",
        "Néctares"
    ]

    categoria = np.random.choice(
        categorias,
        size=dias,
        p=[0.30, 0.30, 0.18, 0.12, 0.10]
    )

    tendencia = np.linspace(0, 18, dias)

    estacionalidad = (
        15 * np.sin(np.arange(dias) * 2 * np.pi / 30)
    )

    demanda = (
        520
        + tendencia
        + estacionalidad
        + np.random.normal(0, 35, dias)
    )

    demanda = np.maximum(demanda, 250).round().astype(int)

    ventas = demanda - np.random.normal(
        15, 12, dias
    )

    ventas = np.maximum(
        ventas,
        demanda * 0.82
    ).round().astype(int)

    inventario = (
        1700
        + np.random.normal(0, 180, dias)
        - np.arange(dias) * 1.5
    )

    inventario = np.maximum(
        inventario,
        500
    ).round().astype(int)

    costo_unitario = np.random.uniform(
        450,
        1200,
        dias
    )

    lead_time = np.random.randint(
        2,
        7,
        dias
    )

    df = pd.DataFrame({
        "fecha": fechas,
        "categoria": categoria,
        "demanda": demanda,
        "ventas": ventas,
        "inventario": inventario,
        "costo_unitario": costo_unitario,
        "lead_time": lead_time
    })

    df["fill_rate"] = (
        df["ventas"] / df["demanda"]
    ).clip(0, 1)

    return df


# ============================================================
# FORECAST
# ============================================================

def calcular_forecast(df):

    data = df.copy()

    data["t"] = np.arange(len(data))

    modelo = LinearRegression()

    modelo.fit(
        data[["t"]],
        data["demanda"]
    )

    data["forecast"] = modelo.predict(
        data[["t"]]
    )

    data["forecast"] = data["forecast"].clip(
        lower=data["demanda"].min() * 0.7
    )

    return data


# ============================================================
# DATOS
# ============================================================

df = generar_datos(180)

forecast_df = calcular_forecast(df)


# ============================================================
# KPIs
# ============================================================

fill_rate = df["fill_rate"].mean()

error_forecast = np.mean(
    np.abs(
        df["demanda"] -
        forecast_df["forecast"]
    )
)

inventario_promedio = df["inventario"].mean()

demanda_promedio = df["demanda"].mean()

inventario_valorizado = (
    df["inventario"] *
    df["costo_unitario"]
).mean()


# ============================================================
# MODELO DE INVENTARIO
# ============================================================

demanda_diaria = demanda_promedio

lead_time_promedio = df["lead_time"].mean()

desviacion_demanda = df["demanda"].std()

stock_seguridad = (
    1.65 *
    desviacion_demanda *
    np.sqrt(lead_time_promedio)
)

punto_reorden = (
    demanda_diaria *
    lead_time_promedio
    + stock_seguridad
)

inventario_actual = df["inventario"].iloc[-1]

pedido_sugerido = max(
    0,
    punto_reorden - inventario_actual
)


# ============================================================
# NIVEL DE RIESGO
# ============================================================

cobertura_dias = (
    inventario_actual /
    demanda_diaria
)

if cobertura_dias < lead_time_promedio:
    riesgo = "ALTO"
    color_riesgo = "danger"
elif cobertura_dias < lead_time_promedio + 3:
    riesgo = "MEDIO"
    color_riesgo = "warning"
else:
    riesgo = "BAJO"
    color_riesgo = "success"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🍺 CCU")

st.sidebar.markdown(
    "**Cadena de Suministro Predictiva**"
)

st.sidebar.markdown("---")

st.sidebar.subheader(
    "Parámetros de análisis"
)

horizonte = st.sidebar.slider(
    "Horizonte de análisis",
    30,
    180,
    180
)

categoria_filtro = st.sidebar.multiselect(
    "Categorías",
    sorted(df["categoria"].unique()),
    default=sorted(df["categoria"].unique())
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Modelo demostrativo con datos simulados. "
    "La plataforma representa una propuesta de "
    "analítica para soporte a decisiones logísticas."
)


# ============================================================
# FILTRO
# ============================================================

df_view = df[
    df["categoria"].isin(categoria_filtro)
].tail(horizonte)

forecast_view = forecast_df[
    forecast_df["categoria"].isin(categoria_filtro)
].tail(horizonte)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="header">

<h1>🍺 CCU — Cadena de Suministro Predictiva</h1>

<p>
Centro ejecutivo de planificación, pronóstico de demanda,
gestión de inventario y soporte a decisiones logísticas.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard Ejecutivo",
    "📈 Demanda Predictiva",
    "📦 Inventario",
    "🎯 Decisión Logística"
])


# ============================================================
# DASHBOARD
# ============================================================

with tab1:

    st.markdown(
        '<div class="section">Indicadores Ejecutivos</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.markdown(f"""
    <div class="kpi">
        <div class="kpi-title">Nivel de Servicio</div>
        <div class="kpi-value">{fill_rate:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

    c2.markdown(f"""
    <div class="kpi">
        <div class="kpi-title">Error Forecast</div>
        <div class="kpi-value">{error_forecast:.0f}</div>
    </div>
    """, unsafe_allow_html=True)

    c3.markdown(f"""
    <div class="kpi">
        <div class="kpi-title">Inventario Promedio</div>
        <div class="kpi-value">{inventario_promedio:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

    c4.markdown(f"""
    <div class="kpi">
        <div class="kpi-title">Riesgo Logístico</div>
        <div class="kpi-value">{riesgo}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section">Evolución de la Cadena</div>',
        unsafe_allow_html=True
    )

    grafico = forecast_view.set_index(
        "fecha"
    )[[
        "demanda",
        "forecast"
    ]]

    st.line_chart(grafico)

    st.markdown(
        '<div class="executive">'
        '<b>Lectura ejecutiva:</b> '
        'La plataforma utiliza el comportamiento histórico de la demanda '
        'para generar una referencia predictiva que permite anticipar '
        'necesidades de inventario y apoyar la planificación logística.'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# FORECAST
# ============================================================

with tab2:

    st.markdown(
        '<div class="section">Pronóstico de Demanda</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        El modelo utiliza una regresión de tendencia sobre la demanda
        histórica para generar una referencia predictiva.
        """,
    )

    st.line_chart(
        forecast_view.set_index("fecha")[[
            "demanda",
            "forecast"
        ]]
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Demanda promedio",
            f"{df_view['demanda'].mean():,.0f}"
        )

    with col2:
        st.metric(
            "Forecast actual",
            f"{forecast_df['forecast'].iloc[-1]:,.0f}"
        )

    with col3:
        variacion = (
            forecast_df["forecast"].iloc[-1]
            /
            forecast_df["forecast"].iloc[0]
            - 1
        )

        st.metric(
            "Tendencia",
            f"{variacion:+.1%}"
        )


# ============================================================
# INVENTARIO
# ============================================================

with tab3:

    st.markdown(
        '<div class="section">Gestión Predictiva de Inventario</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Inventario actual",
        f"{inventario_actual:,.0f}"
    )

    c2.metric(
        "Punto de reorden",
        f"{punto_reorden:,.0f}"
    )

    c3.metric(
        "Stock de seguridad",
        f"{stock_seguridad:,.0f}"
    )

    st.markdown("### Inventario vs. Demanda")

    inventario_chart = df_view.set_index(
        "fecha"
    )[[
        "inventario",
        "demanda"
    ]]

    st.line_chart(inventario_chart)

    st.markdown("### Distribución por categoría")

    resumen = (
        df_view
        .groupby("categoria")
        .agg(
            Demanda=("demanda", "sum"),
            Inventario=("inventario", "mean"),
            Nivel_Servicio=("fill_rate", "mean")
        )
        .reset_index()
    )

    resumen["Nivel_Servicio"] = (
        resumen["Nivel_Servicio"] * 100
    ).round(1)

    st.dataframe(
        resumen,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DECISIÓN LOGÍSTICA
# ============================================================

with tab4:

    st.markdown(
        '<div class="section">Motor de Decisión Logística</div>',
        unsafe_allow_html=True
    )

    if riesgo == "ALTO":

        st.markdown(
            f"""
            <div class="danger">
            <h3>⚠️ Riesgo alto de quiebre</h3>
            La cobertura actual es de
            <b>{cobertura_dias:.1f} días</b>,
            inferior al requerimiento estimado de abastecimiento.
            </div>
            """,
            unsafe_allow_html=True
        )

    elif riesgo == "MEDIO":

        st.markdown(
            f"""
            <div class="warning">
            <h3>🟠 Riesgo moderado</h3>
            Se recomienda monitorear la evolución de la demanda
            y preparar una reposición preventiva.
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="success">
            <h3>🟢 Situación controlada</h3>
            La cobertura actual permite absorber la demanda
            estimada bajo las condiciones del modelo.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Recomendación de reposición")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Demanda diaria",
        f"{demanda_diaria:,.0f}"
    )

    c2.metric(
        "Lead Time",
        f"{lead_time_promedio:.1f} días"
    )

    c3.metric(
        "Cobertura",
        f"{cobertura_dias:.1f} días"
    )

    c4.metric(
        "Pedido sugerido",
        f"{pedido_sugerido:,.0f}"
    )

    st.markdown("### Flujo de la cadena predictiva")

    st.markdown("""
    **Datos históricos**
    ↓
    **Pronóstico de demanda**
    ↓
    **Proyección de inventario**
    ↓
    **Cálculo de riesgo**
    ↓
    **Punto de reorden**
    ↓
    **Decisión de reposición**
    """)

    st.markdown("### Indicadores económicos")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Inventario valorizado promedio",
            f"${inventario_valorizado:,.0f}"
        )

    with col2:
        ahorro_potencial = (
            inventario_valorizado * 0.10
        )

        st.metric(
            "Capital potencialmente optimizable*",
            f"${ahorro_potencial:,.0f}"
        )

    st.caption(
        "*Estimación referencial sobre el modelo simulado; "
        "no corresponde a información financiera real de CCU."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    **CCU — Cadena de Suministro Predictiva**

    Plataforma demostrativa de analítica logística para
    pronóstico de demanda, inventario y soporte a decisiones.

    *Datos utilizados: simulación académica.*
    """
)
