import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="CCU | Predictive Supply Chain",
    page_icon="🍺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# ESTILO
# ============================================================

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: Arial, sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 50% 20%, #123b5d 0%, #071522 42%, #02070c 100%);
    color: white;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
    max-width: 1500px;
}

.hero {
    text-align:center;
    padding: 10px 0 5px 0;
}

.hero h1 {
    font-size: 42px;
    letter-spacing: 5px;
    margin-bottom: 0;
    color: #ffffff;
}

.hero p {
    color: #70c9ff;
    font-size: 14px;
    letter-spacing: 3px;
}

.metric {
    background: rgba(7, 22, 35, .72);
    border: 1px solid rgba(80,190,255,.25);
    border-radius: 12px;
    padding: 12px;
    text-align: center;
    backdrop-filter: blur(8px);
}

.metric-label {
    color: #7fa5bd;
    font-size: 11px;
    letter-spacing: 1px;
}

.metric-value {
    font-size: 25px;
    font-weight: bold;
    color: white;
}

.status {
    text-align:center;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0 12px 0;
    font-weight: bold;
    letter-spacing: 1px;
}

.normal {
    background: rgba(0,190,130,.15);
    border: 1px solid #00c98b;
    color: #42f5b5;
}

.warning {
    background: rgba(255,170,0,.15);
    border: 1px solid #ffaa00;
    color: #ffc44d;
}

.danger {
    background: rgba(255,60,60,.15);
    border: 1px solid #ff4040;
    color: #ff7373;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# TÍTULO
# ============================================================

st.markdown("""
<div class="hero">
    <h1>CCU</h1>
    <p>PREDICTIVE SUPPLY CHAIN · DIGITAL CONTROL TOWER</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# CONTROLES
# ============================================================

with st.sidebar:

    st.markdown("## SIMULATION CONTROL")

    demanda = st.slider(
        "Demanda diaria",
        min_value=80,
        max_value=300,
        value=150,
        step=5
    )

    lead_time = st.slider(
        "Lead Time",
        min_value=1,
        max_value=15,
        value=5,
        step=1
    )

    capacidad = st.slider(
        "Capacidad logística",
        min_value=50,
        max_value=120,
        value=100,
        step=5
    )

    service_level = st.slider(
        "Nivel de servicio objetivo",
        min_value=85,
        max_value=99,
        value=95,
        step=1
    )

    st.markdown("---")

    escenario = st.selectbox(
        "Escenario",
        [
            "Operación normal",
            "Alta demanda",
            "Restricción logística"
        ]
    )

# ============================================================
# ESCENARIOS
# ============================================================

demanda_real = demanda
lead_real = lead_time
capacidad_real = capacidad

if escenario == "Alta demanda":
    demanda_real = int(demanda * 1.25)

elif escenario == "Restricción logística":
    capacidad_real = max(40, int(capacidad * 0.65))
    lead_real = lead_time + 3

# ============================================================
# MOTOR PREDICTIVO
# ============================================================

np.random.seed(42)

dias = np.arange(30)

historico = (
    demanda_real
    + np.sin(dias / 3) * demanda_real * 0.08
    + np.random.normal(0, demanda_real * 0.04, len(dias))
)

forecast = (
    demanda_real
    + np.sin(np.arange(30, 60) / 3) * demanda_real * 0.08
)

forecast_prom = float(np.mean(forecast))

# ============================================================
# INVENTARIO
# ============================================================

stock_seguridad = (
    demanda_real
    * (lead_real ** 0.5)
    * ((100 - service_level) / 20 + 0.8)
)

rop = (
    demanda_real * lead_real
    + stock_seguridad
)

inventario_inicial = demanda_real * 9

consumo_proyectado = demanda_real * 7

inventario_final = (
    inventario_inicial
    - consumo_proyectado
)

# ============================================================
# REPOSICIÓN
# ============================================================

necesidad_reposicion = max(
    0,
    rop - inventario_final
)

eoq = np.sqrt(
    (2 * demanda_real * 300 * 45000)
    / 3300
)

pedido_sugerido = max(
    necesidad_reposicion,
    eoq * 0.15
)

# Capacidad logística afecta el pedido efectivo
flujo_logistico = pedido_sugerido * (capacidad_real / 100)

# ============================================================
# RIESGO
# ============================================================

ratio_inventario = inventario_final / max(rop, 1)

if ratio_inventario < 0.65 or capacidad_real < 60:
    riesgo = "CRÍTICO"
    color = "danger"

elif ratio_inventario < 1:
    riesgo = "MONITOREAR"
    color = "warning"

else:
    riesgo = "ESTABLE"
    color = "normal"

# ============================================================
# KPIs
# ============================================================

c1, c2, c3, c4, c5 = st.columns(5)

def metric(col, label, value):
    col.markdown(
        f"""
        <div class="metric">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

metric(c1, "DEMANDA", f"{demanda_real:,.0f}/día")
metric(c2, "FORECAST", f"{forecast_prom:,.0f}")
metric(c3, "INVENTARIO", f"{inventario_final:,.0f}")
metric(c4, "ROP", f"{rop:,.0f}")
metric(c5, "REPOSICIÓN", f"{pedido_sugerido:,.0f}")

st.markdown(
    f"""
    <div class="status {color}">
        ESTADO DE LA CADENA · {riesgo}
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# CADENA 3D
# ============================================================

st.markdown("### LIVE SUPPLY CHAIN")

# Coordenadas principales
nodes = {
    "Proveedor": (-8, 1.5, 0),
    "Planta CCU": (-4, 0, 1),
    "CD Norte": (0, 3, 0),
    "CD Centro": (0, 0, 0),
    "CD Sur": (0, -3, 0),
    "Puntos de venta": (5, 0, 1),
    "Cliente": (8, 0, 0)
}

# ============================================================
# FIGURA 3D
# ============================================================

fig = go.Figure()

# ------------------------------------------------------------
# CONEXIONES
# ------------------------------------------------------------

connections = [
    ("Proveedor", "Planta CCU"),
    ("Planta CCU", "CD Norte"),
    ("Planta CCU", "CD Centro"),
    ("Planta CCU", "CD Sur"),
    ("CD Norte", "Puntos de venta"),
    ("CD Centro", "Puntos de venta"),
    ("CD Sur", "Puntos de venta"),
    ("Puntos de venta", "Cliente")
]

for a, b in connections:

    x = [nodes[a][0], nodes[b][0]]
    y = [nodes[a][1], nodes[b][1]]
    z = [nodes[a][2], nodes[b][2]]

    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            line=dict(
                color="#248cc7",
                width=5
            ),
            hoverinfo="none",
            showlegend=False
        )
    )

# ============================================================
# NODOS
# ============================================================

node_names = list(nodes.keys())

node_x = [nodes[n][0] for n in node_names]
node_y = [nodes[n][1] for n in node_names]
node_z = [nodes[n][2] for n in node_names]

# Color según estado
node_colors = []

for n in node_names:

    if "CD" in n:

        if riesgo == "CRÍTICO":
            node_colors.append("#ff3030")

        elif riesgo == "MONITOREAR":
            node_colors.append("#ffaa00")

        else:
            node_colors.append("#00d9a0")

    elif n == "Planta CCU":
        node_colors.append("#4db8ff")

    else:
        node_colors.append("#7a8cff")

fig.add_trace(
    go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode="markers+text",
        text=node_names,
        textposition="top center",
        marker=dict(
            size=[
                13 if n == "Planta CCU" else 9
                for n in node_names
            ],
            color=node_colors,
            opacity=.95,
            line=dict(
                color="white",
                width=1
            )
        ),
        hovertemplate="%{text}<extra></extra>",
        showlegend=False
    )
)

# ============================================================
# PARTÍCULAS / FLUJO
# ============================================================

particle_count = int(
    max(20, min(100, flujo_logistico / 8))
)

particle_x = []
particle_y = []
particle_z = []

# Distribución dinámica de partículas
for i in range(particle_count):

    t = (i / particle_count)

    # flujo planta -> distribución -> cliente
    if t < 0.45:

        local = t / 0.45

        x = -4 + local * 4

        branch = i % 3

        if branch == 0:
            y = 0 + local * 3
        elif branch == 1:
            y = 0
        else:
            y = 0 - local * 3

        z = 1 - local

    else:

        local = (t - 0.45) / 0.55

        x = 0 + local * 8
        y = 0
        z = 0 + np.sin(local * np.pi) * 0.7

    particle_x.append(x)
    particle_y.append(y)
    particle_z.append(z)

fig.add_trace(
    go.Scatter3d(
        x=particle_x,
        y=particle_y,
        z=particle_z,
        mode="markers",
        marker=dict(
            size=3,
            color="#58d7ff",
            opacity=.8
        ),
        hoverinfo="none",
        showlegend=False
    )
)

# ============================================================
# CONFIGURACIÓN 3D
# ============================================================

fig.update_layout(

    height=650,

    paper_bgcolor="rgba(0,0,0,0)",

    plot_bgcolor="rgba(0,0,0,0)",

    margin=dict(
        l=0,
        r=0,
        t=0,
        b=0
    ),

    scene=dict(

        bgcolor="rgba(0,0,0,0)",

        xaxis=dict(
            visible=False
        ),

        yaxis=dict(
            visible=False
        ),

        zaxis=dict(
            visible=False
        ),

        camera=dict(
            eye=dict(
                x=1.55,
                y=1.55,
                z=1.25
            )
        ),

        aspectmode="manual",

        aspectratio=dict(
            x=2.3,
            y=1.3,
            z=.8
        )
    )
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displayModeBar": False
    }
)

# ============================================================
# EXPLICACIÓN AUTOMÁTICA
# ============================================================

st.markdown("### Lectura del sistema")

if riesgo == "ESTABLE":

    mensaje = (
        f"La demanda proyectada se encuentra en {forecast_prom:.0f} unidades/día. "
        f"El inventario proyectado de {inventario_final:.0f} unidades "
        f"se mantiene por encima del punto de reorden de {rop:.0f}. "
        f"La cadena opera dentro de los parámetros definidos."
    )

elif riesgo == "MONITOREAR":

    mensaje = (
        f"El inventario proyectado ({inventario_final:.0f}) "
        f"se aproxima al punto de reorden ({rop:.0f}). "
        f"El sistema identifica una necesidad potencial de reposición "
        f"por aproximadamente {pedido_sugerido:.0f} unidades."
    )

else:

    mensaje = (
        f"La simulación detecta una condición crítica. "
        f"La combinación de demanda de {demanda_real:.0f} unidades/día, "
        f"lead time de {lead_real} días y capacidad logística de "
        f"{capacidad_real}% genera presión sobre el inventario. "
        f"Se recomienda priorizar la reposición."
    )

st.info(mensaje)

# ============================================================
# FLUJO PREDICTIVO
# ============================================================

st.markdown("### Motor de decisión")

flow_cols = st.columns(5)

flow_cols[0].markdown(
    f"**01 · DEMANDA**\n\n{demanda_real:,.0f} / día"
)

flow_cols[1].markdown(
    f"**02 · FORECAST**\n\n{forecast_prom:,.0f}"
)

flow_cols[2].markdown(
    f"**03 · INVENTARIO**\n\n{inventario_final:,.0f}"
)

flow_cols[3].markdown(
    f"**04 · ROP**\n\n{rop:,.0f}"
)

flow_cols[4].markdown(
    f"**05 · DECISIÓN**\n\n{pedido_sugerido:,.0f} unidades"
)

# ============================================================
# FORECAST
# ============================================================

with st.expander("Ver análisis predictivo"):

    forecast_df = pd.DataFrame({
        "Día": np.arange(1, 31),
        "Demanda proyectada": forecast
    })

    st.line_chart(
        forecast_df.set_index("Día")
    )

# ============================================================
# INVENTARIO
# ============================================================

with st.expander("Ver evolución del inventario"):

    inventario_sim = []

    stock = inventario_inicial

    for d in range(30):

        stock -= demanda_real

        if stock < rop:

            stock += pedido_sugerido

        inventario_sim.append(stock)

    inv_df = pd.DataFrame({
        "Día": np.arange(1, 31),
        "Inventario": inventario_sim
    })

    st.line_chart(
        inv_df.set_index("Día")
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "CCU · Predictive Supply Chain · Modelo demostrativo de soporte a decisiones logísticas"
)
