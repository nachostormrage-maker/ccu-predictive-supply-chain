import io

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)

from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_CENTER, TA_LEFT


def generar_pdf_bytes(df, kpis):

    """
    Genera un informe ejecutivo PDF para la
    Cadena de Suministro Predictiva de CCU.

    Los indicadores se obtienen directamente
    desde los datos utilizados por la aplicación.
    """

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

    # ========================================================
    # ESTILOS
    # ========================================================

    title = ParagraphStyle(
        "CCUTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#123B5D"),
        fontSize=22,
        leading=26,
        spaceAfter=10
    )

    subtitle = ParagraphStyle(
        "CCUSubtitle",
        parent=styles["Heading2"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#176B9C"),
        fontSize=13,
        leading=17
    )

    section = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        alignment=TA_LEFT,
        textColor=colors.HexColor("#123B5D"),
        fontSize=14,
        leading=18,
        spaceBefore=8,
        spaceAfter=8
    )

    body = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        alignment=TA_LEFT,
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#334155")
    )

    small = ParagraphStyle(
        "Small",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748B")
    )

    content = []

    # ========================================================
    # PORTADA
    # ========================================================

    content.append(Spacer(1, 35))

    content.append(
        Paragraph(
            "CADENA DE SUMINISTRO<br/>PREDICTIVA DE CCU",
            title
        )
    )

    content.append(
        Paragraph(
            "Informe Ejecutivo de Analítica Logística",
            subtitle
        )
    )

    content.append(Spacer(1, 30))

    portada = Table(
        [
            [
                Paragraph(
                    "<b>Proyecto</b><br/>"
                    "Sistema de soporte a decisiones para "
                    "la cadena de suministro",
                    body
                )
            ],
            [
                Paragraph(
                    "<b>Empresa analizada</b><br/>"
                    "Compañía Cervecerías Unidas S.A. (CCU)",
                    body
                )
            ],
            [
                Paragraph(
                    "<b>Enfoque</b><br/>"
                    "Forecast de demanda, inventario y reposición",
                    body
                )
            ],
            [
                Paragraph(
                    "<b>Responsable</b><br/>"
                    "Ignacio Álvarez",
                    body
                )
            ]
        ],
        colWidths=[470]
    )

    portada.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                colors.HexColor("#F1F5F9")
            ),
            (
                "BOX",
                (0, 0),
                (-1, -1),
                1,
                colors.HexColor("#CBD5E1")
            ),
            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#E2E8F0")
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                15
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                15
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                12
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                12
            )
        ])
    )

    content.append(portada)

    content.append(Spacer(1, 30))

    content.append(
        Paragraph(
            "Documento generado automáticamente por "
            "la plataforma de analítica.",
            small
        )
    )

    # ========================================================
    # RESUMEN EJECUTIVO
    # ========================================================

    content.append(PageBreak())

    content.append(
        Paragraph(
            "1. Resumen Ejecutivo",
            section
        )
    )

    content.append(
        Paragraph(
            """
            La Cadena de Suministro Predictiva de CCU propone utilizar
            información histórica de demanda para apoyar la planificación
            de inventarios y las decisiones de reposición.
            <br/><br/>
            El modelo integra tres niveles principales: pronóstico de
            demanda, análisis del inventario y cálculo de parámetros
            de reposición.
            <br/><br/>
            De esta manera, la información operacional puede transformarse
            en indicadores y recomendaciones para apoyar la toma de
            decisiones logísticas.
            """,
            body
        )
    )

    content.append(Spacer(1, 15))

    # ========================================================
    # KPIs
    # ========================================================

    content.append(
        Paragraph(
            "2. Indicadores Clave",
            section
        )
    )

    fill_rate = float(
        kpis.get("fill_rate", 0)
    )

    mae = float(
        kpis.get("mae", 0)
    )

    inventario = float(
        kpis.get("inventario_prom", 0)
    )

    cobertura = float(
        kpis.get("cobertura_inventario", 0)
    )

    kpi_table = Table(
        [
            [
                Paragraph(
                    f"<b>Nivel de servicio</b><br/>"
                    f"{fill_rate:.1%}",
                    body
                ),

                Paragraph(
                    f"<b>Error medio</b><br/>"
                    f"{mae:.1f}",
                    body
                ),

                Paragraph(
                    f"<b>Inventario promedio</b><br/>"
                    f"{inventario:,.0f}",
                    body
                ),

                Paragraph(
                    f"<b>Cobertura</b><br/>"
                    f"{cobertura:.1f} días",
                    body
                )
            ]
        ],
        colWidths=[
            117,
            117,
            117,
            117
        ]
    )

    kpi_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (0, 0),
                colors.HexColor("#123B5D")
            ),
            (
                "BACKGROUND",
                (1, 0),
                (1, 0),
                colors.HexColor("#176B9C")
            ),
            (
                "BACKGROUND",
                (2, 0),
                (2, 0),
                colors.HexColor("#2188B5")
            ),
            (
                "BACKGROUND",
                (3, 0),
                (3, 0),
                colors.HexColor("#2A9D8F")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, -1),
                colors.white
            ),
            (
                "ALIGN",
                (0, 0),
                (-1, -1),
                "CENTER"
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                12
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                12
            )
        ])
    )

    content.append(kpi_table)

    # ========================================================
    # MODELO PREDICTIVO
    # ========================================================

    content.append(Spacer(1, 18))

    content.append(
        Paragraph(
            "3. Modelo de Cadena Predictiva",
            section
        )
    )

    flujo = Table(
        [
            [
                "Demanda\nHistórica",
                "Forecast",
                "Inventario",
                "Optimización",
                "Decisión"
            ]
        ],
        colWidths=[
            94,
            94,
            94,
            94,
            94
        ]
    )

    flujo.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                colors.HexColor("#E8F4FD")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, -1),
                colors.HexColor("#123B5D")
            ),
            (
                "BOX",
                (0, 0),
                (-1, -1),
                1,
                colors.HexColor("#176B9C")
            ),
            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#CBD5E1")
            ),
            (
                "ALIGN",
                (0, 0),
                (-1, -1),
                "CENTER"
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, -1),
                "Helvetica-Bold"
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                12
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                12
            )
        ])
    )

    content.append(flujo)

    content.append(Spacer(1, 15))

    content.append(
        Paragraph(
            """
            <b>Interpretación:</b> el sistema utiliza la demanda histórica
            como entrada, genera una estimación de demanda futura y utiliza
            esa información junto con el inventario y el lead time para
            apoyar decisiones de reposición.
            """,
            body
        )
    )

    # ========================================================
    # IMPACTO LOGÍSTICO
    # ========================================================

    content.append(
        Paragraph(
            "4. Impacto Logístico",
            section
        )
    )

    content.append(
        Paragraph(
            """
            La utilización de una herramienta predictiva puede apoyar
            la gestión de inventarios mediante una mayor visibilidad
            sobre la demanda esperada.
            <br/><br/>
            Los principales ámbitos de impacto corresponden a:
            <br/>
            • planificación de inventarios<br/>
            • reducción del riesgo de quiebres<br/>
            • control de sobrestock<br/>
            • planificación de reposición<br/>
            • utilización de información histórica<br/>
            • apoyo a decisiones operacionales
            """,
            body
        )
    )

    # ========================================================
    # LIMITACIONES
    # ========================================================

    content.append(
        Paragraph(
            "5. Consideraciones del Modelo",
            section
        )
    )

    content.append(
        Paragraph(
            """
            Para efectos académicos, la plataforma utiliza datos
            simulados. Por lo tanto, los indicadores y recomendaciones
            obtenidos no representan resultados financieros reales de CCU.
            <br/><br/>
            Para una implementación empresarial sería necesario integrar
            datos reales de ventas, inventario, centros de distribución,
            proveedores, costos logísticos y tiempos de abastecimiento.
            """,
            body
        )
    )

    # ========================================================
    # CONCLUSIÓN
    # ========================================================

    content.append(
        Paragraph(
            "6. Conclusión",
            section
        )
    )

    content.append(
        Paragraph(
            """
            La Cadena de Suministro Predictiva permite visualizar cómo
            herramientas de analítica pueden apoyar la gestión logística.
            <br/><br/>
            La integración de forecast, inventario y optimización permite
            pasar desde una visión principalmente reactiva hacia una
            gestión basada en información y proyecciones.
            <br/><br/>
            El siguiente nivel de desarrollo consiste en incorporar
            información operacional real y modelos predictivos de mayor
            precisión para evaluar la viabilidad económica de una
            implementación empresarial.
            """,
            body
        )
    )

    content.append(Spacer(1, 25))

    content.append(
        Paragraph(
            "<b>CADENA DE SUMINISTRO PREDICTIVA DE CCU</b>",
            subtitle
        )
    )

    content.append(Spacer(1, 10))

    content.append(
        Paragraph(
            "Informe generado desde la plataforma Streamlit.",
            small
        )
    )

    # ========================================================
    # GENERAR PDF
    # ========================================================

    doc.build(content)

    buffer.seek(0)

    return buffer.read()
