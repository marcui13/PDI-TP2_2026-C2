#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador del Instructivo Narrativo y Guía de Defensa Oral en PDF - TP N° 2
Materia: Procesamiento de Imágenes I (TUIA - FCEIA - UNR)
Año: 2026 - 2° Cuatrimestre
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable,
    Preformatted,
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Canvas de dos pasadas para numeración exacta de páginas y encabezados institucionales."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        w, h = A4

        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#1E3A8A"))
            self.drawString(42, h - 32, "TUIA - FCEIA - UNR | PDI I (2026 - C2)")
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#475569"))
            self.drawString(195, h - 32, "|  Instructivo Narrativo y Defensa Oral — TP N° 2")
            self.drawRightString(w - 42, h - 32, "Unidades 1 a 7")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.6)
            self.line(42, h - 38, w - 42, h - 38)

        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(
            42, 26,
            "Integrantes: Agustín Marquardt, Fabián Alvarez — Instructivo de Defensa TP2"
        )
        self.drawRightString(w - 42, 26, f"Página {self._pageNumber} de {page_count}")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(42, 36, w - 42, 36)
        self.restoreState()


def crear_pdf_defensa(ruta_pdf):
    doc = SimpleDocTemplate(
        ruta_pdf,
        pagesize=A4,
        leftMargin=42,
        rightMargin=42,
        topMargin=46,
        bottomMargin=46,
    )

    styles = getSampleStyleSheet()

    c_primary = colors.HexColor("#0F172A")
    c_accent = colors.HexColor("#1E3A8A")
    c_border = colors.HexColor("#CBD5E1")
    c_code_bg = colors.HexColor("#F1F5F9")
    c_box_blue = colors.HexColor("#EFF6FF")
    c_box_green = colors.HexColor("#F0FDF4")
    c_green_dark = colors.HexColor("#166534")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=16.5,
        leading=20,
        textColor=c_primary,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.8,
        leading=13.5,
        textColor=c_accent,
        spaceAfter=8,
    )
    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15,
        textColor=colors.white,
        backColor=c_accent,
        borderPadding=(5, 8, 5, 8),
        spaceBefore=10,
        spaceAfter=7,
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.2,
        leading=13.5,
        textColor=c_accent,
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.0,
        leading=12.8,
        textColor=c_primary,
        spaceAfter=4.5,
    )
    bullet_style = ParagraphStyle(
        "BulletItem",
        parent=body_style,
        leftIndent=14,
        firstLineIndent=-8,
        spaceAfter=3.5,
    )
    code_style = ParagraphStyle(
        "CodeBlock",
        fontName="Courier",
        fontSize=7.6,
        leading=9.8,
        textColor=colors.HexColor("#0F172A"),
    )
    th_style = ParagraphStyle(
        "TH",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.0,
        leading=10,
        textColor=colors.white,
    )
    td_style = ParagraphStyle(
        "TD",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.8,
        leading=10.2,
        textColor=c_primary,
    )
    td_bold = ParagraphStyle(
        "TDBold",
        parent=td_style,
        fontName="Helvetica-Bold",
        textColor=c_accent,
    )

    def caja_codigo(codigo_str):
        pre = Preformatted(codigo_str.strip("\n"), code_style)
        t = Table([[pre]], colWidths=[511])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), c_code_bg),
            ("BOX", (0, 0), (-1, -1), 0.7, c_border),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        return t

    def caja_concepto(titulo, texto, es_verde=False):
        bg = c_box_green if es_verde else c_box_blue
        borde = c_green_dark if es_verde else c_accent
        p = Paragraph(f"<b>{titulo}:</b> {texto}", body_style)
        t = Table([[p]], colWidths=[511])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), bg),
            ("BOX", (0, 0), (-1, -1), 0.9, borde),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        return t

    story = []

    # =========================================================================
    # ENCABEZADO
    # =========================================================================
    story.append(Paragraph("Instructivo Narrativo y Guía de Estudio para Defensa Oral — TP N° 2", title_style))
    story.append(Paragraph(
        "Procesamiento de Imágenes I (IA 4.4) — TUIA — FCEIA — Universidad Nacional de Rosario (2026 - C2)<br/>"
        "Autores: Agustín Marquardt, Fabián Alvarez",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.2, color=c_accent, spaceAfter=6))

    story.append(Paragraph(
        "Este documento está redactado en <b>modo narrativa, paso a paso y método por método</b>, para que puedas "
        "leerlo de corrido, explicarle fácilmente a tu compañero de grupo cómo funcionan <b>problema1_monedas_y_dados.py</b> "
        "y <b>problema2_deteccion_patentes.py</b>, y defender cada decisión técnica con los fundamentos de las <b>Unidades 1 a 7</b>.",
        body_style
    ))

    # =========================================================================
    # SECCIÓN 1: MAPA TEÓRICO (U1 A U7)
    # =========================================================================
    story.append(Paragraph("1. Mapa Teórico: Qué Usamos de Cada Unidad de la Materia (U1 a U7)", h1_style))

    datos_mapa = [
        [Paragraph("Unidad", th_style), Paragraph("Herramienta Teórica", th_style), Paragraph("Aplicación en Nuestro Código (Problemas 1 y 2)", th_style)],
        [
            Paragraph("Unidades 1 y 2", td_bold),
            Paragraph("Luminancia BGR2GRAY y Filtro No Lineal de Mediana (<b>cv2.medianBlur</b>)", td_style),
            Paragraph("En P1.A aplicamos mediana 11x11 antes de derivar: limpia la textura granulada de la mesa sin redondear los bordes de monedas y dados.", td_style)
        ],
        [
            Paragraph("Unidad 3", td_bold),
            Paragraph("Gradiente de Sobel (<b>cv2.Sobel</b>), Otsu y Componentes Conectadas (<b>connectedComponentsWithStats</b>)", td_style),
            Paragraph("En P1.A el gradiente ||G|| = sqrt(Gx^2 + Gy^2) anula la rampa de iluminación del fondo. En P2 (Puntos A y B), connectedComponentsWithStats etiqueta candidatos y devuelve (x,y,w,h), área y centroide.", td_style)
        ],
        [
            Paragraph("Unidad 6", td_bold),
            Paragraph("Apertura, Clausura, Reconstrucción Geodésica (<b>imreconstruct, imfillhole, imclearborder</b>) y <b>Top-Hat</b>", td_style),
            Paragraph("En P1: Clausura 15x15 + imfillhole + Apertura 31x31 (Punto A); imclearborder para borrar el fondo externo de los dados (Punto C). En P2: Top-Hat (21x9) para resaltar letras claras e imclearborder para borrar el marco blanco de la patente.", td_style)
        ],
        [
            Paragraph("Unidad 7", td_bold),
            Paragraph("Área (A), Perímetro (P), <b>Factor de Forma</b> (Fp = A/P^2, rho = 4*pi*Fp) y Relación de Aspecto (H/W)", td_style),
            Paragraph("En P1: rho > 0.80 separa Monedas de Dados (Punto A) y puntos circulares de cara superior de los elípticos laterales (Punto C); el Área clasifica monedas (Punto B). En P2: H/W en [1.25, 3.5] y 2 tripletes colineales aíslan la patente.", td_style)
        ],
    ]
    tabla_mapa = Table(datos_mapa, colWidths=[72, 148, 291])
    tabla_mapa.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), c_accent),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("GRID", (0, 0), (-1, -1), 0.5, c_border),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(tabla_mapa)

    # =========================================================================
    # SECCIÓN 2: OPERADORES MORFOLÓGICOS COMPARTIDOS (U6)
    # =========================================================================
    story.append(Paragraph("2. Operadores Morfológicos Compartidos (Unidad 6 — Cátedra)", h1_style))
    story.append(Paragraph(
        "Si primero le explicás estas tres funciones auxiliares a tu compañero (tomadas de "
        "<b>PI_U6_Morfologico_p2_Reconstruccion.py</b>), va a entender al instante cómo se limpian los huecos y los bordes en ambos problemas:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>imreconstruct(marker, mask, kernel):</b> Parte de una semilla <i>marker</i>, la dilata 3x3 y la intersecta con <i>mask</i> "
        "en un bucle <i>while</i> hasta que no cambia más. No engorda objetos: rescata de <i>mask</i> las componentes completas tocadas por <i>marker</i>.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>imfillhole(img):</b> Inunda con <b>cv2.floodFill</b> desde (0, 0) todo el fondo exterior conectado al borde, invierte esa "
        "inundación (quedando blancos solo los huecos cerrados) y hace <b>img | img_flood_fill_inv</b> para volver sólidos los objetos.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>imclearborder(img):</b> Pone en cero el interior de <i>marker</i> dejando solo el marco de 1 px del borde, reconstruye con "
        "<i>imreconstruct(marker, img)</i> todo objeto que toque el borde y lo resta con <b>cv2.subtract</b>. Sirve para borrar el fondo "
        "alrededor de los dados (P1.C) y el marco blanco de la patente argentina (P2.B).",
        bullet_style
    ))

    # =========================================================================
    # SECCIÓN 3: NARRATIVA DEL PROBLEMA 1 (MÉTODO POR MÉTODO)
    # =========================================================================
    story.append(Paragraph("3. Narrativa del Problema 1: Monedas y Dados (problema1_monedas_y_dados.py)", h1_style))
    story.append(Paragraph(
        "<b>Flujo de ejecución en main():</b> Lee <b>monedas.jpg</b> (3871x2681 px), llama a <b>segmentar_objetos</b> (Punto A), "
        "luego a <b>clasificar_monedas</b> (Punto B), después a <b>analizar_dados</b> (Punto C), imprime la tabla en consola y guarda "
        "las 3 figuras diagnósticas con <b>generar_figuras_problema1</b>.",
        body_style
    ))

    story.append(Paragraph("3.1. Método segmentar_objetos(img_bgr) — Punto A", h2_style))
    story.append(Paragraph(
        "El tablero tiene una iluminación muy despareja: izquierda I ~ 60 y derecha I ~ 185. Si usamos Otsu directo (T=115), toda la "
        "mitad derecha del fondo se vuelve blanca y se pega con las monedas. ¿Cómo lo resolvemos paso a paso?",
        body_style
    ))
    story.append(Paragraph("1. <b>Escala de grises + Mediana 11x11 (cv2.medianBlur — U2):</b> Atenúa el grano de la mesa sin redondear los bordes escalón.", bullet_style))
    story.append(Paragraph(
        "2. <b>Magnitud del Gradiente de Sobel (cv2.Sobel con CV_64F — U3):</b> Calculamos Gx y Gy en float64 (para no truncar derivadas "
        "negativas a 0) y hacemos sqrt(Gx^2 + Gy^2). Como la rampa del fondo varía lento (baja frecuencia), su derivada es ~0; en cambio, los "
        "bordes de monedas y dados (alta frecuencia) dan un pico alto. Se anulan 10 px de borde para que (0,0) sea negro puro.",
        bullet_style
    ))
    story.append(Paragraph("3. <b>Umbralado estadístico del gradiente (U3):</b> T = round(media + 0.9*std) = 20.0. Obtenemos los 19 anillos de borde.", bullet_style))
    story.append(Paragraph("4. <b>Clausura (15x15) + imfillhole + Apertura (31x31) (U6):</b> La clausura sella micro-cortes del borde; imfillhole rellena el interior de monedas bimetálicas y dados; la apertura 31x31 suaviza el contorno externo.", bullet_style))
    story.append(Paragraph(
        "5. <b>Componentes Conexas y Factor de Forma (U3 y U7):</b> Con <b>connectedComponentsWithStats</b> (AREA_MIN=15000) y "
        "<b>findContours</b> medimos Área (A) y Perímetro (P). Calculamos Fp = A/P^2 y <b>rho = 4*pi*Fp</b>. Con el umbral de cátedra "
        "<b>RHO_TH = 0.80</b> (clasifico_celulas.py), separamos las <b>17 Monedas</b> (rho en [0.8056, 0.8904]) de los <b>2 Dados</b> (rho = 0.7742 y 0.6590).",
        bullet_style
    ))

    story.append(Paragraph("3.2. Método clasificar_monedas(monedas) — Punto B", h2_style))
    story.append(Paragraph(
        "Con la cámara a altura fija, el <b>Área en píxeles²</b> refleja el diámetro físico de cada moneda. Con fronteras en "
        "<b>80,000 px²</b> y <b>102,000 px²</b> (más de 24,000 px² de margen libre entre clases), el método clasifica y suma:",
        body_style
    ))
    story.append(Paragraph("• <b>10 Centavos</b> (19.0 mm, Área &lt; 80k): 9 unidades ([62.6k – 67.6k] px²) &rarr; $0.90.", bullet_style))
    story.append(Paragraph("• <b>1 Peso</b> (23.0 mm, 80k &le; Área &lt; 102k): 5 unidades ([91.9k – 97.2k] px²) &rarr; $5.00.", bullet_style))
    story.append(Paragraph("• <b>50 Centavos</b> (25.2 mm, Área &ge; 102k): 3 unidades ([107.6k – 110.0k] px²) &rarr; $1.50. <b>Total: 17 monedas = $7.40.</b>", bullet_style))

    story.append(Paragraph("3.3. Método analizar_dados(img_bgr, gray, dados) — Punto C", h2_style))
    story.append(Paragraph(
        "En el Dado #1 la cámara ve los 5 puntos de la cara superior y 3 puntos de la cara lateral derecha. Para contar solo la cara superior:",
        body_style
    ))
    story.append(Paragraph("1. Recortamos la ROI del dado y aplicamos <b>umbralado inverso (T = 90)</b>: quedan blancos los puntos y el fondo exterior.", bullet_style))
    story.append(Paragraph("2. Aplicamos <b>imclearborder(pips_bw)</b> (U6): borra el fondo exterior que toca los bordes del recorte.", bullet_style))
    story.append(Paragraph("3. Aplicamos <b>Clausura (9x9) + Apertura (5x5)</b> (U6): rellena los brillos blancos dentro de los puntos negros.", bullet_style))
    story.append(Paragraph(
        "4. Medimos <b>rho = 4*pi*A/P^2</b> de cada punto (U7): los puntos de la <b>cara superior</b> (paralela a la cámara) son círculos "
        "(rho en [0.871, 0.900] &gt; 0.80, área ~ 1950 px²); los de la <b>cara lateral</b> (inclinada ~65°) sufren escorzo y son elipses "
        "angostas (rho en [0.550, 0.590] &le; 0.80, área ~ 630 px²). Resultado: <b>Dado #1 = 5</b>, <b>Dado #2 = 3</b>, <b>Suma = 8</b>.",
        bullet_style
    ))

    # =========================================================================
    # SECCIÓN 4: NARRATIVA DEL PROBLEMA 2 (MÉTODO POR MÉTODO)
    # =========================================================================
    story.append(Paragraph("4. Narrativa del Problema 2: Detección de Patentes (problema2_deteccion_patentes.py)", h1_style))
    story.append(Paragraph(
        "<b>Flujo de ejecución en main():</b> Recorre en bucle las 12 imágenes (img01.png a img12.png, 640x360 px). Para cada una ejecuta "
        "<b>detectar_y_segmentar_patente</b> (Punto A) y le pasa exclusivamente ese recorte a <b>segmentar_caracteres_patente</b> (Punto B).",
        body_style
    ))

    story.append(Paragraph("4.1. Punto A: detectar_y_segmentar_patente + filtrar_candidatos_caracter + buscar_constelacion_patente", h2_style))
    story.append(Paragraph(
        "1. <b>Transformada Top-Hat en grises (cv2.MORPH_TOPHAT, EE Rectangular 21x9 — U6):</b> Calcula <i>I_tophat = I - (I o B)</i>. "
        "Como los caracteres miden entre 5 y 13 px de ancho (siempre &lt; 21 px), la apertura <i>(I o B)</i> los borra pero conserva las "
        "grandes superficies claras (autos blancos, cielo, pavimento). Al restar, el fondo claro desaparece y brillan solo las letras.",
        bullet_style
    ))
    story.append(Paragraph(
        "2. <b>Barrido de umbral (35 a 160, paso 5) + Regla 1 en filtrar_candidatos_caracter (U3 y U7):</b> Para contemplar patentes al sol "
        "(img01, T=140) o en sombra (img06, T=45), se llama a <b>connectedComponentsWithStats</b> y se filtra por la Regla 1 de "
        "<i>info_patentes.txt</i>: <b>1.25 &le; H/W &le; 3.5</b>, H en [9, 30] px, W en [3, 20] px y Área &ge; 15. Quedan solo 7 a 12 candidatos.",
        bullet_style
    ))
    story.append(Paragraph(
        "3. <b>Reglas 2 y 3 en buscar_constelacion_patente (U7):</b> Primero busca <b>tripletes</b> (grupos de 3 caracteres de altura pareja "
        "max|h - h_mean| &le; 3.5, alineados |dy| &le; 0.35*h y espaciado regular 0.40*h &le; dx &le; 1.15*h). Luego aparea dos tripletes "
        "(3 letras + 3 dígitos) separados por el espacio central mayor (1.25*dx_intra &le; dx_central &le; 2.80*dx_intra) y colineales. "
        "El grupo de 6 que maximiza <b>Suma(Áreas) - 10*std(H)</b> define la caja contenedora + margen (0.45*hm horiz., 0.35*hm vert.). "
        "<b>Efectividad: 12/12 patentes detectadas (100%).</b>",
        bullet_style
    ))

    story.append(Paragraph("4.2. Punto B: segmentar_caracteres_patente(recorte_bgr)", h2_style))
    story.append(Paragraph(
        "El recorte de la patente argentina incluye su <b>marco blanco perimetral</b>, que al usar un umbral directo suele pegarse por 1 px "
        "con las letras. El método lo resuelve en 4 pasos:",
        body_style
    ))
    story.append(Paragraph("1. Aplica <b>Top-Hat local (21x9)</b> sobre el recorte en gris para uniformizar sombras oblicuas.", bullet_style))
    story.append(Paragraph("2. Binariza y aplica <b>imclearborder(bw_raw, conn=8)</b> (U6): como el marco blanco toca el borde del recorte, desaparece por completo.", bullet_style))
    story.append(Paragraph("3. Evalúa <b>connectedComponentsWithStats</b> en el barrido de umbrales (30 a 175) buscando exactamente <b>6 componentes internas</b> con H en [0.35*hc, 0.92*hc], 1.25 &le; H/W &le; 3.5 y max(H)-min(H) &le; 4 px.", bullet_style))
    story.append(Paragraph("4. Ordena de izquierda a derecha (C1..C3 letras, C4..C6 dígitos) y recorta cada carácter. <b>Efectividad: 72/72 caracteres (100%).</b>", bullet_style))

    # =========================================================================
    # SECCIÓN 5: PREGUNTAS FRECUENTES DE DEFENSA ORAL
    # =========================================================================
    story.append(Paragraph("5. Preguntas Clave de Examen / Defensa Oral y Respuestas", h1_style))

    preguntas = [
        (
            "P1: ¿Por qué en monedas.jpg usaron Sobel en vez de Otsu o HSV?",
            "Porque el fondo y las monedas son acromáticos y la rampa de iluminación (I~60 a I~185) afecta por igual a RGB y al canal V de HSV. "
            "Al ser una rampa suave de baja frecuencia, su derivada espacial es ~0: Sobel anula el fondo no uniforme y resalta solo los bordes."
        ),
        (
            "P2: ¿Por qué hace falta una Clausura (15x15) antes de imfillhole en el Problema 1?",
            "Porque imfillhole inunda desde (0,0). Si el anillo de Sobel tuviera una brecha de 1 px, el floodFill entraría a la moneda y no la rellenaría. "
            "La clausura suelda cualquier micro-corte del contorno."
        ),
        (
            "P3: ¿Por qué el mismo umbral rho = 0.80 funciona tanto en P1.A como en P1.C?",
            "Porque rho = 4*pi*A/P^2 es invariante a escala. En P1.A separa círculos (monedas) de cuadrados (dados). En P1.C separa puntos "
            "frontales circulares de la cara superior (rho en [0.87, 0.90]) de puntos inclinados elípticos de las caras laterales (rho en [0.55, 0.59])."
        ),
        (
            "P4: ¿Qué aporta Top-Hat en P2.A e imclearborder en P2.B?",
            "Top-Hat (21x9) suprime grandes superficies claras (capó, calle, cielo) dejando solo trazos finos blancos. En el recorte de P2.B, "
            "imclearborder elimina el marco blanco perimetral de la chapa sin tocar los 6 caracteres internos."
        ),
    ]
    for preg, resp in preguntas:
        story.append(KeepTogether([
            caja_concepto(preg, resp, es_verde=True),
            Spacer(1, 4),
        ]))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[OK] PDF instructivo generado en: {ruta_pdf}")


if __name__ == "__main__":
    salida = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Defensa_Oral_TP2_PDI.pdf")
    crear_pdf_defensa(salida)
