#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador del Documento PDF de Estudio para la Defensa Oral del Trabajo Práctico N° 2
Materia: Procesamiento Digital de Imágenes I (TUIA - FCEIA - UNR)
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
    PageBreak,
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
            self.drawString(205, h - 32, "|  Guía de Estudio para Defensa Oral — Trabajo Práctico N° 2")
            self.drawRightString(w - 42, h - 32, "Unidades 1 a 7")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.6)
            self.line(42, h - 38, w - 42, h - 38)

        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(
            42, 26,
            "Integrantes: Agustín Marquardt, Damián Turco, Fabián Alvarez — Defensa Oral TP2"
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
    c_sub = colors.HexColor("#334155")
    c_border = colors.HexColor("#CBD5E1")
    c_code_bg = colors.HexColor("#F1F5F9")
    c_box_blue = colors.HexColor("#EFF6FF")
    c_box_green = colors.HexColor("#F0FDF4")
    c_green_dark = colors.HexColor("#166534")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=21,
        textColor=c_primary,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=c_accent,
        spaceAfter=10,
    )
    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.white,
         backColor=c_accent,
        borderPadding=(5, 8, 5, 8),
        spaceBefore=10,
        spaceAfter=8,
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=c_accent,
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=13.2,
        textColor=c_primary,
        spaceAfter=5,
    )
    bullet_style = ParagraphStyle(
        "BulletItem",
        parent=body_style,
        leftIndent=14,
        firstLineIndent=-8,
        spaceAfter=4,
    )
    code_style = ParagraphStyle(
        "CodeBlock",
        fontName="Courier",
        fontSize=7.8,
        leading=10.2,
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
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
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
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        return t

    story = []

    # =========================================================================
    # PORTADA / ENCABEZADO
    # =========================================================================
    story.append(Paragraph("Guía de Estudio y Fundamentación para la Defensa Oral — TP N° 2", title_style))
    story.append(Paragraph(
        "Procesamiento de Imágenes I (IA 4.4) — TUIA — FCEIA — Universidad Nacional de Rosario (2026 - C2)<br/>"
        "Autores: Agustín Marquardt, Damián Turco, Fabián Alvarez",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.2, color=c_accent, spaceAfter=8))

    story.append(Paragraph(
        "Este documento sintetiza la explicación línea por línea de los scripts "
        "<b>problema1_monedas_y_dados.py</b> y <b>problema2_deteccion_patentes.py</b>, vinculando "
        "cada decisión algorítmica con los contenidos teóricos de las <b>Unidades 1 a 7</b> de la asignatura "
        "y respondiendo las preguntas clave que suelen formularse en la instancia de defensa oral.",
        body_style
    ))

    # =========================================================================
    # SECCIÓN 1: VINCULACIÓN CON LAS UNIDADES TEÓRICAS (U1 A U7)
    # =========================================================================
    story.append(Paragraph("1. Vinculación entre el Código Implementado y la Teoría de Cátedra (Unidades 1 a 7)", h1_style))

    datos_mapa = [
        [Paragraph("Unidad", th_style), Paragraph("Herramienta Teórica", th_style), Paragraph("Aplicación en Nuestro Código (Problemas 1 y 2)", th_style)],
        [
            Paragraph("Unidades 1 y 2", td_bold),
            Paragraph("Luminancia BGR2GRAY y Filtro No Lineal de Mediana (<b>cv2.medianBlur</b>)", td_style),
            Paragraph("En el Problema 1.A se aplica un filtro de mediana 11x11 antes de derivar. Elimina el grano superficial del tablero sin difuminar los escalones de borde de las monedas y dados.", td_style)
        ],
        [
            Paragraph("Unidad 3", td_bold),
            Paragraph("Gradiente de Sobel (<b>cv2.Sobel</b>), Umbralado de Otsu y Componentes Conectadas (<b>connectedComponentsWithStats</b>)", td_style),
            Paragraph("En el Problema 1.A el gradiente ||G|| = sqrt(Gx^2 + Gy^2) elimina la rampa suave de iluminación del fondo. En el Problema 2 (Puntos A y B), connectedComponentsWithStats etiqueta los candidatos a caracteres y extrae sus cajas (x,y,w,h), área y centroide.", td_style)
        ],
        [
            Paragraph("Unidad 6", td_bold),
            Paragraph("Morfología Binaria y en Grises: Apertura, Clausura, Reconstrucción (<b>imreconstruct, imfillhole, imclearborder</b>) y <b>Top-Hat</b>", td_style),
            Paragraph("En P1.A: Clausura 15x15 + imfillhole + Apertura 31x31 para obtener máscaras sólidas. En P1.C: imclearborder para borrar el fondo externo del recorte de cada dado. En P2.A y P2.B: Transformada Top-Hat (21x9) para resaltar caracteres claros sobre fondo oscuro e imclearborder para remover el marco blanco de la patente.", td_style)
        ],
        [
            Paragraph("Unidad 7", td_bold),
            Paragraph("Descriptores Regionales: Área (A), Perímetro (P), <b>Factor de Forma</b> (Fp = A/P^2, rho = 4*pi*Fp) y Relación de Aspecto (H/W)", td_style),
            Paragraph("En P1.A: rho > 0.80 separa las 17 Monedas (círculos) de los 2 Dados (cuadrados). En P1.B: el Área clasifica 10 ctvs (9), $1 (5) y 50 ctvs (3). En P1.C: rho > 0.80 separa los puntos circulares de la cara superior de los puntos elípticos laterales. En P2: H/W en [1.25, 3.5] y 2 tripletes colineales aíslan la patente.", td_style)
        ],
    ]
    tabla_mapa = Table(datos_mapa, colWidths=[75, 145, 291])
    tabla_mapa.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), c_accent),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("GRID", (0, 0), (-1, -1), 0.5, c_border),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(tabla_mapa)
    story.append(Spacer(1, 6))

    # =========================================================================
    # SECCIÓN 2: RECONSTRUCCIÓN MORFOLÓGICA DE LA UNIDAD 6
    # =========================================================================
    story.append(Paragraph("2. Operadores de Reconstrucción Morfológica (Unidad 6) Usados en Ambos Problemas", h1_style))
    story.append(Paragraph(
        "Siguiendo el script de cátedra <b>PI_U6_Morfologico_p2_Reconstruccion.py</b>, ambos problemas se apoyan en tres "
        "operadores geodésicos fundamentales:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>1. Reconstrucción Geodésica Binaria (imreconstruct):</b> Dada una máscara <i>G</i> y un marcador <i>F &sube; G</i>, "
        "aplica dilataciones geodésicas sucesivas <i>&delta;<sub>G</sub>(F) = (F &oplus; B) &cap; G</i> hasta convergencia. "
        "A diferencia de una dilatación estándar, <b>no deforma los objetos</b>: reconstruye íntegramente aquellas componentes conexas "
        "de la máscara que fueron tocadas por el marcador.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>2. Rellenado Automático de Huecos (imfillhole):</b> Un hueco es una región de fondo encerrada por un contorno que no "
        "tiene camino hacia el borde exterior. Inundando el fondo exterior desde la esquina (0,0) (<b>cv2.floodFill</b>) e invirtiendo "
        "el resultado, se rellenan todas las cavidades internas de las monedas y los dados sin alterar su perímetro externo.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>3. Limpieza de Objetos en Contacto con el Borde (imclearborder):</b> Construye un marcador <i>F</i> activo únicamente "
        "en la primera/última fila y primera/última columna de la imagen, reconstruye <i>R<sub>I</sub>(F)</i> y lo resta de <i>I</i>. "
        "Es la pieza clave para: <b>(a)</b> borrar el fondo exterior oscuro alrededor de cada dado en el Punto 1.C, y "
        "<b>(b)</b> borrar el marco blanco perimetral de la patente argentina en el Punto 2.B.",
        bullet_style
    ))
    story.append(caja_codigo("""
def imreconstruct(marker, mask, kernel=None):
    if kernel is None:
        kernel = np.ones((3, 3), np.uint8)
    prev = np.zeros_like(marker)
    curr = marker.copy()
    while not np.array_equal(curr, prev):
        prev = curr.copy()
        curr = cv2.min(cv2.dilate(curr, kernel), mask)  # Intersección con la máscara G
    return curr

def imclearborder(f, conn=8):
    marker = np.zeros_like(f)
    marker[0, :] = f[0, :]; marker[-1, :] = f[-1, :]
    marker[:, 0] = f[:, 0]; marker[:, -1] = f[:, -1]
    border_objs = imreconstruct(marker, f)
    return cv2.subtract(f, border_objs)
"""))

    # =========================================================================
    # SECCIÓN 3: PROBLEMA 1 (MONEDAS Y DADOS)
    # =========================================================================
    story.append(Paragraph("3. Problema 1: Detección y Clasificación de Monedas y Dados (monedas.jpg)", h1_style))

    story.append(Paragraph("3.1. Punto A: ¿Por qué falla Otsu directo y por qué el Gradiente de Sobel lo resuelve?", h2_style))
    story.append(Paragraph(
        "En <b>monedas.jpg</b> (3871x2681 px), el tablero de fondo presenta una iluminación no uniforme que crece suavemente "
        "de izquierda (I ~ 60) a derecha (I ~ 185). Si aplicamos el método de Otsu directamente sobre los niveles de gris "
        "(<b>T = 115</b>), todos los píxeles del fondo derecho con intensidad entre 116 y 185 superan el umbral y se "
        "convierten en blanco, fusionándose con las monedas y dados.",
        body_style
    ))
    story.append(caja_concepto(
        "Fundamento Teórico Clave para la Defensa (Unidad 3)",
        "Como la rampa de iluminación del fondo b(x,y) varía lentamente (baja frecuencia espacial), su primera derivada espacial "
        "es prácticamente nula (||grad b(x,y)|| ~ 0). En cambio, el borde físico de cada moneda o dado presenta una discontinuidad "
        "abrupta (alta frecuencia espacial) con un módulo de gradiente elevado. Por lo tanto, el operador de Sobel actúa como un "
        "bloqueador natural de la iluminación no uniforme."
    ))
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Secuencia exacta de procesamiento en segmentar_objetos(img_bgr):</b>", body_style))
    story.append(Paragraph(
        "1. <b>Filtro de Mediana 11x11 (cv2.medianBlur):</b> Atenúa la rugosidad del tablero preservando los bordes escalón.",
        bullet_style
    ))
    story.append(Paragraph(
        "2. <b>Magnitud del Gradiente de Sobel (cv2.Sobel con CV_64F):</b> Se calculan Gx y Gy en punto flotante de 64 bits "
        "(para no truncar derivadas negativas a 0) y se obtiene la norma exacta ||G|| = sqrt(Gx^2 + Gy^2) con <b>cv2.magnitude</b>.",
        bullet_style
    ))
    story.append(Paragraph(
        "3. <b>Umbralado Estadístico (T = round(media + 0.9*std) = 20.0):</b> Separa los anillos de borde de los 19 objetos.",
        bullet_style
    ))
    story.append(Paragraph(
        "4. <b>Clausura (Elipse 15x15) + Rellenado (imfillhole) + Apertura (Elipse 31x31):</b> La clausura sella brechas en los "
        "contornos; imfillhole rellena el interior completo de las 17 monedas y 2 dados; la apertura 31x31 alisa el perímetro.",
        bullet_style
    ))

    story.append(Paragraph("3.2. Punto A: Discriminación Moneda vs. Dado mediante el Factor de Forma (Unidad 7)", h2_style))
    story.append(Paragraph(
        "Sobre los 19 contornos exteriores extraídos con <b>cv2.findContours(..., cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)</b>, "
        "se mide el Área (A) y el Perímetro (P). De acuerdo con <b>Factor de Forma.txt</b> y <b>u6/clasifico_celulas.py</b>:",
        body_style
    ))
    story.append(caja_codigo("""
# Factor de Forma (Factor de Forma.txt): Fp = A / P^2
#   - Círculo perfecto : Fp = (pi * r^2) / (2 * pi * r)^2 = 1 / (4*pi) = 0.0796  -> 1/Fp = 12.57
#   - Cuadrado perfecto: Fp = a^2 / (4*a)^2               = 1 / 16     = 0.0625  -> 1/Fp = 16.00
# Circularidad normalizada (clasifico_celulas.py): rho = 4 * pi * Fp = 4 * pi * A / P^2
#   - Umbral de cátedra: RHO_TH = 0.80 (equivalente a 1/Fp = 15.71)
#   - 17 Monedas detectadas: rho en [0.8056, 0.8904] > 0.80
#   -  2 Dados detectados  : rho = 0.7742 y rho = 0.6590 <= 0.80
"""))

    story.append(Paragraph("3.3. Punto B: Clasificación Granulométrica y Conteo Automático de Monedas", h2_style))
    story.append(Paragraph(
        "Dado que la cámara se encuentra a distancia constante del plano de la mesa, el <b>Área en píxeles cuadrados</b> "
        "de cada moneda es proporcional al cuadrado de su diámetro físico. Ordenando las 17 áreas de menor a mayor y "
        "detectando automáticamente los dos mayores saltos (<b>np.diff(np.sort(areas))</b>), el código ubica las fronteras "
        "óptimas en <b>U1 = 79,744 px²</b> y <b>U2 = 102,381 px²</b>:",
        body_style
    ))
    story.append(Paragraph("• <b>10 Centavos (Área &lt; 79,744 px²):</b> 9 monedas (rango [62,612 – 67,606] px²) &rarr; $0.90.", bullet_style))
    story.append(Paragraph("• <b>1 Peso (79,744 &le; Área &lt; 102,381 px²):</b> 5 monedas (rango [91,882 – 97,154] px²) &rarr; $5.00.", bullet_style))
    story.append(Paragraph("• <b>50 Centavos (Área &ge; 102,381 px²):</b> 3 monedas (rango [107,607 – 110,049] px²) &rarr; $1.50.", bullet_style))
    story.append(Paragraph("• <b>Total Conteo Punto B:</b> 17 monedas clasificadas con margen > 24,000 px² entre clases (Monto Total = <b>$7.40</b>).", bullet_style))

    story.append(Paragraph("3.4. Punto C: Valor de la Cara Superior de Cada Dado (Descarte de Puntos Laterales)", h2_style))
    story.append(Paragraph(
        "En el Dado #1, debido a la perspectiva oblicua de la cámara, son visibles tanto los <b>5 puntos de la cara superior</b> "
        "como <b>3 puntos de la cara lateral inclinada</b>. La función <b>analizar_dados</b> los separa en 4 pasos:",
        body_style
    ))
    story.append(Paragraph("1. <b>Umbralado Inverso (T = 90, THRESH_BINARY_INV):</b> Activa en blanco (255) los puntos oscuros y el fondo exterior.", bullet_style))
    story.append(Paragraph("2. <b>imclearborder (Unidad 6):</b> Elimina el fondo exterior conectado al borde del recorte, dejando solo los puntos internos.", bullet_style))
    story.append(Paragraph("3. <b>Clausura (9x9) + Apertura (5x5):</b> Rellena los reflejos de luz dentro de cada punto negro.", bullet_style))
    story.append(Paragraph(
        "4. <b>Discriminación por Factor de Forma (rho > 0.80):</b> La cara superior es paralela al sensor y sus puntos se proyectan "
        "como <b>círculos</b> (rho en [0.871, 0.900] > 0.80, área ~ 1950 px²). La cara lateral está inclinada ~65° y sus puntos sufren "
        "escorzo proyectivo, proyectándose como <b>elipses alargadas</b> (rho en [0.550, 0.590] <= 0.80, área ~ 630 px²). "
        "Resultado: <b>Dado #1 = 5</b>, <b>Dado #2 = 3</b>, <b>Suma Total = 8</b>.",
        bullet_style
    ))

    # =========================================================================
    # SECCIÓN 4: PROBLEMA 2 (DETECCIÓN DE PATENTES Y SEGMENTACIÓN DE CARACTERES)
    # =========================================================================
    story.append(Paragraph("4. Problema 2: Detección de Patentes y Segmentación de Caracteres (img01 a img12)", h1_style))

    story.append(Paragraph("4.1. Punto A: Transformada Top-Hat en Niveles de Gris (Unidad 6)", h2_style))
    story.append(Paragraph(
        "En las 12 imágenes vehiculares (640x360 px), los caracteres de la patente son trazos blancos delgados (ancho entre 5 y 13 px) "
        "sobre el fondo negro de la chapa. Sin embargo, en la escena hay grandes regiones mucho más brillantes (autos blancos o plateados, "
        "cielo, reflejos en paragolpes y pavimento). Para resaltar los caracteres suprimiendo simultáneamente el fondo claro, aplicamos la "
        "<b>Transformada Top-Hat</b> (<b>PI_U6_Morfologico_p4_Grises.py</b>):",
        body_style
    ))
    story.append(caja_codigo("""
se_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 9))
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se_tophat)  # I_tophat = I - (I o B)
"""))
    story.append(Spacer(1, 4))
    story.append(caja_concepto(
        "¿Por qué se eligió un Elemento Estructurante Rectangular de 21x9 píxeles?",
        "Porque en la apertura morfológica (I o B), toda estructura brillante cuyo ancho sea menor que el ancho del elemento estructurante "
        "(21 px) es eliminada por la erosión y no se recupera en la dilatación. Como los caracteres de las 12 patentes miden entre 5 y 13 px "
        "de ancho (siempre < 21 px), la apertura borra los 6 caracteres y estima solo el fondo luminoso envolvente. Al restar I - (I o B), "
        "el fondo desaparece y emergen los 6 caracteres con máximo contraste."
    ))

    story.append(Paragraph("4.2. Punto A: Las 3 Reglas Geométricas de Dominio (info_patentes.txt y Unidad 7)", h2_style))
    story.append(Paragraph(
        "Sobre la imagen Top-Hat se realiza un barrido de umbralización (de 35 a 160 con paso 5) para contemplar tanto patentes al sol "
        "(img01, T=140) como patentes en sombra (img06, T=45). En cada umbral se etiquetan las componentes conectadas con "
        "<b>cv2.connectedComponentsWithStats</b> y se aplican las 3 reglas de <b>info_patentes.txt</b>:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Regla 1 (Relación Alto/Ancho de cada carácter ~ 1.5 a 3.0):</b> En <b>filtrar_candidatos_caracter</b> se exige "
        "1.25 &le; H/W &le; 3.5, altura H &isin; [9, 30] px, ancho W &isin; [3, 20] px y Área &ge; 15 px². Esto filtra el 99% del ruido.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Regla 2 (Los caracteres forman grupos de 3 — Tripletes):</b> En <b>buscar_constelacion_patente</b> se buscan ternas "
        "(c1, c2, c3) ordenadas de izquierda a derecha con alturas uniformes (max|h - h_mean| &le; 3.5 px), alineación cuasi-horizontal "
        "(|dy| &le; 0.35*h_mean) y espaciado regular entre centroides (0.40*h_mean &le; dx &le; 1.15*h_mean).",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Regla 3 (La patente está formada por dos grupos de 3 caracteres):</b> Se aparean dos tripletes T1 (letras) y T2 (números) "
        "de igual altura media separados por el espacio central de la chapa (1.25*dx_intra &le; dx_central &le; 2.80*dx_intra) y colineales "
        "sobre la misma recta. La constelación de 6 caracteres con mayor puntaje <b>Score = Suma(Áreas) - 10*std(Alturas)</b> define la "
        "patente detectada (<b>12/12 patentes detectadas = 100%</b>).",
        bullet_style
    ))

    story.append(Paragraph("4.3. Punto B: Segmentación de los 6 Caracteres Individuales y Remoción del Marco con imclearborder", h2_style))
    story.append(Paragraph(
        "A partir exclusivamente del recorte de la patente obtenida en el Punto A (<b>recorte_bgr</b>), la función "
        "<b>segmentar_caracteres_patente</b> debe aislar las 6 letras/dígitos individuales. El obstáculo crítico en las patentes "
        "argentinas es el <b>marco blanco perimetral</b> de la chapa: si se binariza con un umbral global simple (como Otsu directo), "
        "el marco blanco queda unido por puentes de 1 px con las letras, fusionando varios caracteres.",
        body_style
    ))
    story.append(Paragraph("<b>Solución en 4 pasos (Unidades 3, 6 y 7):</b>", body_style))
    story.append(Paragraph("1. <b>Top-Hat Local (21x9)</b> sobre el recorte en gris de la patente.", bullet_style))
    story.append(Paragraph(
        "2. <b>Umbralado + imclearborder (Unidad 6):</b> Como el marco blanco perimetral toca los bordes exteriores del recorte mientras "
        "que los 6 caracteres están en el interior, <b>imclearborder</b> elimina el marco blanco completo.",
        bullet_style
    ))
    story.append(Paragraph(
        "3. <b>Verificación de las 6 Componentes Conexas (connectedComponentsWithStats):</b> Se selecciona el umbral local T_B que produce "
        "exactamente 6 componentes con 1.25 &le; H/W &le; 3.5, altura entre 35% y 92% del alto del recorte y dispersión de alturas "
        "max(H) - min(H) &le; 4 px.",
        bullet_style
    ))
    story.append(Paragraph(
        "4. <b>Ordenamiento Izquierda-Derecha:</b> Se ordenan por coordenada x creciente para entregar C1..C3 (letras) y C4..C6 (dígitos), "
        "logrando segmentar los <b>72/72 caracteres individuales (100%)</b>.",
        bullet_style
    ))

    # =========================================================================
    # SECCIÓN 5: PREGUNTAS FRECUENTES DE DEFENSA ORAL
    # =========================================================================
    story.append(Paragraph("5. Preguntas Clave de Examen / Defensa Oral y Respuestas", h1_style))

    preguntas = [
        (
            "P1: ¿Por qué en monedas.jpg usaron el Gradiente de Sobel en vez de umbralizar con Otsu o pasar a HSV?",
            "Porque tanto el fondo como la mayoría de las monedas y los dados son acromáticos (baja saturación), y la rampa de iluminación "
            "de izquierda (I~60) a derecha (I~185) afecta por igual a los canales RGB y al canal Brillo (V) de HSV. En cambio, al ser una "
            "variación suave de baja frecuencia, su derivada espacial es casi cero. El Gradiente de Sobel elimina la rampa de iluminación "
            "y deja únicamente los bordes de alta frecuencia de los 19 objetos."
        ),
        (
            "P2: ¿Por qué es imprescindible aplicar una Clausura Morfológica (15x15) antes de llamar a imfillhole en el Problema 1?",
            "Porque imfillhole rellena regiones encerradas inundando el fondo desde el exterior (0,0). Si el anillo de borde obtenido con "
            "Sobel tuviera aunque sea una brecha de 1 píxel sin cerrar, la inundación entraría al interior de la moneda o del dado y no lo "
            "rellenaría. La clausura con elipse 15x15 suelda cualquier micro-interrupción del contorno antes del rellenado."
        ),
        (
            "P3: ¿Cómo logran que el mismo descriptor (rho = 4*pi*A/P^2 con umbral 0.80) sirva tanto en el Punto 1.A como en el Punto 1.C?",
            "Por la invariancia a la escala del Factor de Forma. En el Punto 1.A (escala macro, áreas ~ 70,000 px²) separa círculos "
            "(monedas, rho > 0.80) de cuadrados proyectados (dados, rho <= 0.80). En el Punto 1.C (escala micro, áreas ~ 2,000 px²) separa "
            "los puntos de la cara superior (que se ven circulares de frente, rho en [0.87, 0.90] > 0.80) de los puntos de las caras laterales "
            "inclinadas (que por escorzo proyectivo se ven como elipses angostas con rho en [0.55, 0.59] <= 0.80)."
        ),
        (
            "P4: ¿Qué ventaja aporta combinar Top-Hat en el Punto 2.A con imclearborder en el Punto 2.B para las patentes?",
            "En el Punto 2.A (escena completa), el Top-Hat (21x9) suprime las grandes superficies brillantes (autos blancos, calle, cielo) "
            "y resalta los trazos finos de la patente. Una vez recortada la patente, en el Punto 2.B el principal distractor pasa a ser el "
            "marco blanco perimetral de la chapa que toca los bordes del recorte; allí imclearborder reconstruye desde el borde del recorte "
            "y suprime ese marco sin afectar ninguno de los 6 caracteres centrales."
        ),
    ]
    for preg, resp in preguntas:
        story.append(KeepTogether([
            caja_concepto(preg, resp, es_verde=True),
            Spacer(1, 5),
        ]))

    # =========================================================================
    # SECCIÓN 6: DICCIONARIO RÁPIDO DE FUNCIONES OPENCV Y NUMPY
    # =========================================================================
    story.append(Paragraph("6. Diccionario Rápido de Funciones OpenCV (cv2) y NumPy (np) para la Defensa", h1_style))
    datos_cv = [
        [Paragraph("Función / Método", th_style), Paragraph("Unidad", th_style), Paragraph("Parámetros Clave y Retorno en el Código", th_style)],
        [Paragraph("cv2.medianBlur(gray, 11)", td_bold), Paragraph("U2", td_style), Paragraph("Ventana impar 11x11. Reemplaza cada píxel por la mediana local, atenuando textura sin difuminar bordes.", td_style)],
        [Paragraph("cv2.Sobel(img, cv2.CV_64F, dx, dy, ksize=3)", td_bold), Paragraph("U3", td_style), Paragraph("Calcula derivadas Gx (1,0) y Gy (0,1) en float64 para no truncar valores negativos; cv2.magnitude(gx,gy) da ||G||.", td_style)],
        [Paragraph("cv2.getStructuringElement(shape, (w,h))", td_bold), Paragraph("U6", td_style), Paragraph("MORPH_ELLIPSE (15x15 y 31x31 en P1) y MORPH_RECT (21x9 en P2). Define la geometría del elemento estructurante B.", td_style)],
        [Paragraph("cv2.morphologyEx(src, op, kernel)", td_bold), Paragraph("U6", td_style), Paragraph("MORPH_CLOSE (clausura), MORPH_OPEN (apertura) y MORPH_TOPHAT (I - I o B). Operaciones morfológicas compuestas.", td_style)],
        [Paragraph("cv2.findContours(bw, RETR_EXTERNAL, CHAIN_APPROX_NONE)", td_bold), Paragraph("U3 / U7", td_style), Paragraph("Extrae solo los contornos exteriores guardando todos los vértices para medir cv2.contourArea(c) y cv2.arcLength(c, True).", td_style)],
        [Paragraph("cv2.connectedComponentsWithStats(bw, 8)", td_bold), Paragraph("U3 / U7", td_style), Paragraph("Devuelve (num_labels, labels, stats, centroids). stats[i] = [x, y, w, h, area] y centroids[i] = [cx, cy].", td_style)],
    ]
    tabla_cv = Table(datos_cv, colWidths=[175, 45, 291])
    tabla_cv.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), c_accent),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("GRID", (0, 0), (-1, -1), 0.5, c_border),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(tabla_cv)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[OK] PDF de estudio para defensa oral generado en: {ruta_pdf}")


if __name__ == "__main__":
    salida = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Defensa_Oral_TP2_PDI.pdf")
    crear_pdf_defensa(salida)
