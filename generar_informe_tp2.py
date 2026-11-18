#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
GENERADOR DEL INFORME TÉCNICO EN PDF - TRABAJO PRÁCTICO N° 2 (PDI - TUIA - UNR)
===============================================================================
Genera el documento formal `Informe_TP2_PDI.pdf` con el desarrollo teórico,
análisis de problemáticas, descripción de las técnicas empleadas (Unidades 1-7),
tablas cuantitativas, capturas de todas las etapas intermedias y conclusiones.

Autores:
    - Agustín Marquardt
    - Damián Turco
    - Fabián Alvarez
===============================================================================
"""

import os
import textwrap

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch


# Dimensiones A4 en pulgadas
A4_W, A4_H = 8.27, 11.69


def nueva_pagina(titulo_seccion, num_pagina, total_paginas=6):
    """Crea un lienzo A4 con encabezado institucional y pie de página numerado."""
    fig = plt.figure(figsize=(A4_W, A4_H), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 141.4)
    ax.axis("off")

    # Barra superior institucional
    ax.add_patch(FancyBboxPatch(
        (4.5, 134.8), 91.0, 4.2,
        boxstyle="round,pad=0.2,rounding_size=0.4",
        facecolor="#12284c", edgecolor="none"
    ))
    ax.text(
        6.0, 136.9,
        "UNR - FCEIA - TUIA | Procesamiento de Imágenes I | TP N° 2 (2026 - C2)",
        color="white", fontsize=7.6, fontweight="bold", va="center"
    )
    ax.text(
        94.0, 136.9,
        titulo_seccion,
        color="#bcd4ff", fontsize=7.5, fontweight="bold", va="center", ha="right"
    )

    # Pie de página
    ax.plot([4.5, 95.5], [4.8, 4.8], color="#cbd5e1", linewidth=0.8)
    ax.text(
        5.0, 3.0,
        "Integrantes: Agustín Marquardt, Damián Turco, Fabián Alvarez",
        color="#475569", fontsize=7.8, va="center"
    )
    ax.text(
        95.0, 3.0,
        f"Página {num_pagina} de {total_paginas}",
        color="#475569", fontsize=7.8, fontweight="bold", va="center", ha="right"
    )
    return fig, ax


def bloque_texto(ax, x, y, parrafo, width=108, fontsize=8.4, line_spacing=1.52, color="#1e293b", bold=False):
    """Dibuja un párrafo ajustado al ancho especificado y devuelve la nueva coordenada y."""
    lineas = textwrap.wrap(parrafo, width=width)
    weight = "bold" if bold else "normal"
    for linea in lineas:
        ax.text(x, y, linea, fontsize=fontsize, color=color, fontweight=weight, va="top", family="sans-serif")
        y -= line_spacing
    return y


def subtitulo_caja(ax, y, texto, color_bg="#e8eef7", color_borde="#1e3a8a", color_txt="#0f172a"):
    """Dibuja un subtítulo destacado con banda lateral."""
    ax.add_patch(FancyBboxPatch(
        (4.8, y - 3.0), 90.4, 3.2,
        boxstyle="round,pad=0.1,rounding_size=0.3",
        facecolor=color_bg, edgecolor=color_borde, linewidth=0.8
    ))
    ax.text(6.2, y - 1.4, texto, fontsize=9.2, fontweight="bold", color=color_txt, va="center")
    return y - 4.2


def insertar_imagen(fig, ruta_img, rect_norm, titulo_pie=None, ax_base=None, y_pie=None):
    """Inserta una imagen PNG de alta resolución dentro de la página A4."""
    if not os.path.exists(ruta_img):
        return
    img = mpimg.imread(ruta_img)
    ax_img = fig.add_axes(rect_norm)
    ax_img.imshow(img)
    ax_img.axis("off")
    if titulo_pie and ax_base is not None and y_pie is not None:
        ax_base.text(
            50.0, y_pie, titulo_pie,
            fontsize=6.9, style="italic", color="#334155", ha="center", va="top"
        )


def dibujar_tabla(ax, x0, y_top, col_widths, headers, rows, row_h=2.05, font_size=7.1):
    """Dibuja una tabla académica formateada sobre el lienzo."""
    total_w = sum(col_widths)
    # Encabezado
    ax.add_patch(FancyBboxPatch(
        (x0, y_top - row_h), total_w, row_h,
        boxstyle="square,pad=0", facecolor="#1e3a8a", edgecolor="#cbd5e1", linewidth=0.5
    ))
    cx = x0
    for w, h_txt in zip(col_widths, headers):
        ax.text(cx + w / 2.0, y_top - row_h / 2.0, h_txt, color="white",
                fontsize=font_size, fontweight="bold", ha="center", va="center")
        cx += w

    y_curr = y_top - row_h
    for r_idx, row in enumerate(rows):
        bg = "#f8fafc" if r_idx % 2 == 0 else "#eef2f6"
        ax.add_patch(FancyBboxPatch(
            (x0, y_curr - row_h), total_w, row_h,
            boxstyle="square,pad=0", facecolor=bg, edgecolor="#cbd5e1", linewidth=0.4
        ))
        cx = x0
        for w, val in zip(col_widths, row):
            ax.text(cx + w / 2.0, y_curr - row_h / 2.0, str(val), color="#0f172a",
                    fontsize=font_size, ha="center", va="center")
            cx += w
        y_curr -= row_h

    return y_curr


def generar_pdf(out_pdf):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dir_p1 = os.path.join(base_dir, "resultados_problema1")
    dir_p2 = os.path.join(base_dir, "resultados_problema2")

    with PdfPages(out_pdf) as pdf:
        # =====================================================================
        # PÁGINA 1: PORTADA / RESUMEN, PUNTO A Y HISTOGRAMA DE grad_u8
        # =====================================================================
        fig1, ax1 = nueva_pagina("Problema 1: Monedas y Dados (Punto A)", 1, 7)

        # Bloque de título principal
        ax1.add_patch(FancyBboxPatch(
            (4.8, 120.8), 90.4, 12.2,
            boxstyle="round,pad=0.2,rounding_size=0.6",
            facecolor="#f1f5f9", edgecolor="#1e3a8a", linewidth=1.2
        ))
        ax1.text(
            50.0, 130.6,
            "TRABAJO PRÁCTICO N° 2 — PROCESAMIENTO DE IMÁGENES I",
            fontsize=12.5, fontweight="bold", color="#0f172a", ha="center", va="center"
        )
        ax1.text(
            50.0, 127.4,
            "Segmentación Morfológica, Reconstrucción Geodésica y Clasificación por Descriptores Regionales",
            fontsize=9.3, fontweight="bold", color="#1e3a8a", ha="center", va="center"
        )
        ax1.text(
            50.0, 124.4,
            "Tecnicatura Universitaria en Inteligencia Artificial (TUIA) — FCEIA — Universidad Nacional de Rosario",
            fontsize=8.2, color="#334155", ha="center", va="center"
        )
        ax1.text(
            50.0, 122.0,
            "Autores: Agustín Marquardt  |  Damián Turco  |  Fabián Alvarez     •     Año Académico: 2026 (2° Cuatrimestre)",
            fontsize=8.2, fontweight="bold", color="#0f172a", ha="center", va="center"
        )

        y = 118.8
        y = subtitulo_caja(ax1, y, "1. PROBLEMA 1: DETECCIÓN Y CLASIFICACIÓN DE MONEDAS Y DADOS (monedas.jpg)")

        y = bloque_texto(
            ax1, 5.2, y,
            "1.1. Análisis de la Problemática: La escena 'monedas.jpg' (3871 x 2681 px) contiene 19 objetos apoyados sobre una "
            "superficie gris oscura cuya iluminación presenta un severo gradiente espacial de baja frecuencia: la intensidad media "
            "del fondo crece desde I ~ 60 en el extremo izquierdo hasta I ~ 185 en el extremo derecho. Al aplicar un método de "
            "umbralización global directa como Otsu (Unidad 3, T = 115), toda la mitad derecha del fondo supera el umbral y se "
            "fusiona con los objetos, mientras que el núcleo oscuro de las monedas bimetálicas y los puntos negros de los dados "
            "generan perforaciones internas. Adicionalmente, la toma oblicua hace visibles los puntos de las caras laterales del "
            "Dado #1, exigiendo un criterio geométrico riguroso para aislar únicamente el valor de su cara superior.",
            width=112, fontsize=8.1, line_spacing=1.45
        )
        y -= 0.8

        y = bloque_texto(
            ax1, 5.2, y,
            "1.2. Punto A — Segmentación Automática sobre Fondo No Uniforme: Dado que la variación de iluminación del fondo b(x,y) "
            "es suave y de baja frecuencia espacial, su derivada espacial es despreciable (||grad b(x,y)|| ~ 0), mientras que los "
            "contornos de las monedas y dados presentan transiciones abruptas. La tubería implementada en 'problema1_monedas_y_dados.py' "
            "consta de seis etapas fundamentadas en las Unidades 2, 3, 6 y 7:",
            width=112, fontsize=8.1, line_spacing=1.45
        )

        pasos_p1a = [
            "• Etapa 1 (Filtrado Espacial No Lineal - U2): Conversión a gris y filtro de mediana 11x11 (cv2.medianBlur) para atenuar la rugosidad del tablero sin difuminar los bordes.",
            "• Etapa 2 (Operador Diferencial de Sobel - U3): Cálculo de la magnitud exacta del gradiente ||G|| = sqrt(Gx^2 + Gy^2) con kernels 3x3 (cv2.Sobel), eliminando el sesgo de iluminación.",
            "• Etapa 3 (Umbralado Estadístico Dinámico - U3): Como el histograma de grad_u8 es fuertemente asimétrico a la derecha (masa en 0 y cola de bordes), Otsu falla (T=65). Se barre k en [0.5, 1.5] para T = round(media + k*desv), eligiendo el mayor k que maximiza objetos cerrados válidos (k=0.9 -> T=20.0).",
            "• Etapa 4 (Clausura + Reconstrucción Geodésica - U6): Clausura con elipse 15x15 seguida de rellenado de huecos (imfillhole, PI_U6_Morfologico_p2_Reconstruccion.py) y apertura 31x31.",
            "• Etapa 5 (Separación Moneda vs. Dado por Factor de Forma - U7 y Factor de Forma.txt): Filtro por área mínima dinámica (0.14% de HxW ~ 14,529 px^2) y cálculo de Fp = Área/Perímetro^2 y rho = 4*pi*Fp.",
            "  Un círculo perfecto cumple Fp = 1/(4*pi) = 0.0796 (1/Fp = 12.57, rho = 1.00) y un cuadrado cumple Fp = 1/16 = 0.0625 (1/Fp = 16.00, rho = 0.785). Empleando el umbral de cátedra",
            "  RHO_TH = 0.80 (clasifico_celulas.py), el sistema separa con 100% de exactitud las 17 Monedas (rho en [0.8056, 0.8904]) de los 2 Dados (rho = 0.7742 y rho = 0.6590).",
        ]
        for p in pasos_p1a:
            y = bloque_texto(ax1, 6.2, y, p, width=114, fontsize=7.85, line_spacing=1.36)

        insertar_imagen(
            fig1,
            os.path.join(dir_p1, "p1_puntoA_histograma_y_umbrales.png"),
            [0.048, 0.058, 0.904, 0.430],
            "Figura 1.1 (Punto A): Histograma logarítmico de grad_u8, barrido dinámico de k y comparación visual entre umbral menor (k=0.2), óptimo (k=0.9) y mayor (k=2.5).",
            ax1, 7.6
        )

        pdf.savefig(fig1)
        plt.close(fig1)

        # =====================================================================
        # PÁGINA 2: PROBLEMA 1 — PUNTO A (ETAPAS DE SEGMENTACIÓN EN 2 COLS x 4 FILAS)
        # =====================================================================
        fig1_etapas, ax1_etapas = nueva_pagina("Problema 1: Etapas de Segmentación (Punto A)", 2, 7)
        y_et = 132.5
        y_et = subtitulo_caja(ax1_etapas, y_et, "1.2.1. VISUALIZACIÓN DETALLADA DE LAS 8 ETAPAS DE SEGMENTACIÓN (2 COLUMNAS x 4 FILAS)")
        insertar_imagen(
            fig1_etapas,
            os.path.join(dir_p1, "p1_puntoA_etapas_segmentacion.png"),
            [0.08, 0.075, 0.84, 0.825],
            "Figura 1.2 (Punto A): Evolución paso a paso del preprocesamiento, gradiente de Sobel, umbralado dinámico, morfología (imfillhole) y segmentación de las 17 monedas y 2 dados.",
            ax1_etapas, 7.6
        )
        pdf.savefig(fig1_etapas)
        plt.close(fig1_etapas)

        # =====================================================================
        # PÁGINA 3: PROBLEMA 1 — PUNTO B (CLASIFICACIÓN DE MONEDAS Y TABLA)
        # =====================================================================
        fig2, ax2 = nueva_pagina("Problema 1: Clasificación de Monedas (Punto B)", 3, 7)

        y = 132.5
        y = subtitulo_caja(ax2, y, "1.3. PUNTO B: CLASIFICACIÓN GRANULOMÉTRICA Y CONTEO AUTOMÁTICO DE MONEDAS")

        y = bloque_texto(
            ax2, 5.2, y,
            "Una vez aisladas las 17 regiones circulares (rho > 0.80), se aborda la clasificación automática de los tres tipos "
            "de monedas presentes en la escena: 10 Centavos, 1 Peso (bimetálica) y 50 Centavos. Dado que la cámara se encuentra "
            "a distancia fija del plano de apoyo, el descriptor regional de Área (A = cv2.contourArea, Unidad 7) es directamente "
            "proporcional al cuadrado del diámetro físico de cada denominación monetaria.",
            width=112, fontsize=8.1, line_spacing=1.45
        )
        y -= 0.4
        y = bloque_texto(
            ax2, 5.2, y,
            "Para determinar las fronteras de decisión sin fijar umbrales manuales arbitrarios, el algoritmo ordena las áreas de "
            "las 17 monedas de menor a mayor y detecta los dos mayores saltos inter-clase consecutivos (criterio de máxima "
            "separación en 1D), ubicando automáticamente los umbrales óptimos en U1 = 79,744 px^2 y U2 = 102,381 px^2:",
            width=112, fontsize=8.1, line_spacing=1.45
        )
        y -= 0.3
        items_b = [
            "• Clase 1 — Monedas de 10 Centavos (Área < 79,744 px^2): 9 unidades detectadas (rango experimental: [62,612 - 67,606] px^2) -> Subtotal: $0.90.",
            "• Clase 2 — Monedas de 1 Peso (79,744 <= Área < 102,381 px^2): 5 unidades detectadas (rango experimental: [91,882 - 97,154] px^2) -> Subtotal: $5.00.",
            "• Clase 3 — Monedas de 50 Centavos (Área >= 102,381 px^2): 3 unidades detectadas (rango experimental: [107,607 - 110,049] px^2) -> Subtotal: $1.50.",
            "• Resultado Global del Conteo Automático (Punto B): 17 Monedas clasificadas sin solapamiento entre clases, acumulando un monto total de $7.40.",
        ]
        for it in items_b:
            y = bloque_texto(ax2, 6.2, y, it, width=114, fontsize=7.9, line_spacing=1.38, bold=("Resultado Global" in it))

        insertar_imagen(
            fig2,
            os.path.join(dir_p1, "p1_puntoB_clasificacion_monedas.png"),
            [0.048, 0.445, 0.904, 0.315],
            "Figura 1.3 (Punto B): Izquierda: Mapa de clasificación y conteo por denominación. Derecha: Espacio de descriptores Área vs. Circularidad (rho).",
            ax2, 61.8
        )

        # Tabla 1.1: Mediciones completas de los 19 objetos
        ax2.text(
            50.0, 58.6,
            "Tabla 1.1: Descriptores geométricos (Área, Perímetro, Factor de Forma Fp, 1/Fp y Circularidad rho) de los 19 objetos segmentados",
            fontsize=6.9, fontweight="bold", color="#0f172a", ha="center"
        )
        headers_t1 = ["ID", "Tipo (Punto A)", "Clase / Valor (Puntos B-C)", "Área [px²]", "Perímetro [px]", "Fp = A/P²", "1 / Fp", "rho = 4πFp", "Centroide (x, y)"]
        rows_t1 = [
            ["1", "Moneda", "10 Centavos ($0.10)", "62,612.0", "988.3", "0.0641", "15.60", "0.8056", "(1697.2, 1679.2)"],
            ["2", "Moneda", "10 Centavos ($0.10)", "63,301.0", "968.5", "0.0675", "14.82", "0.8481", "(1989.0, 1350.2)"],
            ["3", "Moneda", "10 Centavos ($0.10)", "63,851.5", "956.2", "0.0698", "14.32", "0.8775", "(875.3, 1045.3)"],
            ["4", "Moneda", "10 Centavos ($0.10)", "64,122.5", "951.3", "0.0709", "14.11", "0.8904", "(1073.5, 1881.5)"],
            ["5", "Moneda", "10 Centavos ($0.10)", "64,149.0", "966.8", "0.0686", "14.57", "0.8624", "(1615.1, 677.2)"],
            ["6", "Moneda", "10 Centavos ($0.10)", "64,637.0", "978.6", "0.0675", "14.82", "0.8482", "(2061.7, 1901.5)"],
            ["7", "Moneda", "10 Centavos ($0.10)", "64,990.0", "970.6", "0.0690", "14.50", "0.8669", "(3086.1, 803.5)"],
            ["8", "Moneda", "10 Centavos ($0.10)", "66,270.5", "973.6", "0.0699", "14.30", "0.8786", "(3371.0, 1324.3)"],
            ["9", "Dado (#2)", "Cara Superior = 3", "66,616.5", "1127.0", "0.0524", "19.07", "0.6590", "(2856.5, 1159.2)"],
            ["10", "Moneda", "10 Centavos ($0.10)", "67,606.0", "986.6", "0.0695", "14.40", "0.8727", "(2996.7, 1719.5)"],
            ["11", "Dado (#1)", "Cara Superior = 5", "74,461.5", "1099.4", "0.0616", "16.23", "0.7742", "(1047.1, 490.9)"],
            ["12", "Moneda", "1 Peso ($1.00)", "91,881.5", "1153.4", "0.0691", "14.48", "0.8679", "(1544.3, 2111.6)"],
            ["13", "Moneda", "1 Peso ($1.00)", "93,405.5", "1168.9", "0.0684", "14.63", "0.8591", "(2071.7, 1019.2)"],
            ["14", "Moneda", "1 Peso ($1.00)", "93,630.0", "1192.8", "0.0658", "15.20", "0.8270", "(635.0, 1333.1)"],
            ["15", "Moneda", "1 Peso ($1.00)", "95,154.5", "1182.2", "0.0681", "14.69", "0.8556", "(2549.5, 910.9)"],
            ["16", "Moneda", "1 Peso ($1.00)", "97,154.5", "1190.1", "0.0686", "14.58", "0.8620", "(2803.8, 2087.6)"],
            ["17", "Moneda", "50 Centavos ($0.50)", "107,607.0", "1243.9", "0.0695", "14.38", "0.8739", "(1373.2, 1179.1)"],
            ["18", "Moneda", "50 Centavos ($0.50)", "109,558.5", "1270.4", "0.0679", "14.73", "0.8530", "(2411.5, 1580.7)"],
            ["19", "Moneda", "50 Centavos ($0.50)", "110,049.0", "1260.5", "0.0693", "14.44", "0.8703", "(2166.9, 567.4)"],
        ]
        dibujar_tabla(ax2, 4.8, 57.5, [4.5, 10.5, 19.0, 10.5, 11.0, 8.5, 7.5, 8.5, 10.4], headers_t1, rows_t1, row_h=2.45, font_size=6.8)

        pdf.savefig(fig2)
        plt.close(fig2)

        # =====================================================================
        # PÁGINA 3: PROBLEMA 1 (PUNTO C) Y PROBLEMA 2 (PLANTEO Y PUNTO A)
        # =====================================================================
        fig3, ax3 = nueva_pagina("Problema 1 (Punto C) y Problema 2 (Punto A)", 4, 7)

        y = 132.5
        y = subtitulo_caja(ax3, y, "1.4. PUNTO C: DETERMINACIÓN DEL VALOR DE LA CARA SUPERIOR DE CADA DADO")

        y = bloque_texto(
            ax3, 5.2, y,
            "Para determinar automáticamente el valor representado únicamente por la cara superior de cada dado y descartar los "
            "puntos visibles en las caras laterales inclinadas, cada región clasificada como Dado en el Punto A (rho <= 0.80) se "
            "recorta según su caja contenedora y se procesa mediante cuatro etapas morfológicas y geométricas (Unidades 3, 6 y 7):",
            width=112, fontsize=8.1, line_spacing=1.45
        )
        pasos_p1c = [
            "1. Umbralado Inverso (T = 90, THRESH_BINARY_INV): Activa los puntos negros del dado y el fondo oscuro exterior que rodea al cubo blanco.",
            "2. Limpieza de Borde por Reconstrucción Morfológica (imclearborder, Unidad 6): Reconstruye desde el marco del recorte y elimina exactamente el fondo exterior.",
            "3. Clausura (Elipse 9x9) + Apertura (Elipse 5x5): Rellena los brillos especulares dentro de las cavidades negras de los puntos y suprime ruido aislado.",
            "4. Discriminación Cara Superior vs. Caras Laterales por Factor de Forma (rho = 4*pi*A/P^2): La cara superior es paralela al plano de la cámara, por lo que sus puntos",
            "   proyectan círculos casi perfectos (rho en [0.871, 0.900] > 0.80, área ~ 1950 px^2). En cambio, las caras laterales forman un ángulo oblicuo y sus puntos se proyectan",
            "   como elipses fuertemente elongadas (rho en [0.550, 0.590] <= 0.80, área ~ 630 px^2). El mismo umbral RHO_TH = 0.80 contabiliza exactamente:",
            "   • Dado #1 (izquierda): 5 puntos circulares en cara superior (y descarta los 3 puntos elípticos laterales) -> Valor = 5.",
            "   • Dado #2 (derecha):   3 puntos circulares en cara superior (0 puntos laterales desprendidos del borde)    -> Valor = 3.   Suma Total de Dados = 8.",
        ]
        for p in pasos_p1c:
            y = bloque_texto(ax3, 6.2, y, p, width=114, fontsize=7.85, line_spacing=1.36, bold=("Dado #" in p))

        insertar_imagen(
            fig3,
            os.path.join(dir_p1, "p1_puntoC_analisis_dados.png"),
            [0.048, 0.420, 0.904, 0.285],
            "Figura 1.4 (Punto C): Etapas de aislamiento (umbralado inverso, imclearborder, clausura+apertura) y filtrado por Factor de Forma de la cara superior de cada dado.",
            ax3, 59.0
        )

        y = 55.5
        y = subtitulo_caja(ax3, y, "2. PROBLEMA 2: DETECCIÓN AUTOMÁTICA DE PATENTES Y SEGMENTACIÓN DE CARACTERES")

        y = bloque_texto(
            ax3, 5.2, y,
            "2.1. Análisis de la Problemática: El conjunto de prueba ('img01.png' a 'img12.png', 640 x 360 px) comprende 12 escenas "
            "vehiculares reales tomadas con variaciones pronunciadas de distancia (caracteres desde 11 px hasta 21 px de altura), "
            "ángulos de guiñada e inclinación en el plano, iluminación diurna no uniforme, reflejos solares sobre paragolpes cromados, "
            "texturas repetitivas de adoquines y rejillas de radiador, y el marco blanco perimetral propio de las patentes argentinas.",
            width=112, fontsize=8.1, line_spacing=1.45
        )
        y -= 0.4
        y = bloque_texto(
            ax3, 5.2, y,
            "2.2. Punto A — Detección Automática y Segmentación de la Patente ('problema2_deteccion_patentes.py'): Las patentes del "
            "conjunto de prueba presentan 6 caracteres alfanuméricos blancos sobre fondo negro. Para resaltar simultáneamente los "
            "trazos brillantes y suprimir las grandes estructuras claras del vehículo y la calzada, se aplica la Transformada Top-Hat "
            "en niveles de gris (Unidad 6, PI_U6_Morfologico_p4_Grises.py):",
            width=112, fontsize=8.1, line_spacing=1.45
        )
        y = bloque_texto(
            ax3, 12.0, y,
            "I_tophat(x, y) = I(x, y) - (I o B)(x, y),   con Elemento Estructurante Rectangular B de 21 x 9 píxeles.",
            width=105, fontsize=8.1, line_spacing=1.45, bold=True, color="#1e3a8a"
        )
        y = bloque_texto(
            ax3, 5.2, y,
            "El ancho del EE (21 px) supera el grosor de cualquier trazo alfanumérico (<= 13 px), por lo que la apertura morfológica "
            "(I o B) elimina los caracteres y estima el fondo local; al restarla de la imagen original, los 6 caracteres emergen con "
            "alto contraste. A continuación, se evalúan las componentes conectadas (cv2.connectedComponentsWithStats) bajo las 3 reglas "
            "geométricas de 'info_patentes.txt' (Unidad 7):",
            width=112, fontsize=8.1, line_spacing=1.42
        )
        reglas_p2 = [
            "• Regla 1 (Relación de Aspecto Individual): Cada candidato debe verificar Alto/Ancho en [1.25, 3.5] (entorno de 1.5 - 3.0), altura H en [9, 30] px y Área >= 15 px^2.",
            "• Regla 2 (Agrupamiento en Tripletes): 3 candidatos forman un grupo válido si sus alturas difieren en <= 3.5 px, están alineados (|dy| <= 0.35*H) y equiespaciados (0.40*H <= dx <= 1.15*H).",
            "• Regla 3 (Apareamiento de 2 Grupos de 3 Caracteres): Dos tripletes colineales de igual escala separados por el espacio central de la chapa (1.25*dx_intra <= dx_central <= 2.80*dx_intra)",
            "  definen la patente. La constelación de máximo puntaje (Suma de Áreas - 10*std(H)) determina la caja contenedora más un margen proporcional (0.45*H horiz., 0.35*H vert.).",
        ]
        for r in reglas_p2:
            y = bloque_texto(ax3, 6.2, y, r, width=114, fontsize=7.8, line_spacing=1.35)

        pdf.savefig(fig3)
        plt.close(fig3)

        # =====================================================================
        # PÁGINA 4: FIGURAS DEL PUNTO A (PIPELINE DETALLADO Y MOSAICO 12 IMÁGENES)
        # =====================================================================
        fig4, ax4 = nueva_pagina("Problema 2: Detección de Patentes (Punto A)", 5, 7)

        y = 132.5
        y = subtitulo_caja(ax4, y, "2.2.1. ETAPAS INTERMEDIAS DE PROCESAMIENTO Y DETECCIÓN EN LAS 12 IMÁGENES (PUNTO A)")

        insertar_imagen(
            fig4,
            os.path.join(dir_p2, "p2_etapas_detalle_img01.png"),
            [0.045, 0.615, 0.910, 0.305],
            "Figura 2.1: Ilustración completa de las 8 etapas del pipeline (Puntos A y B) sobre img01.png [Patente BUF 817].",
            ax4, 86.2
        )

        insertar_imagen(
            fig4,
            os.path.join(dir_p2, "p2_puntoA_deteccion_12_patentes.png"),
            [0.045, 0.055, 0.910, 0.525],
            "Figura 2.2 (Punto A): Detección automática (caja verde) y recorte ampliado (esquina sup. izq.) en las 12 imágenes de prueba (12/12 = 100% de efectividad).",
            ax4, 7.4
        )

        pdf.savefig(fig4)
        plt.close(fig4)

        # =====================================================================
        # PÁGINA 5: PROBLEMA 2 — PUNTO B (SEGMENTACIÓN DE CARACTERES INDIVIDUALES)
        # =====================================================================
        fig5, ax5 = nueva_pagina("Problema 2: Segmentación de Caracteres (Punto B)", 6, 7)

        y = 132.5
        y = subtitulo_caja(ax5, y, "2.3. PUNTO B: ALGORITMO DE SEGMENTACIÓN DE CARACTERES INDIVIDUALES")

        y = bloque_texto(
            ax5, 5.2, y,
            "El Punto B exige segmentar los caracteres individuales tomando como entrada exclusivamente la región de la patente "
            "detectada y recortada en el Punto A. El principal desafío técnico dentro del recorte de la patente argentina es su "
            "marco blanco perimetral: al binarizar el recorte mediante un umbral global simple (como Otsu directo), el marco blanco "
            "superior e inferior frecuentemente queda conectado por puentes de 1 píxel con los extremos de las letras o dígitos "
            "(como ocurre en img01, img03, img08 e img11), fusionando varios caracteres en una única componente conectada.",
            width=112, fontsize=8.1, line_spacing=1.45
        )
        y -= 0.4
        y = bloque_texto(
            ax5, 5.2, y,
            "Para resolver este problema utilizando estrictamente las herramientas de la Unidad 6, el algoritmo implementado en "
            "la función 'segmentar_caracteres_patente(recorte_bgr)' ejecuta cuatro etapas sobre el recorte:",
            width=112, fontsize=8.1, line_spacing=1.45
        )
        pasos_p2b = [
            "1. Realce Top-Hat Local (EE Rectangular 21x9): Compensa gradientes de sombra sobre la chapa y atenúa la respuesta del marco continuo frente a los trazos delgados.",
            "2. Binarización + Limpieza de Bordes por Reconstrucción Morfológica (imclearborder, Unidad 6): Dado que el marco perimetral blanco toca los bordes exteriores del recorte",
            "   de la patente mientras que los 6 caracteres están contenidos en el interior, la reconstrucción geodésica desde el borde (imreconstruct) elimina el marco blanco completo.",
            "3. Selección Guiada por Consistencia Geométrica (Unidad 7): Se selecciona el umbral sobre el Top-Hat local que produce exactamente 6 componentes conectadas internas con",
            "   relación Alto/Ancho en [1.25, 3.5], altura entre el 35% y el 92% del alto del recorte y dispersión máxima de alturas max(H) - min(H) <= 4 px.",
            "4. Extracción y Ordenamiento Izquierda-Derecha: Las 6 regiones conexas se ordenan por coordenada x creciente (C1, C2, C3 = Letras; C4, C5, C6 = Dígitos), logrando segmentar",
            "   correctamente los 72 caracteres individuales de las 12 patentes (tasa de éxito: 72/72 = 100%).",
        ]
        for p in pasos_p2b:
            y = bloque_texto(ax5, 6.2, y, p, width=114, fontsize=7.85, line_spacing=1.36)

        insertar_imagen(
            fig5,
            os.path.join(dir_p2, "p2_puntoB_segmentacion_caracteres_12_patentes.png"),
            [0.045, 0.055, 0.910, 0.615],
            "Figura 2.3 (Punto B): Segmentación individual de los 6 caracteres (C1..C6) en las 12 patentes tras aplicar Top-Hat local + imclearborder (72/72 caracteres).",
            ax5, 7.4
        )

        pdf.savefig(fig5)
        plt.close(fig5)

        # =====================================================================
        # PÁGINA 6: TABLA CUANTITATIVA DEL PROBLEMA 2, CASO INCLINADO Y CONCLUSIONES
        # =====================================================================
        fig6, ax6 = nueva_pagina("Resultados Cuantitativos y Conclusiones", 7, 7)

        y = 132.5
        y = subtitulo_caja(ax6, y, "2.4. REPORTE CUANTITATIVO DE LAS 12 PATENTES Y CASO CON INCLINACIÓN (img04.png)")

        headers_t2 = [
            "Imagen", "Patente", "BBox Punto A (x, y, w, h)", "T_A",
            "Cand->Trip", "T_B", "Dimensiones C1..C6 (w×h px)", "Alturas [px]", "Estado"
        ]
        rows_t2 = [
            ["img01.png", "BUF 817", "(333, 140, 77×33)", "140", "7 -> 2", "110", "8×16, 8×15, 7×16, 8×16, 7×16, 8×15", "[16,15,16,16,16,15]", "OK (6/6)"],
            ["img02.png", "GXQ 798", "(320, 196, 73×35)", "70", "8 -> 3", "70", "7×16, 7×15, 11×15, 7×15, 7×14, 8×15", "[16,15,15,15,14,15]", "OK (6/6)"],
            ["img03.png", "DMH 024", "(233, 174, 82×41)", "85", "10 -> 2", "85", "10×16, 10×17, 10×16, 9×16, 9×17, 10×16", "[16,17,16,16,17,16]", "OK (6/6)"],
            ["img04.png", "LHE 986", "(312, 176, 84×42)", "105", "7 -> 2", "105", "10×16, 10×18, 10×18, 10×17, 9×16, 9×16", "[16,18,18,17,16,16]", "OK (6/6)"],
            ["img05.png", "FJY 344", "(250, 108, 88×44)", "55", "10 -> 2", "55", "9×19, 9×19, 8×19, 10×19, 9×18, 8×18", "[19,19,19,19,18,18]", "OK (6/6)"],
            ["img06.png", "LEH 992", "(279, 174, 108×46)", "45", "11 -> 2", "45", "13×21, 13×21, 13×21, 12×20, 12×21, 12×21", "[21,21,21,20,21,21]", "OK (6/6)"],
            ["img07.png", "LOD 123", "(254, 163, 68×24)", "70", "9 -> 2", "55", "7×14, 8×14, 8×13, 8×13, 7×13, 7×13", "[14,14,13,13,13,13]", "OK (6/6)"],
            ["img08.png", "JMO 337", "(185, 203, 77×32)", "110", "9 -> 2", "110", "8×14, 9×15, 8×15, 8×14, 8×15, 8×14", "[14,15,15,14,15,14]", "OK (6/6)"],
            ["img09.png", "FFD 765", "(360, 232, 89×30)", "65", "9 -> 2", "65", "8×16, 9×16, 9×17, 8×17, 9×18, 9×17", "[16,16,17,17,18,17]", "OK (6/6)"],
            ["img10.png", "DOA 322", "(292, 254, 79×30)", "105", "9 -> 2", "105", "8×15, 8×15, 7×16, 7×15, 9×16, 7×16", "[15,15,16,15,16,16]", "OK (6/6)"],
            ["img11.png", "GMQ 569", "(255, 206, 69×28)", "125", "10 -> 2", "125", "7×12, 8×12, 7×11, 7×11, 7×11, 7×11", "[12,12,11,11,11,11]", "OK (6/6)"],
            ["img12.png", "BQX 387", "(325, 191, 54×24)", "60", "12 -> 2", "60", "6×12, 6×11, 5×11, 6×11, 5×11, 5×11", "[12,11,11,11,11,11]", "OK (6/6)"],
        ]
        y_after_t2 = dibujar_tabla(
            ax6, 4.8, 127.5,
            [7.2, 6.8, 13.8, 5.5, 7.6, 5.5, 25.8, 11.4, 6.8],
            headers_t2, rows_t2, row_h=2.35, font_size=6.4
        )

        insertar_imagen(
            fig6,
            os.path.join(dir_p2, "p2_etapas_detalle_img04.png"),
            [0.045, 0.345, 0.910, 0.305],
            "Figura 2.4: Etapas de detección y segmentación sobre img04.png [LHE 986], evidenciando robustez ante perspectiva oblicua y poste de oclusión cercano.",
            ax6, 48.0
        )

        y = 44.2
        y = subtitulo_caja(ax6, y, "3. CONCLUSIONES GENERALES DEL TRABAJO PRÁCTICO N° 2")

        conclusiones = [
            "1. Invarianza frente a iluminación no uniforme mediante operadores diferenciales y morfológicos (Unidades 3 y 6): Tanto en 'monedas.jpg' como en las 12 escenas",
            "   vehiculares, la umbralización global directa sobre niveles de gris fracasa debido a los gradientes de iluminación y reflejos solares. En el Problema 1, la magnitud",
            "   del Gradiente de Sobel anuló la rampa suave del fondo y permitió recuperar los contornos completos; en el Problema 2, la Transformada Top-Hat con EE rectangular 21x9",
            "   actuó como un filtro pasa-altos morfológico sintonizado a la escala de los trazos alfanuméricos.",
            "2. Eficacia de la Reconstrucción Morfológica Binaria (Unidad 6): Los algoritmos geodésicos 'imfillhole' e 'imclearborder' resultaron determinantes en ambos problemas.",
            "   'imfillhole' convirtió los contornos cerrados del gradiente en máscaras sólidas sin alterar su perímetro externo, mientras que 'imclearborder' permitió eliminar",
            "   tanto el fondo exterior en los recortes de los dados (Problema 1.C) como el marco blanco perimetral de las patentes argentinas (Problema 2.B).",
            "3. Poder discriminante del Factor de Forma y de las relaciones estructurales (Unidad 7): El descriptor adimensional rho = 4*pi*A/P^2 con un único umbral teórico",
            "   (rho = 0.80) resolvió dos problemas geométricos distintos en el Problema 1: separar círculos (monedas) de cuadrados (dados) en el plano general, y separar",
            "   proyecciones circulares (puntos de la cara superior) de proyecciones elípticas escorzadas (puntos de las caras laterales) dentro de cada dado. Asimismo, la",
            "   combinación de relación de aspecto y agrupamiento en dos tripletes colineales alcanzó 100% de exactitud en las 12 patentes y en sus 72 caracteres individuales.",
        ]
        for c in conclusiones:
            y = bloque_texto(ax6, 5.4, y, c, width=113, fontsize=7.85, line_spacing=1.36, bold=c.startswith(("1.", "2.", "3.")))

        pdf.savefig(fig6)
        plt.close(fig6)

    print(f"[OK] Informe PDF generado exitosamente en: {out_pdf}")


if __name__ == "__main__":
    ruta_salida = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Informe_TP2_PDI.pdf")
    generar_pdf(ruta_salida)
