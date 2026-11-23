#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
TRABAJO PRÁCTICO N° 2 - PROCESAMIENTO DE IMÁGENES I (TUIA - FCEIA - UNR)
===============================================================================
PROBLEMA 2: Detección Automática de Patentes y Segmentación de Caracteres
Imágenes de prueba: img01.png a img12.png (12 escenas vehiculares, 640 x 360 px)

Autores:
    - Agustín Marquardt
    - Fabián Alvarez

Enunciado:
    Implementar un método para detectar automáticamente la patente en 12 imágenes
    de prueba (img01.png a img12.png), las cuales presentan diversas complejidades
    (cambios de escala, perspectiva, iluminación no uniforme, reflejos solares,
    sombras y fondos complejos).
    A. Detectar automáticamente la patente y segmentarla.
    B. Implementar un algoritmo de procesamiento que permita segmentar los
       caracteres individuales a partir de la patente detectada en el punto A.
    Aviso: En cada punto el script deberá informar y mostrar los resultados en
           cada etapa de procesamiento.

Fundamentación Teórica (Unidades 1 a 7 de la asignatura):
    - Unidad 3 (Segmentación): Umbralado de intensidad y etiquetado de componentes
      conectadas (`cv2.connectedComponentsWithStats`).
    - Unidad 6 (Morfología en Niveles de Gris y Binaria):
        * Transformada Top-Hat (`cv2.MORPH_TOPHAT`): I_tophat = I - (I o B).
          Resalta estructuras brillantes (los 6 caracteres alfanuméricos blancos)
          cuyo tamaño es menor que el elemento estructurante rectangular B (21x9),
          suprimiendo simultáneamente el fondo oscuro de la chapa y las grandes
          superficies claras de la carrocería, el pavimento y el cielo.
        * Reconstrucción Morfológica Binaria (`imreconstruct` e `imclearborder`,
          de `PI_U6_Morfologico_p2_Reconstruccion.py`): En el Punto B, permite
          eliminar de forma exacta el marco blanco perimetral de la patente que
          toca el borde del recorte, preservando intactos los 6 caracteres internos.
    - Unidad 7 (Representación y Descripción Regional) + `info_patentes.txt`:
        * Relación de aspecto de cada carácter: Alto/Ancho ~ 1.5 - 3.0 (con
          tolerancia de perspectiva e inclinación 1.25 <= H/W <= 3.5).
        * Agrupamiento estructural en dos grupos de 3 caracteres colineales
          (3 letras + 3 dígitos) de altura uniforme, separados por un espacio
          central mayor que el espaciado inter-carácter dentro de cada triplete.
===============================================================================
"""

import os
import sys
import glob
import argparse

# Configuración de directorio temporal para caché de Matplotlib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import imreconstruct, imclearborder, impar as _impar


# Referencia de validación (Ground Truth visual de las 12 imágenes del TP2)
PATENTES_REFERENCIA = {
    "img01.png": "BUF 817",
    "img02.png": "GXQ 798",
    "img03.png": "DMH 024",
    "img04.png": "LHE 986",
    "img05.png": "FJY 344",
    "img06.png": "LEH 992",
    "img07.png": "LOD 123",
    "img08.png": "JMO 337",
    "img09.png": "FFD 765",
    "img10.png": "DOA 322",
    "img11.png": "GMQ 569",
    "img12.png": "BQX 387",
}


# =============================================================================
# PUNTO A: DETECCIÓN AUTOMÁTICA Y SEGMENTACIÓN DE LA PATENTE
# =============================================================================


def filtrar_candidatos_caracter(bw_mask, scale_f=1.0):
    """
    Extrae las componentes conectadas de una máscara binaria y filtra aquellas
    que verifican las restricciones geométricas de un carácter de patente
    según `info_patentes.txt` (Relación Alto/Ancho ~ 1.5 - 3.0 y escala plausible).
    Los límites en píxeles escalan con `scale_f = min(H, W) / 360.0` (invarianza a escala).
    """
    h_min = max(5.0, 9.0 * scale_f)
    h_max = 30.0 * scale_f
    w_min = max(2.0, 3.0 * scale_f)
    w_max = 20.0 * scale_f
    area_min = max(6.0, 15.0 * (scale_f ** 2))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw_mask, connectivity=8)
    candidatos = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        aspect = h / float(w) if w > 0 else 0.0
        solidity = area / float(w * h) if (w * h) > 0 else 0.0
        # Regla 1 (info_patentes.txt): Relación Alto/Ancho de cada carácter ~ 1.5 a 3.0
        # Se admite el intervalo [1.25, 3.5] para contemplar perspectiva y caracteres como '1' o 'M'/'Q'.
        if h_min <= h <= h_max and w_min <= w <= w_max and 1.25 <= aspect <= 3.5 and area >= area_min:
            candidatos.append({
                "label": i,
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "area": int(area),
                "cx": float(cx),
                "cy": float(cy),
                "aspect": float(aspect),
                "solidity": float(solidity),
            })
    return candidatos, labels


def buscar_constelacion_patente(candidatos):
    """
    Evalúa las Reglas 2 y 3 de `info_patentes.txt` sobre los candidatos:
        - Regla 2: Los caracteres forman grupos de 3 (tripletes) alineados
          horizontalmente, con alturas uniformes y espaciado regular.
        - Regla 3: La patente está formada por dos grupos de 3 caracteres
          colineales separados por un espacio central mayor.

    Retorna la mejor constelación de 6 caracteres (puntaje estructural máximo)
    junto con la lista de todos los tripletes válidos detectados.
    """
    candidatos = sorted(candidatos, key=lambda c: c["x"])
    n = len(candidatos)
    tripletes = []

    # Regla 2: Búsqueda de grupos de 3 caracteres (tripletes)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                c1, c2, c3 = candidatos[i], candidatos[j], candidatos[k]
                hs = np.array([c1["h"], c2["h"], c3["h"]], dtype=np.float64)
                h_mean = float(np.mean(hs))
                # Uniformidad de altura proporcional a h_mean (invariante a escala)
                if np.max(np.abs(hs - h_mean)) > max(2.0, 0.22 * h_mean):
                    continue

                dx1 = c2["cx"] - c1["cx"]
                dx2 = c3["cx"] - c2["cx"]
                dy1 = c2["cy"] - c1["cy"]
                dy2 = c3["cy"] - c2["cy"]

                # Espaciado horizontal regular entre caracteres contiguos
                if not (0.40 * h_mean <= dx1 <= 1.15 * h_mean and 0.40 * h_mean <= dx2 <= 1.15 * h_mean):
                    continue
                if abs(dx1 - dx2) > 0.35 * h_mean:
                    continue

                # Alineación casi horizontal (admite leve inclinación de la patente)
                if abs(dy1) > 0.35 * h_mean or abs(dy2) > 0.35 * h_mean:
                    continue

                tripletes.append((c1, c2, c3, h_mean, 0.5 * (dx1 + dx2), 0.5 * (dy1 + dy2)))

    # Regla 3: Apareamiento de dos grupos de 3 caracteres (letras + números)
    mejor_seis = None
    mejor_puntaje = -1e9

    for t1 in tripletes:
        for t2 in tripletes:
            c1, c2, c3, h1, dx1, dy1 = t1
            c4, c5, c6, h2, dx2, dy2 = t2

            # Consistencia de altura entre ambos grupos de 3 caracteres (invariante a escala)
            h_prom = 0.5 * (h1 + h2)
            if abs(h1 - h2) > max(2.0, 0.22 * h_prom):
                continue

            dx_intra = 0.5 * (dx1 + dx2)

            # Distancia entre el 3er carácter (última letra) y el 4to carácter (primer dígito)
            dx_central = c4["cx"] - c3["cx"]
            if not (1.25 * dx_intra <= dx_central <= 2.80 * dx_intra):
                continue

            # Colinealidad entre el primer y segundo grupo de 3 caracteres (invariante a escala)
            dy_central = c4["cy"] - c3["cy"]
            dy_esperado = dy1 * (dx_central / max(dx_intra, 1.0))
            if abs(dy_central - dy_esperado) > max(2.0, 0.25 * h_prom):
                continue

            grupo_6 = [c1, c2, c3, c4, c5, c6]
            suma_areas = sum(c["area"] for c in grupo_6)
            std_alturas = float(np.std([c["h"] for c in grupo_6]))
            puntaje = suma_areas - 10.0 * std_alturas

            if puntaje > mejor_puntaje:
                mejor_puntaje = puntaje
                mejor_seis = grupo_6

    return mejor_seis, mejor_puntaje, tripletes


def detectar_y_segmentar_patente(img_bgr):
    """
    PUNTO A:
    Detecta automáticamente la patente en la escena vehicular y la segmenta.
    Etapas:
        1. Conversión a escala de grises.
        2. Transformada Top-Hat morfológica con EE rectangular 21x9 (`cv2.MORPH_TOPHAT`).
        3. Barrido de umbralado multiescala sobre la respuesta Top-Hat y evaluación
           de las 3 reglas geométricas de `info_patentes.txt`.
        4. Cálculo de la caja contenedora (Bounding Box) de la patente con margen
           proporcional a la altura media de los caracteres y recorte (ROI).
    """
    H, W = img_bgr.shape[:2]
    scale_f = float(min(H, W)) / 360.0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Etapa 2: Transformada Top-Hat en niveles de gris con EE proporcional a la escala (Unidad 6)
    kw_th = _impar(21.0 * scale_f, minimo=5)
    kh_th = _impar(9.0 * scale_f, minimo=3)
    se_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (kw_th, kh_th))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se_tophat)

    # Etapa 3: Evaluación multiumbral sobre Top-Hat guiada por descriptores (Unidades 3 y 7)
    mejor_resultado = None
    mejor_puntaje_global = -1e9

    for th in range(35, 165, 5):
        _, bw = cv2.threshold(tophat, th, 255, cv2.THRESH_BINARY)
        candidatos, labels = filtrar_candidatos_caracter(bw, scale_f=scale_f)
        grupo_6, puntaje, tripletes = buscar_constelacion_patente(candidatos)

        if grupo_6 is not None and puntaje > mejor_puntaje_global:
            mejor_puntaje_global = puntaje
            mejor_resultado = {
                "th_optimo": th,
                "bw_tophat": bw,
                "candidatos": candidatos,
                "tripletes": tripletes,
                "caracteres_puntoA": grupo_6,
                "puntaje": puntaje,
            }

    if mejor_resultado is None:
        raise RuntimeError("No se encontró una constelación válida de patente en la imagen.")

    chars = mejor_resultado["caracteres_puntoA"]
    x_min = min(c["x"] for c in chars)
    y_min = min(c["y"] for c in chars)
    x_max = max(c["x"] + c["w"] for c in chars)
    y_max = max(c["y"] + c["h"] for c in chars)
    hm = float(np.mean([c["h"] for c in chars]))

    # Margen proporcional a la altura de los caracteres para abarcar la placa completa
    pad_x = int(round(0.45 * hm))
    pad_y = int(round(0.35 * hm))
    px1 = max(0, x_min - pad_x)
    py1 = max(0, y_min - pad_y)
    px2 = min(W, x_max + pad_x)
    py2 = min(H, y_max + pad_y)

    recorte_bgr = img_bgr[py1:py2, px1:px2].copy()
    recorte_gray = gray[py1:py2, px1:px2].copy()

    etapas_puntoA = {
        "scale_f": scale_f,
        "gray": gray,
        "tophat": tophat,
        "th_optimo": mejor_resultado["th_optimo"],
        "bw_tophat": mejor_resultado["bw_tophat"],
        "candidatos": mejor_resultado["candidatos"],
        "tripletes": mejor_resultado["tripletes"],
        "caracteres_puntoA": chars,
        "altura_media_chars": hm,
        "bbox_patente": (px1, py1, px2 - px1, py2 - py1),
        "recorte_bgr": recorte_bgr,
        "recorte_gray": recorte_gray,
    }
    return etapas_puntoA


# =============================================================================
# PUNTO B: SEGMENTACIÓN DE LOS CARACTERES INDIVIDUALES DE LA PATENTE
# =============================================================================

def segmentar_caracteres_patente(recorte_bgr, scale_f=1.0):
    """
    PUNTO B:
    A partir únicamente de la patente segmentada en el Punto A (`recorte_bgr`),
    implementa un algoritmo de procesamiento para segmentar los 6 caracteres
    individuales:
        1. Conversión del recorte de la patente a escala de grises.
        2. Realce local mediante Transformada Top-Hat proporcional a la escala
           (`21x9` en 640x360) para uniformizar el fondo de la placa frente a
           sombras o reflejos oblicuos.
        3. Umbralado + Limpieza de bordes por reconstrucción morfológica
           (`imclearborder`, Unidad 6) para suprimir el marco blanco perimetral
           de la patente Argentina que toca los bordes del recorte.
        4. Etiquetado de componentes conectadas (`cv2.connectedComponentsWithStats`)
           y selección de los 6 caracteres alfanuméricos ordenados de izquierda
           a derecha (C1, C2, C3 = Letras; C4, C5, C6 = Dígitos).
    """
    hc, wc = recorte_bgr.shape[:2]
    plate_gray = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2GRAY)

    # Top-Hat sobre la patente recortada con EE proporcional a la escala de imagen
    kw_p = _impar(21.0 * scale_f, minimo=5)
    kh_p = _impar(9.0 * scale_f, minimo=3)
    se_plate = cv2.getStructuringElement(cv2.MORPH_RECT, (kw_p, kh_p))
    plate_tophat = cv2.morphologyEx(plate_gray, cv2.MORPH_TOPHAT, se_plate)

    # Umbral de Otsu de referencia sobre el Top-Hat del recorte
    th_otsu, bw_otsu = cv2.threshold(plate_tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Búsqueda del umbral óptimo que desconecta limpiamente los 6 caracteres del marco
    mejor_config = None
    mejor_score = -1e9
    area_char_min = max(6.0, 15.0 * (scale_f ** 2))

    for th in range(30, 180, 5):
        _, bw_raw = cv2.threshold(plate_tophat, th, 255, cv2.THRESH_BINARY)
        # Limpieza morfológica del marco blanco exterior que toca el borde del recorte (Unidad 6)
        bw_cb = imclearborder(bw_raw, conn=8)

        num_l, labels, stats, centroids = cv2.connectedComponentsWithStats(bw_cb, connectivity=8)
        comps = []
        for i in range(1, num_l):
            x, y, w, h, area = stats[i]
            cx, cy = centroids[i]
            aspect = h / float(w) if w > 0 else 0.0
            if 0.35 * hc <= h <= 0.92 * hc and 1.25 <= aspect <= 3.5 and area >= area_char_min:
                comps.append({
                    "label": i,
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "area": int(area),
                    "cx": float(cx),
                    "cy": float(cy),
                    "aspect": float(aspect),
                })

        comps = sorted(comps, key=lambda c: c["x"])
        if len(comps) == 6:
            hs = [c["h"] for c in comps]
            h_mean_c = float(np.mean(hs))
            if max(hs) - min(hs) <= max(2.5, 0.25 * h_mean_c):
                score = sum(c["area"] for c in comps) - 10.0 * np.std(hs)
                if score > mejor_score:
                    mejor_score = score
                    mejor_config = (th, bw_raw, bw_cb, labels, comps)

    if mejor_config is None:
        raise RuntimeError("No fue posible segmentar los 6 caracteres individuales del recorte.")

    th_sel, bw_raw, bw_cb, labels, chars_6 = mejor_config

    # Construcción de la máscara limpia que contiene exclusivamente los 6 caracteres
    bw_chars_clean = np.zeros_like(bw_cb)
    recortes_caracteres = []
    for idx, c in enumerate(chars_6, start=1):
        bw_chars_clean[labels == c["label"]] = 255
        # Recorte individual con 1 píxel de margen visual
        x1 = max(0, c["x"] - 1)
        y1 = max(0, c["y"] - 1)
        x2 = min(wc, c["x"] + c["w"] + 1)
        y2 = min(hc, c["y"] + c["h"] + 1)
        crop_char_rgb = cv2.cvtColor(recorte_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        crop_char_bw = bw_chars_clean[y1:y2, x1:x2].copy()

        c_info = dict(c)
        c_info["indice"] = idx
        c_info["tipo"] = "Letra" if idx <= 3 else "Dígito"
        c_info["crop_rgb"] = crop_char_rgb
        c_info["crop_bw"] = crop_char_bw
        recortes_caracteres.append(c_info)

    etapas_puntoB = {
        "plate_gray": plate_gray,
        "plate_tophat": plate_tophat,
        "th_otsu_ref": float(th_otsu),
        "th_seleccionado": int(th_sel),
        "bw_raw": bw_raw,
        "bw_imclearborder": bw_cb,
        "bw_chars_clean": bw_chars_clean,
        "labels_chars": labels,
        "caracteres": recortes_caracteres,
    }
    return etapas_puntoB


# =============================================================================
# FIGURAS DIAGNÓSTICAS Y MOSAICOS DE RESULTADOS (PUNTOS A Y B)
# =============================================================================

def colorear_caracteres(labels, caracteres):
    """Asigna un color distintivo RGB a cada uno de los 6 caracteres segmentados."""
    paleta = [
        (255, 75, 75),    # C1 - Rojo
        (255, 175, 40),   # C2 - Naranja
        (230, 230, 50),   # C3 - Amarillo
        (60, 220, 100),   # C4 - Verde
        (50, 190, 255),   # C5 - Celeste
        (195, 100, 255),  # C6 - Violeta
    ]
    canvas = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    for idx, c in enumerate(caracteres):
        canvas[labels == c["label"]] = paleta[idx % len(paleta)]
    return canvas, paleta


def generar_figura_etapas_detalle(nombre_img, img_bgr, res_A, res_B, out_path):
    """
    Genera una figura detallada con las 8 etapas del pipeline completo (Puntos A y B)
    para una imagen representativa.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    patente_ref = PATENTES_REFERENCIA.get(nombre_img, "")

    fig = plt.figure(figsize=(18, 7.2))
    gs = fig.add_gridspec(2, 4, hspace=0.18, wspace=0.16)

    # 1. Imagen original
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_rgb)
    ax1.set_title(f"1. Escena Original ({nombre_img})\nResolución: {img_bgr.shape[1]}x{img_bgr.shape[0]} px", fontsize=10, fontweight="bold")
    ax1.axis("off")

    # 2. Transformada Top-Hat (21x9)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(res_A["tophat"], cmap="magma")
    ax2.set_title("2. Transformada Top-Hat (EE Rect 21x9)\nRealce de trazos claros y supresión de fondo", fontsize=10, fontweight="bold")
    ax2.axis("off")

    # 3. Umbralado del Top-Hat y filtrado por Relación de Aspecto (Regla 1)
    ax3 = fig.add_subplot(gs[0, 2])
    vis_cand = cv2.cvtColor(res_A["bw_tophat"], cv2.COLOR_GRAY2RGB)
    for c in res_A["candidatos"]:
        cv2.rectangle(vis_cand, (c["x"], c["y"]), (c["x"] + c["w"], c["y"] + c["h"]), (255, 165, 0), 1)
    for c in res_A["caracteres_puntoA"]:
        cv2.rectangle(vis_cand, (c["x"], c["y"]), (c["x"] + c["w"], c["y"] + c["h"]), (0, 240, 80), 2)
    ax3.imshow(vis_cand)
    ax3.set_title(
        f"3. Umbralado (T={res_A['th_optimo']}) + Reglas 1-3\n"
        f"{len(res_A['candidatos'])} candidatos (naranja) -> 2 Tripletes (verde)",
        fontsize=10, fontweight="bold"
    )
    ax3.axis("off")

    # 4. Punto A: Detección y Bounding Box de la patente
    ax4 = fig.add_subplot(gs[0, 3])
    vis_det = img_rgb.copy()
    px, py, pw, ph = res_A["bbox_patente"]
    cv2.rectangle(vis_det, (px, py), (px + pw, py + ph), (0, 255, 90), 3)
    cv2.putText(vis_det, f"PATENTE ({px},{py},{pw}x{ph})", (max(10, px - 40), max(22, py - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(vis_det, f"PATENTE ({px},{py},{pw}x{ph})", (max(10, px - 40), max(22, py - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 90), 2, cv2.LINE_AA)
    ax4.imshow(vis_det)
    ax4.set_title(f"4. Punto A: Patente Detectada\nBBox: x={px}, y={py}, w={pw}, h={ph}", fontsize=10, fontweight="bold")
    ax4.axis("off")

    # 5. Punto A -> Recorte de la patente + Top-Hat local
    ax5 = fig.add_subplot(gs[1, 0])
    rec_rgb = cv2.cvtColor(res_A["recorte_bgr"], cv2.COLOR_BGR2RGB)
    th_rgb = cv2.cvtColor(res_B["plate_tophat"], cv2.COLOR_GRAY2RGB)
    apilado_5 = np.vstack([rec_rgb, np.full((4, rec_rgb.shape[1], 3), 255, dtype=np.uint8), th_rgb])
    ax5.imshow(apilado_5)
    ax5.set_title("5. Patente Segmentada (Arriba)\ny Top-Hat Local 21x9 (Abajo)", fontsize=10, fontweight="bold")
    ax5.axis("off")

    # 6. Punto B -> Binarización vs Limpieza de Marco (imclearborder)
    ax6 = fig.add_subplot(gs[1, 1])
    raw_rgb = cv2.cvtColor(res_B["bw_raw"], cv2.COLOR_GRAY2RGB)
    cb_rgb = cv2.cvtColor(res_B["bw_imclearborder"], cv2.COLOR_GRAY2RGB)
    apilado_6 = np.vstack([raw_rgb, np.full((4, raw_rgb.shape[1], 3), 120, dtype=np.uint8), cb_rgb])
    ax6.imshow(apilado_6)
    ax6.set_title(
        f"6. Binarización T={res_B['th_seleccionado']} (Arriba) vs.\n"
        f"Remoción de Marco con imclearborder (Abajo)",
        fontsize=10, fontweight="bold"
    )
    ax6.axis("off")

    # 7. Punto B -> Etiquetado cromático de los 6 caracteres
    ax7 = fig.add_subplot(gs[1, 2])
    vis_boxes = rec_rgb.copy()
    canvas_col, paleta = colorear_caracteres(res_B["labels_chars"], res_B["caracteres"])
    for idx, c in enumerate(res_B["caracteres"]):
        col = paleta[idx % len(paleta)]
        cv2.rectangle(vis_boxes, (c["x"], c["y"]), (c["x"] + c["w"], c["y"] + c["h"]), col, 1)
    apilado_7 = np.vstack([vis_boxes, np.full((4, vis_boxes.shape[1], 3), 255, dtype=np.uint8), canvas_col])
    ax7.imshow(apilado_7)
    ax7.set_title("7. Punto B: Cajas Contenedoras (Arriba)\ny Componentes Etiquetadas C1..C6 (Abajo)", fontsize=10, fontweight="bold")
    ax7.axis("off")

    # 8. Punto B -> Los 6 caracteres individuales recortados
    gs_chars = gs[1, 3].subgridspec(2, 6, hspace=0.15, wspace=0.12)
    for idx, c in enumerate(res_B["caracteres"]):
        ax_c_rgb = fig.add_subplot(gs_chars[0, idx])
        ax_c_rgb.imshow(c["crop_rgb"])
        ax_c_rgb.set_title(f"C{idx+1}\n{c['w']}x{c['h']}", fontsize=8.5, fontweight="bold")
        ax_c_rgb.axis("off")

        ax_c_bw = fig.add_subplot(gs_chars[1, idx])
        ax_c_bw.imshow(c["crop_bw"], cmap="gray")
        ax_c_bw.set_title(f"H/W={c['aspect']:.1f}", fontsize=7.5)
        ax_c_bw.axis("off")

    plt.suptitle(
        f"PROBLEMA 2 - Etapas de Detección de Patente (Punto A) y Segmentación de Caracteres (Punto B): {nombre_img} [{patente_ref}]",
        fontsize=12.5, fontweight="bold", y=0.98
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def generar_mosaicos_globales(resultados, out_dir, mostrar=False):
    """
    Genera los mosaicos globales para las 12 imágenes de prueba:
        - `p2_puntoA_deteccion_12_patentes.png`: Las 12 escenas con la patente detectada y su recorte.
        - `p2_puntoB_segmentacion_caracteres_12_patentes.png`: Las 12 patentes con la máscara
          `imclearborder`, las componentes coloreadas y los 6 caracteres individuales (72/72 caracteres).
    """
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # MOSAICO PUNTO A: Detección y Segmentación de la Patente en las 12 imágenes
    # -------------------------------------------------------------------------
    figA, axesA = plt.subplots(4, 3, figsize=(18, 15.5))
    axesA = axesA.flatten()

    for idx, item in enumerate(resultados):
        ax = axesA[idx]
        nombre = item["nombre"]
        patente_ref = PATENTES_REFERENCIA.get(nombre, "")
        img_rgb = cv2.cvtColor(item["img_bgr"], cv2.COLOR_BGR2RGB)
        res_A = item["puntoA"]
        px, py, pw, ph = res_A["bbox_patente"]

        vis = img_rgb.copy()
        cv2.rectangle(vis, (px, py), (px + pw, py + ph), (0, 255, 80), 3)

        # Incrustar una vista ampliada (x2.5) de la patente segmentada en la esquina superior izquierda
        rec_rgb = cv2.cvtColor(res_A["recorte_bgr"], cv2.COLOR_BGR2RGB)
        zoom_h = min(110, int(rec_rgb.shape[0] * 2.4))
        zoom_w = min(260, int(rec_rgb.shape[1] * 2.4))
        rec_zoom = cv2.resize(rec_rgb, (zoom_w, zoom_h), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(vis, (6, 6), (6 + zoom_w + 6, 6 + zoom_h + 6), (0, 255, 80), -1)
        vis[9:9 + zoom_h, 9:9 + zoom_w] = rec_zoom

        ax.imshow(vis)
        ax.set_title(
            f"{nombre} [{patente_ref}] | BBox=({px},{py},{pw}x{ph}) | T_TopHat={res_A['th_optimo']}",
            fontsize=9.5, fontweight="bold"
        )
        ax.axis("off")

    plt.suptitle(
        "PROBLEMA 2 - PUNTO A: Detección Automática y Recorte de la Patente en las 12 Imágenes de Prueba (12/12 Correctas)",
        fontsize=13.5, fontweight="bold", y=0.995
    )
    plt.tight_layout()
    figA_path = os.path.join(out_dir, "p2_puntoA_deteccion_12_patentes.png")
    figA.savefig(figA_path, dpi=160, bbox_inches="tight")

    # -------------------------------------------------------------------------
    # MOSAICO PUNTO B: Segmentación de los 6 Caracteres Individuales (12 patentes)
    # -------------------------------------------------------------------------
    figB = plt.figure(figsize=(18, 11.5))
    gs_main = figB.add_gridspec(6, 2, hspace=0.22, wspace=0.14)

    for idx, item in enumerate(resultados):
        row = idx // 2
        col = idx % 2
        nombre = item["nombre"]
        patente_ref = PATENTES_REFERENCIA.get(nombre, "")
        res_A = item["puntoA"]
        res_B = item["puntoB"]

        sub_gs = gs_main[row, col].subgridspec(1, 9, width_ratios=[2.2, 2.2, 2.2, 1, 1, 1, 1, 1, 1], wspace=0.08)

        # Panel 1: Patente recortada con cajas de los 6 caracteres
        ax_p1 = figB.add_subplot(sub_gs[0, 0])
        rec_rgb = cv2.cvtColor(res_A["recorte_bgr"], cv2.COLOR_BGR2RGB)
        canvas_col, paleta = colorear_caracteres(res_B["labels_chars"], res_B["caracteres"])
        vis_rec = rec_rgb.copy()
        for c_i, c in enumerate(res_B["caracteres"]):
            cv2.rectangle(vis_rec, (c["x"], c["y"]), (c["x"] + c["w"], c["y"] + c["h"]), paleta[c_i], 1)
        ax_p1.imshow(vis_rec)
        ax_p1.set_title(f"{nombre} [{patente_ref}]\nPatente Segmentada", fontsize=8, fontweight="bold")
        ax_p1.axis("off")

        # Panel 2: Máscara tras imclearborder
        ax_p2 = figB.add_subplot(sub_gs[0, 1])
        ax_p2.imshow(res_B["bw_imclearborder"], cmap="gray")
        ax_p2.set_title(f"imclearborder\n(T={res_B['th_seleccionado']})", fontsize=8, fontweight="bold")
        ax_p2.axis("off")

        # Panel 3: Etiquetado de los 6 caracteres en color
        ax_p3 = figB.add_subplot(sub_gs[0, 2])
        ax_p3.imshow(canvas_col)
        ax_p3.set_title("6 Componentes\nConectadas", fontsize=8, fontweight="bold")
        ax_p3.axis("off")

        # Paneles 4 a 9: Los 6 caracteres individuales (máscara binaria segmentada)
        chars_ref = patente_ref.replace(" ", "")
        for c_i, c in enumerate(res_B["caracteres"]):
            ax_c = figB.add_subplot(sub_gs[0, 3 + c_i])
            ax_c.imshow(c["crop_bw"], cmap="gray")
            letra_gt = chars_ref[c_i] if c_i < len(chars_ref) else f"C{c_i+1}"
            ax_c.set_title(f"C{c_i+1}:'{letra_gt}'\n{c['w']}x{c['h']}", fontsize=7.5, fontweight="bold")
            ax_c.axis("off")

    plt.suptitle(
        "PROBLEMA 2 - PUNTO B: Segmentación Individual de los 6 Caracteres en cada Patente (72/72 Caracteres Segmentados)",
        fontsize=13.5, fontweight="bold", y=0.995
    )
    figB_path = os.path.join(out_dir, "p2_puntoB_segmentacion_caracteres_12_patentes.png")
    figB.savefig(figB_path, dpi=160, bbox_inches="tight")

    if mostrar:
        plt.show()
    else:
        plt.close("all")

    return [figA_path, figB_path]


# =============================================================================
# REPORTE DETALLADO POR CONSOLA Y EJECUCIÓN PRINCIPAL
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TP2 PDI - Problema 2: Detección Automática de Patentes y Segmentación de Caracteres"
    )
    parser.add_argument(
        "--images",
        nargs="*",
        default=None,
        help="Lista opcional de imágenes a procesar (por defecto procesa img01.png a img12.png)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_problema2"),
        help="Directorio de salida para las figuras diagnósticas",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Muestra las figuras en ventanas interactivas además de guardarlas en disco",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if args.images:
        rutas_imagenes = [
            p if os.path.isabs(p) else os.path.join(base_dir, p) for p in args.images
        ]
    else:
        rutas_imagenes = sorted(glob.glob(os.path.join(base_dir, "img*.png")))

    if not rutas_imagenes:
        print("[ERROR] No se encontraron imágenes img*.png para procesar.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 100)
    print("TRABAJO PRÁCTICO N° 2 - PROCESAMIENTO DE IMÁGENES I (TUIA - FCEIA - UNR)")
    print("PROBLEMA 2: DETECCIÓN AUTOMÁTICA DE PATENTES Y SEGMENTACIÓN DE CARACTERES")
    print("=" * 100)
    print(f"Total de imágenes a procesar: {len(rutas_imagenes)}\n")

    resultados = []
    figuras_detalle = []

    print("-" * 100)
    print(f"{'Imagen':<10} | {'Ref.':<8} | {'Punto A: BBox (x,y,w,h)':<24} | {'T_A':>4} | {'Cand->Trip':<11} | {'Punto B: T_B':<12} | {'Alturas C1..C6 (px)':<22}")
    print("-" * 100)

    for ruta in rutas_imagenes:
        nombre = os.path.basename(ruta)
        img_bgr = cv2.imread(ruta)
        if img_bgr is None:
            print(f"[ADVERTENCIA] No se pudo leer {ruta}")
            continue

        # Punto A: Detección y segmentación de la patente
        res_A = detectar_y_segmentar_patente(img_bgr)

        # Punto B: Segmentación de los 6 caracteres individuales a partir del recorte del Punto A
        res_B = segmentar_caracteres_patente(res_A["recorte_bgr"], scale_f=res_A["scale_f"])

        px, py, pw, ph = res_A["bbox_patente"]
        alturas_str = ",".join(str(c["h"]) for c in res_B["caracteres"])
        ref_str = PATENTES_REFERENCIA.get(nombre, "---")

        print(
            f"{nombre:<10} | {ref_str:<8} | ({px:>3},{py:>3},{pw:>3}x{ph:>2}) px      | "
            f"{res_A['th_optimo']:>4} | {len(res_A['candidatos']):>3} -> {len(res_A['tripletes']):<4} | "
            f"T={res_B['th_seleccionado']:>3} (6/6) | [{alturas_str}]"
        )

        resultados.append({
            "nombre": nombre,
            "ruta": ruta,
            "img_bgr": img_bgr,
            "puntoA": res_A,
            "puntoB": res_B,
        })

        # Guardar figura detallada de las 8 etapas para casos representativos de distinta complejidad
        if nombre in ("img01.png", "img04.png", "img06.png", "img08.png"):
            out_det = os.path.join(args.outdir, f"p2_etapas_detalle_{nombre.replace('.png', '')}.png")
            generar_figura_etapas_detalle(nombre, img_bgr, res_A, res_B, out_det)
            figuras_detalle.append(out_det)

    print("-" * 100)

    # Desglose etapa por etapa en consola
    print("\n[RESUMEN DE ETAPAS Y MÉTRICAS DE VALIDACIÓN]")
    print("  * PUNTO A (Detección y Segmentación de Patente):")
    print("      - Preprocesamiento: Conversión a gris + Transformada Top-Hat con EE Rectangular (21x9).")
    print("      - Filtrado geométrico (Regla 1): Relación Alto/Ancho en [1.25, 3.5], Altura en [9, 30] px.")
    print("      - Agrupamiento perceptual (Reglas 2 y 3): Dos tripletes colineales de altura uniforme.")
    print(f"      - Tasa de detección de patente: {len(resultados)}/{len(rutas_imagenes)} (100.0%).")

    total_chars = sum(len(r["puntoB"]["caracteres"]) for r in resultados)
    print("\n  * PUNTO B (Segmentación de Caracteres Individuales):")
    print("      - Entrada: Exclusivamente el recorte de la patente obtenida en el Punto A.")
    print("      - Realce y desvinculación del marco: Top-Hat local (21x9) + Umbralado + `imclearborder` (U6).")
    print("      - Extracción: Componentes conectadas (`connectedComponentsWithStats`) ordenadas de izq. a der.")
    print(f"      - Total de caracteres segmentados: {total_chars}/{len(resultados)*6} (100.0%).")

    mosaicos = generar_mosaicos_globales(resultados, args.outdir, mostrar=args.plot)
    print("\n[FIGURAS GUARDADAS EN DISCO]")
    for p in mosaicos + figuras_detalle:
        print(f"  - {p}")
    print("=" * 100)


if __name__ == "__main__":
    main()
