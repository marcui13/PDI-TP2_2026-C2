"""
Trabajo Práctico N° 2 - Procesamiento de Imágenes I (TUIA - FCEIA - UNR)
Año 2026 - 2° Semestre

Problema 1: Detección y clasificación de monedas y dados
========================================================
A. Procesar la imagen 'monedas.jpg' de manera de segmentar las monedas y los dados
   de manera automática sobre un fondo de intensidad no uniforme.
B. Clasificar los distintos tipos de monedas y realizar un conteo automático.
C. Determinar el número del valor que representa la cara superior de cada dado y
   realizar un conteo automático.

Metodología (Alineada estrictamente a las Unidades 1 a 7 de la asignatura):
- Unidad 1 y 2: Conversión a escala de grises y filtrado espacial de mediana (cv2.medianBlur)
  para suprimir ruido de textura del fondo preservando la nitidez de los bordes.
- Unidad 3: Detección de bordes mediante magnitud exacta del Gradiente de Sobel (cv2.Sobel),
  cuya derivada espacial anula la componente de baja frecuencia del fondo no uniforme,
  seguido de umbralado automático estadístico (media + desvío estándar).
- Unidad 6: Operaciones morfológicas binarias (Clausura cv2.MORPH_CLOSE y Apertura cv2.MORPH_OPEN
  con elemento estructurante elíptico) y algoritmos de Reconstrucción Morfológica (imreconstruct,
  rellenado de huecos imfillhole y limpieza de bordes imclearborder).
- Unidad 7 y Material de Cátedra ('Factor de Forma.txt' y 'clasifico_celulas.py'):
  Extracción de componentes conexas (cv2.connectedComponentsWithStats) y contornos (cv2.findContours),
  cálculo del Factor de Forma Fp = Área / Perímetro^2 y circularidad normalizada rho = 4*pi*Fp
  (umbral RHO_TH = 0.80) para separar monedas (circulares) de dados (poligonales), clasificación
  granulométrica por área para los 3 tipos de monedas, y discriminación geométrica por Factor de
  Forma para contar exclusivamente los puntos (pips) circulares de la cara superior de cada dado,
  descartando los puntos elípticos escorzados de las caras laterales.
"""

import os
import sys
import argparse

# Configuración de caché para Matplotlib antes de importar pyplot
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

# Auto-detección del entorno virtual local (.venv) si no está activo
for p in [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv/lib/python3.9/site-packages"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.venv/lib/python3.9/site-packages"),
]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, os.path.abspath(p))

import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import imreconstruct, imfillhole, imclearborder, impar as _impar


# =============================================================================
# CONSTANTES GLOBALES
# =============================================================================

RHO_TH = 0.80  # Umbral de circularidad normalizada rho = 4*pi*Fp (clasifico_celulas.py)


# =============================================================================
# PUNTO A: SEGMENTACIÓN AUTOMÁTICA DE MONEDAS Y DADOS
# =============================================================================


def calcular_parametros_escala(h_img, w_img):
    """
    Calcula los tamaños de filtro, margen de borde y elementos estructurantes
    proporcionales a la resolución de la imagen (invarianza a escala).
    """
    dim_min = float(min(h_img, w_img))
    k_med = _impar(dim_min * 0.0041, minimo=3)        # 11 en 3871x2681
    pad_borde = max(3, int(round(dim_min * 0.0037)))  # 10 px en 3871x2681
    k_close = _impar(dim_min * 0.0056, minimo=3)      # 15 en 3871x2681
    k_open = _impar(dim_min * 0.0115, minimo=5)       # 31 en 3871x2681
    return k_med, pad_borde, k_close, k_open


def calcular_gradiente_sobel(img_gray, ksize=3):
    """
    Calcula la magnitud exacta del gradiente de Sobel en uint8
    para suprimir variaciones suaves de iluminación del fondo.
    """
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=ksize)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return np.uint8(np.clip(grad_mag, 0, 255))


def optimizar_umbral_gradiente(grad_u8, grad_mean, grad_std, se_close, se_open, area_min):
    """
    Realiza un barrido sobre k en [0.5, 1.5] para elegir el mayor factor k
    que maximiza la cantidad de componentes cerradas válidas tras clausura,
    rellenado de huecos (imfillhole) y apertura morfológica.
    """
    sweep_k = []
    best_count = -1
    k_opt = 0.9
    for k_cand in np.round(np.arange(0.5, 1.55, 0.1), 2):
        t_cand = float(np.round(grad_mean + k_cand * grad_std))
        _, bw_e = cv2.threshold(grad_u8, t_cand, 255, cv2.THRESH_BINARY)
        bw_c = cv2.morphologyEx(bw_e, cv2.MORPH_CLOSE, se_close)
        bw_f = imfillhole(bw_c)
        bw_m = cv2.morphologyEx(bw_f, cv2.MORPH_OPEN, se_open)
        n_lbl, _, st, _ = cv2.connectedComponentsWithStats(bw_m)
        valid_cnt = sum(1 for idx_lbl in range(1, n_lbl) if st[idx_lbl, cv2.CC_STAT_AREA] >= area_min)
        sweep_k.append({"k": float(k_cand), "th": t_cand, "valid_count": int(valid_cnt)})
        # En empate de máximo número de objetos, elegimos el mayor k (bordes más limpios sin abrir contornos)
        if valid_cnt >= best_count:
            best_count = valid_cnt
            k_opt = float(k_cand)
    return k_opt, sweep_k


def segmentar_objetos(img_bgr):
    """
    Ejecuta la tubería de preprocesamiento, detección de bordes por gradiente de Sobel
    y morfología matemática para segmentar todos los objetos sobre fondo no uniforme.
    """
    # 1. Conversión a escala de grises (Unidad 1 / Unidad 5)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape

    # Tamaños de filtro y EE proporcionales a la resolución (invarianza a escala)
    k_med, pad_borde, k_close, k_open = calcular_parametros_escala(h_img, w_img)

    # Demostración de por qué falla el umbralado global directo sobre fondo no uniforme
    th_otsu_global, bw_otsu_global = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 2. Filtrado espacial de mediana para eliminar ruido impulsivo y textura del fondo
    #    preservando los bordes de las monedas y dados (Unidad 2)
    gray_med = cv2.medianBlur(gray, k_med)

    # 3. Magnitud exacta del Gradiente de Sobel (Unidad 3: PDI_U3_Segmentacion.py)
    #    La derivada espacial suprime el gradiente suave de iluminación del fondo
    grad_u8 = calcular_gradiente_sobel(gray_med, ksize=3)

    # Anulamos artefactos de borde proporcionalmente para garantizar semilla limpia en (0, 0)
    grad_u8[:pad_borde, :] = 0
    grad_u8[-pad_borde:, :] = 0
    grad_u8[:, :pad_borde] = 0
    grad_u8[:, -pad_borde:] = 0

    # 4. Umbralado estadístico dinámico del gradiente: th_grad = round(mu + k_opt * sigma)
    #    Barrido sobre k en [0.5, 1.5] para elegir el mayor k que maximiza la cantidad
    #    de componentes cerradas válidas tras clausura + imfillhole + apertura.
    AREA_MIN = max(500.0, 0.0014 * float(h_img * w_img))  # ~0.14% del área total (~14,529 px^2 en 3871x2681)

    se_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    se_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))

    grad_mean = float(grad_u8.mean())
    grad_std = float(grad_u8.std())

    k_opt, sweep_k = optimizar_umbral_gradiente(
        grad_u8, grad_mean, grad_std, se_close, se_open, AREA_MIN
    )

    th_grad = float(np.round(grad_mean + k_opt * grad_std))
    _, bw_edges = cv2.threshold(grad_u8, th_grad, 255, cv2.THRESH_BINARY)

    # 5. Clausura morfológica para conectar micro-cortes en el perímetro (Unidad 6)
    bw_closed = cv2.morphologyEx(bw_edges, cv2.MORPH_CLOSE, se_close)

    # 6. Rellenado morfológico de huecos internos (Unidad 6: imfillhole)
    bw_filled = imfillhole(bw_closed)

    # 7. Apertura morfológica para suavizar contornos y eliminar ruido aislado (Unidad 6)
    bw_mask = cv2.morphologyEx(bw_filled, cv2.MORPH_OPEN, se_open)

    # 8. Extracción de componentes conexas y separación Monedas vs Dados por Factor de Forma
    #    Según 'Factor de Forma.txt' y 'clasifico_celulas.py' (Unidad 6 y 7):
    #    Fp = Área / Perímetro^2  (Círculo ideal: Fp = 1/(4*pi) = 0.0796, 1/Fp = 12.57)
    #    rho = 4 * pi * Fp        (Círculo ideal: rho = 1.0; Umbral RHO_TH = 0.80)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw_mask)

    objetos = []
    monedas = []
    dados = []

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < AREA_MIN:
            continue

        obj_mask = (labels == i).astype(np.uint8) * 255
        ext_contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(ext_contours, key=cv2.contourArea)

        area = float(cv2.contourArea(cnt))
        perimetro = float(cv2.arcLength(cnt, True))
        fp = area / (perimetro ** 2)
        inv_fp = 1.0 / fp
        rho = 4.0 * np.pi * fp
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = centroids[i]
        radio_eq = float(np.sqrt(area / np.pi))

        es_moneda = rho > RHO_TH
        tipo_obj = "Moneda" if es_moneda else "Dado"

        info = {
            "id_label": i,
            "tipo_objeto": tipo_obj,
            "es_moneda": es_moneda,
            "cnt": cnt,
            "mask": obj_mask,
            "area": area,
            "perimetro": perimetro,
            "fp": fp,
            "inv_fp": inv_fp,
            "rho": rho,
            "radio_eq": radio_eq,
            "bbox": (x, y, w, h),
            "centroid": (float(cx), float(cy)),
        }
        objetos.append(info)
        if es_moneda:
            monedas.append(info)
        else:
            dados.append(info)

    # Ordenamos por área ascendente para facilitar la lectura y análisis
    objetos.sort(key=lambda d: d["area"])
    monedas.sort(key=lambda d: d["area"])
    dados.sort(key=lambda d: d["bbox"][0])  # Dados de izquierda a derecha

    etapas = {
        "gray": gray,
        "bw_otsu_global": bw_otsu_global,
        "th_otsu_global": th_otsu_global,
        "gray_med": gray_med,
        "grad_u8": grad_u8,
        "grad_mean": grad_mean,
        "grad_std": grad_std,
        "k_med": k_med,
        "k_close": k_close,
        "k_open": k_open,
        "k_opt": k_opt,
        "sweep_k": sweep_k,
        "area_min": AREA_MIN,
        "th_grad": th_grad,
        "bw_edges": bw_edges,
        "bw_closed": bw_closed,
        "bw_filled": bw_filled,
        "bw_mask": bw_mask,
        "labels": labels,
        "num_labels": num_labels,
    }
    return objetos, monedas, dados, etapas


# =============================================================================
# PUNTO B: CLASIFICACIÓN DE LOS DISTINTOS TIPOS DE MONEDAS Y CONTEO AUTOMÁTICO
# =============================================================================

def clasificar_monedas(monedas):
    """
    Clasifica las monedas segmentadas en función del descriptor geométrico de Área
    (Unidad 7), el cual refleja el diámetro físico de cada denominación:
      - 10 Centavos (diámetro 19.0 mm): Área ~ 62,600 - 67,600 px^2 (< 80,000 px^2)
      - 1 Peso      (diámetro 23.0 mm): Área ~ 91,800 - 97,200 px^2 (80,000 - 102,000 px^2)
      - 50 Centavos (diámetro 25.2 mm): Área ~ 107,600 - 110,100 px^2 (> 102,000 px^2)
    """
    conteo = {
        "10 Centavos": 0,
        "1 Peso": 0,
        "50 Centavos": 0,
    }
    valor_unitario = {
        "10 Centavos": 0.10,
        "1 Peso": 1.00,
        "50 Centavos": 0.50,
    }

    # Cálculo dinámico de los umbrales de área (criterio de máxima separación en 1D):
    # Se ordenan las áreas de menor a mayor, se calculan los saltos consecutivos
    # y se ubican los umbrales en el punto medio de los dos mayores saltos inter-clase.
    areas_ord = sorted(m["area"] for m in monedas)
    if len(areas_ord) >= 3:
        diffs = np.diff(areas_ord)
        top2_idx = sorted(np.argsort(diffs)[-2:])
        i1, i2 = int(top2_idx[0]), int(top2_idx[1])
        UMBRAL_AREA_10C_1P = float((areas_ord[i1] + areas_ord[i1 + 1]) / 2.0)
        UMBRAL_AREA_1P_50C = float((areas_ord[i2] + areas_ord[i2 + 1]) / 2.0)
    else:
        UMBRAL_AREA_10C_1P = 80000.0
        UMBRAL_AREA_1P_50C = 102000.0

    monto_total = 0.0
    for idx, m in enumerate(monedas, 1):
        area = m["area"]
        if area < UMBRAL_AREA_10C_1P:
            clase = "10 Centavos"
        elif area < UMBRAL_AREA_1P_50C:
            clase = "1 Peso"
        else:
            clase = "50 Centavos"

        m["id_moneda"] = idx
        m["denominacion"] = clase
        m["valor_monetario"] = valor_unitario[clase]
        conteo[clase] += 1
        monto_total += valor_unitario[clase]

    resumen_monedas = {
        "conteo": conteo,
        "total_monedas": len(monedas),
        "monto_total": monto_total,
        "umbrales_area": (UMBRAL_AREA_10C_1P, UMBRAL_AREA_1P_50C),
    }
    return monedas, resumen_monedas


# =============================================================================
# PUNTO C: DETERMINACIÓN DE LA CARA SUPERIOR DE CADA DADO Y CONTEO AUTOMÁTICO
# =============================================================================

def analizar_dados(img_bgr, gray, dados):
    """
    Determina el valor de la cara superior de cada dado segmentado.
    Como los dados están fotografiados en perspectiva oblicua, se observan puntos (pips)
    tanto en la cara superior como en las caras laterales.
    
    Para contar exclusivamente los puntos de la cara superior:
    1. Se recorta la ROI del dado y se umbralizan las cavidades oscuras (puntos negros).
    2. Se aplica imclearborder (Unidad 6) para remover el fondo oscuro exterior al dado.
    3. Se aplica Clausura (9x9) + Apertura (5x5) para eliminar brillos especulares dentro
       de los puntos negros.
    4. Se evalúa el Factor de Forma Fp = Área / Perímetro^2 (rho = 4*pi*Fp) de cada punto:
       - Los puntos de la CARA SUPERIOR (paralela al plano de cámara) se proyectan como
         círculos perfectos: rho > 0.80 (Fp ~ 0.070, 1/Fp ~ 14.1) y Área ~ 1950 px^2.
       - Los puntos de las CARAS LATERALES (inclinadas por perspectiva) se proyectan como
         elipses alargadas: rho <= 0.80 (Fp ~ 0.045, 1/Fp ~ 22.0) y Área ~ 630 px^2.
    """
    suma_total_dados = 0
    for idx, d in enumerate(dados, 1):
        # Umbrales de área y kernels morfológicos dinámicos proporcionales al dado (invariantes a escala):
        # - Ruido residual: < 0.15% del área del dado (~100 px^2 en monedas.jpg)
        # - Pips de cara superior: ~2.6%-3.2% del área del dado vs laterales ~0.85% -> corte en 1.7% (~1130-1265 px^2)
        l_dado = float(np.sqrt(d["area"]))
        k_close_pip = _impar(l_dado * 0.034, minimo=3)  # 9 en monedas.jpg
        k_open_pip = _impar(l_dado * 0.019, minimo=3)   # 5 en monedas.jpg
        se_close_pip = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close_pip, k_close_pip))
        se_open_pip = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open_pip, k_open_pip))

        area_pip_min = max(20.0, 0.0015 * d["area"])
        area_pip_top_min = 0.017 * d["area"]

        x, y, w, h = d["bbox"]
        crop_rgb = cv2.cvtColor(img_bgr[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
        crop_gray = gray[y:y+h, x:x+w]

        # 1. Umbralado inverso para resaltar los puntos oscuros del dado
        _, pips_bw = cv2.threshold(crop_gray, 90, 255, cv2.THRESH_BINARY_INV)

        # 2. Eliminación del fondo exterior conectado al borde del recorte (Unidad 6)
        pips_cb = imclearborder(pips_bw)

        # 3. Filtrado morfológico (Clausura + Apertura) para consolidar cada punto
        pips_clean = cv2.morphologyEx(pips_cb, cv2.MORPH_CLOSE, se_close_pip)
        pips_clean = cv2.morphologyEx(pips_clean, cv2.MORPH_OPEN, se_open_pip)

        # 4. Detección de contornos de los puntos candidatos y clasificación por Factor de Forma
        cnts_pips, _ = cv2.findContours(pips_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pips_cara_superior = []
        pips_caras_laterales = []

        for c_pip in cnts_pips:
            a_pip = float(cv2.contourArea(c_pip))
            if a_pip < area_pip_min:
                continue
            p_pip = float(cv2.arcLength(c_pip, True))
            fp_pip = a_pip / (p_pip ** 2)
            inv_fp_pip = 1.0 / fp_pip
            rho_pip = 4.0 * np.pi * fp_pip

            M = cv2.moments(c_pip)
            cx_p = float(M["m10"] / M["m00"]) if M["m00"] > 0 else 0.0
            cy_p = float(M["m01"] / M["m00"]) if M["m00"] > 0 else 0.0

            es_cara_superior = (rho_pip > RHO_TH) and (a_pip > area_pip_top_min)
            pip_info = {
                "cnt": c_pip,
                "area": a_pip,
                "perimetro": p_pip,
                "fp": fp_pip,
                "inv_fp": inv_fp_pip,
                "rho": rho_pip,
                "centroid": (cx_p, cy_p),
                "es_cara_superior": es_cara_superior,
            }
            if es_cara_superior:
                pips_cara_superior.append(pip_info)
            else:
                pips_caras_laterales.append(pip_info)

        valor_dado = len(pips_cara_superior)
        suma_total_dados += valor_dado

        d["id_dado"] = idx
        d["valor_cara_superior"] = valor_dado
        d["pips_cara_superior"] = pips_cara_superior
        d["pips_caras_laterales"] = pips_caras_laterales
        d["crop_rgb"] = crop_rgb
        d["crop_gray"] = crop_gray
        d["pips_bw"] = pips_bw
        d["pips_cb"] = pips_cb
        d["pips_clean"] = pips_clean

    resumen_dados = {
        "total_dados": len(dados),
        "valores": [d["valor_cara_superior"] for d in dados],
        "suma_total": suma_total_dados,
    }
    return dados, resumen_dados


# =============================================================================
# GENERACIÓN DE FIGURAS DE ETAPAS INTERMEDIAS Y RESULTADOS
# =============================================================================

def generar_figuras_problema1(img_bgr, objetos, monedas, dados, etapas, resumen_monedas, resumen_dados, out_dir, mostrar=False):
    """
    Genera y guarda las figuras diagnósticas de cada etapa de procesamiento (Puntos A, B y C).
    """
    os.makedirs(out_dir, exist_ok=True)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # -------------------------------------------------------------------------
    # FIGURA 1 (PUNTO A): Tubería de Preprocesamiento y Segmentación Morfológica
    # (Disposición en 4 filas x 2 columnas para mayor tamaño y detalle visual)
    # -------------------------------------------------------------------------
    fig1, axes1 = plt.subplots(4, 2, figsize=(13, 18.5))
    axes1 = axes1.flatten()

    axes1[0].imshow(img_rgb)
    axes1[0].set_title("1. Imagen Original (monedas.jpg)\nFondo de intensidad no uniforme", fontsize=11, fontweight="bold")

    axes1[1].imshow(etapas["bw_otsu_global"], cmap="gray")
    axes1[1].set_title(f"2. Falla de Otsu Global Directo (T={etapas['th_otsu_global']:.0f})\nEl gradiente de iluminación satura la derecha", fontsize=11, fontweight="bold")

    axes1[2].imshow(etapas["gray_med"], cmap="gray")
    axes1[2].set_title(f"3. Escala de Grises + Filtro Mediana ({etapas['k_med']}x{etapas['k_med']})\nAtenuación de textura preservando bordes", fontsize=11, fontweight="bold")

    axes1[3].imshow(etapas["grad_u8"], cmap="gray")
    axes1[3].set_title("4. Magnitud del Gradiente de Sobel\nInvariante a la iluminación no uniforme", fontsize=11, fontweight="bold")

    axes1[4].imshow(etapas["bw_edges"], cmap="gray")
    axes1[4].set_title(f"5. Umbralado Dinámico del Gradiente (T={etapas['th_grad']:.0f}, k={etapas['k_opt']:.1f})\nBordes binarios de monedas y dados", fontsize=11, fontweight="bold")

    axes1[5].imshow(etapas["bw_closed"], cmap="gray")
    axes1[5].set_title(f"6. Clausura Morfológica (Elipse {etapas['k_close']}x{etapas['k_close']})\nCierre de brechas en contornos exteriores", fontsize=11, fontweight="bold")

    axes1[6].imshow(etapas["bw_mask"], cmap="gray")
    axes1[6].set_title(f"7. Rellenado (imfillhole) + Apertura ({etapas['k_open']}x{etapas['k_open']})\nMáscara binaria sólida de los 19 objetos", fontsize=11, fontweight="bold")

    # Visualización de Monedas vs Dados por Factor de Forma
    vis_seg = img_rgb.copy()
    for obj in objetos:
        color = (0, 220, 100) if obj["es_moneda"] else (255, 80, 40)
        cv2.drawContours(vis_seg, [obj["cnt"]], -1, color, 8)
        cx, cy = int(obj["centroid"][0]), int(obj["centroid"][1])
        etiqueta = f"M (rho={obj['rho']:.2f})" if obj["es_moneda"] else f"DADO (rho={obj['rho']:.2f})"
        cv2.putText(vis_seg, etiqueta, (cx - 110, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(vis_seg, etiqueta, (cx - 110, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    axes1[7].imshow(vis_seg)
    axes1[7].set_title(
        f"8. Punto A: Segmentación por Factor de Forma\n"
        f"{len(monedas)} Monedas (rho > {RHO_TH:.2f}) y {len(dados)} Dados (rho <= {RHO_TH:.2f})",
        fontsize=11, fontweight="bold"
    )

    for ax in axes1:
        ax.axis("off")

    plt.suptitle("PROBLEMA 1 - PUNTO A: Segmentación Automática de Monedas y Dados sobre Fondo No Uniforme", fontsize=13.5, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig1_path = os.path.join(out_dir, "p1_puntoA_etapas_segmentacion.png")
    fig1.savefig(fig1_path, dpi=160, bbox_inches="tight")

    # -------------------------------------------------------------------------
    # FIGURA 1B (PUNTO A): Histograma de grad_u8 y Sensibilidad del Umbral k
    # -------------------------------------------------------------------------
    grad_u8 = etapas["grad_u8"]
    grad_mean = etapas["grad_mean"]
    grad_std = etapas["grad_std"]
    k_opt = etapas["k_opt"]
    th_grad = etapas["th_grad"]
    area_min = etapas["area_min"]

    th_grad_otsu, _ = cv2.threshold(grad_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_grad_tri, _ = cv2.threshold(grad_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

    fig1b = plt.figure(figsize=(16, 9.5))
    gs1b = fig1b.add_gridspec(2, 6, height_ratios=[1.1, 1.0], hspace=0.35, wspace=0.25)

    # Panel 1: Histograma de grad_u8 (escala logarítmica para ver la cola)
    ax_hist = fig1b.add_subplot(gs1b[0, :3])
    hist_vals = cv2.calcHist([grad_u8], [0], None, [256], [0, 256]).flatten()
    bins_x = np.arange(101)
    ax_hist.bar(bins_x, hist_vals[:101], width=1.0, color="#64748b", edgecolor="none", alpha=0.85)
    ax_hist.set_yscale("log")
    ax_hist.axvline(grad_mean, color="#0284c7", linestyle=":", linewidth=2.0, label=f"Media μ = {grad_mean:.1f}")
    ax_hist.axvline(th_grad_tri, color="#f59e0b", linestyle="--", linewidth=1.8, label=f"Triangle (T={th_grad_tri:.0f}, ruido)")
    ax_hist.axvline(th_grad, color="#16a34a", linestyle="-", linewidth=2.5, label=f"Dinámico μ+{k_opt:.1f}σ (T={th_grad:.0f})")
    ax_hist.axvline(th_grad_otsu, color="#dc2626", linestyle="--", linewidth=1.8, label=f"Otsu sobre grad (T={th_grad_otsu:.0f}, corta bordes)")
    ax_hist.set_xlim(-1, 100)
    ax_hist.set_title("Histograma de grad_u8 (escala log) — Distribución asimétrica a la derecha", fontsize=10.5, fontweight="bold")
    ax_hist.set_xlabel("Magnitud del Gradiente de Sobel (grad_u8)", fontsize=9.5)
    ax_hist.set_ylabel("Cantidad de píxeles (log)", fontsize=9.5)
    ax_hist.grid(True, alpha=0.25, which="both")
    ax_hist.legend(fontsize=8.5, loc="upper right")

    # Panel 2: Barrido dinámico de k vs cantidad de objetos cerrados válidos
    ax_sweep = fig1b.add_subplot(gs1b[0, 3:])
    ks = [s["k"] for s in etapas["sweep_k"]]
    counts = [s["valid_count"] for s in etapas["sweep_k"]]
    ax_sweep.plot(ks, counts, "-o", color="#1e3a8a", linewidth=2.0, markersize=6)
    ax_sweep.axvline(k_opt, color="#16a34a", linestyle="--", linewidth=2.0, label=f"k óptimo seleccionado = {k_opt:.1f} (T={th_grad:.0f})")
    ax_sweep.scatter([k_opt], [max(counts)], color="#16a34a", s=110, zorder=5)
    ax_sweep.set_title("Cálculo Dinámico de k: Estabilidad Morfológica (Objetos Cerrados Válidos)", fontsize=10.5, fontweight="bold")
    ax_sweep.set_xlabel("Factor multiplicador k  (T = round(μ + k·σ))", fontsize=9.5)
    ax_sweep.set_ylabel("Componentes cerradas válidas", fontsize=9.5)
    ax_sweep.set_ylim(0, max(counts) + 4)
    ax_sweep.grid(True, alpha=0.25)
    ax_sweep.legend(fontsize=8.5, loc="lower left")

    # Paneles inferiores: Demostración visual de umbral menor, óptimo y mayor
    se_c_demo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (etapas["k_close"], etapas["k_close"]))
    se_o_demo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (etapas["k_open"], etapas["k_open"]))
    demos = [
        (0.2, "Umbral MENOR (k=0.2, T={t:.0f})\nRuido fusiona objetos ({n} válidos)", "#f59e0b"),
        (k_opt, f"Umbral ÓPTIMO (k={k_opt:.1f}, T={{t:.0f}})\nTodos cerrados y separados ({{n}} válidos)", "#16a34a"),
        (2.5, "Umbral MAYOR (k=2.5, T={t:.0f})\nContornos abiertos, imfillhole falla ({n} válidos)", "#dc2626"),
    ]
    for idx_d, (k_val, titulo_tpl, col_borde) in enumerate(demos):
        ax_d = fig1b.add_subplot(gs1b[1, idx_d * 2:(idx_d + 1) * 2])
        t_d = float(np.round(grad_mean + k_val * grad_std))
        _, bw_e_d = cv2.threshold(grad_u8, t_d, 255, cv2.THRESH_BINARY)
        bw_c_d = cv2.morphologyEx(bw_e_d, cv2.MORPH_CLOSE, se_c_demo)
        bw_f_d = imfillhole(bw_c_d)
        bw_m_d = cv2.morphologyEx(bw_f_d, cv2.MORPH_OPEN, se_o_demo)
        n_d, _, st_d, _ = cv2.connectedComponentsWithStats(bw_m_d)
        n_valid_d = sum(1 for ii in range(1, n_d) if st_d[ii, cv2.CC_STAT_AREA] >= area_min)
        ax_d.imshow(bw_m_d, cmap="gray")
        ax_d.set_title(titulo_tpl.format(t=t_d, n=n_valid_d), fontsize=10, fontweight="bold", color=col_borde)
        ax_d.axis("off")

    plt.suptitle("PROBLEMA 1 - PUNTO A: Análisis Estadístico de grad_u8 y Efecto del Umbral en la Segmentación", fontsize=13, fontweight="bold")
    fig1b_path = os.path.join(out_dir, "p1_puntoA_histograma_y_umbrales.png")
    fig1b.savefig(fig1b_path, dpi=160, bbox_inches="tight")

    # -------------------------------------------------------------------------
    # FIGURA 2 (PUNTO B): Clasificación y Conteo Automático de Monedas
    # -------------------------------------------------------------------------
    fig2 = plt.figure(figsize=(18, 8.5))
    gs = fig2.add_gridspec(1, 2, width_ratios=[1.35, 1.0])

    ax_img = fig2.add_subplot(gs[0])
    vis_monedas = img_rgb.copy()

    colores_clase = {
        "10 Centavos": (50, 180, 255),   # Celeste
        "1 Peso": (40, 220, 90),         # Verde
        "50 Centavos": (255, 185, 20),   # Dorado / Naranja
    }

    for m in monedas:
        clase = m["denominacion"]
        col = colores_clase[clase]
        cv2.drawContours(vis_monedas, [m["cnt"]], -1, col, 9)
        cx, cy = int(m["centroid"][0]), int(m["centroid"][1])
        cv2.circle(vis_monedas, (cx, cy), 10, col, -1)
        etiqueta_corta = "$0.10" if clase == "10 Centavos" else ("$1.00" if clase == "1 Peso" else "$0.50")
        cv2.putText(vis_monedas, f"#{m['id_moneda']} {etiqueta_corta}", (cx - 95, cy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.05, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(vis_monedas, f"#{m['id_moneda']} {etiqueta_corta}", (cx - 95, cy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.05, col, 2, cv2.LINE_AA)
        cv2.putText(vis_monedas, f"A={m['area']/1000:.1f}k", (cx - 80, cy + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(vis_monedas, f"A={m['area']/1000:.1f}k", (cx - 80, cy + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    ax_img.imshow(vis_monedas)
    c10 = resumen_monedas["conteo"]["10 Centavos"]
    c1p = resumen_monedas["conteo"]["1 Peso"]
    c50 = resumen_monedas["conteo"]["50 Centavos"]
    ax_img.set_title(
        f"Clasificación Espacial de las {resumen_monedas['total_monedas']} Monedas\n"
        f"10 Centavos: {c10} | 1 Peso: {c1p} | 50 Centavos: {c50} (Total: ${resumen_monedas['monto_total']:.2f})",
        fontsize=11, fontweight="bold"
    )
    ax_img.axis("off")

    # Gráfico de dispersión Área vs Circularidad (rho) con fronteras de decisión
    ax_plot = fig2.add_subplot(gs[1])
    for clase, col_rgb in colores_clase.items():
        sub = [m for m in monedas if m["denominacion"] == clase]
        ax_plot.scatter(
            [m["area"] / 1000.0 for m in sub],
            [m["rho"] for m in sub],
            s=110,
            color=np.array(col_rgb) / 255.0,
            edgecolors="black",
            linewidths=1.2,
            label=f"{clase} (n={len(sub)})",
            zorder=4,
        )
    ax_plot.scatter(
        [d["area"] / 1000.0 for d in dados],
        [d["rho"] for d in dados],
        s=140,
        marker="s",
        color=(1.0, 0.3, 0.2),
        edgecolors="black",
        linewidths=1.2,
        label=f"Dados (n={len(dados)})",
        zorder=4,
    )

    ax_plot.axhline(RHO_TH, color="crimson", linestyle="--", linewidth=1.8, label=f"Umbral Circularidad (rho = {RHO_TH:.2f})")
    u1, u2 = resumen_monedas["umbrales_area"]
    ax_plot.axvline(u1 / 1000.0, color="navy", linestyle=":", linewidth=1.6, label=f"Frontera $0.10 / $1.00 ({u1/1000:.0f}k px²)")
    ax_plot.axvline(u2 / 1000.0, color="darkgreen", linestyle="-.", linewidth=1.6, label=f"Frontera $1.00 / $0.50 ({u2/1000:.0f}k px²)")

    ax_plot.set_title("Espacio de Descriptores: Área vs. Circularidad (rho = 4*pi*Fp)", fontsize=11, fontweight="bold")
    ax_plot.set_xlabel("Área del Contorno [miles de píxeles²]", fontsize=10)
    ax_plot.set_ylabel("Circularidad Normalizada rho = 4*pi*Área / Perímetro²", fontsize=10)
    ax_plot.set_ylim(0.55, 0.96)
    ax_plot.grid(True, linestyle="--", alpha=0.45)
    ax_plot.legend(loc="lower right", fontsize=9)

    plt.suptitle("PROBLEMA 1 - PUNTO B: Clasificación Granulométrica y Conteo Automático de Monedas", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig2_path = os.path.join(out_dir, "p1_puntoB_clasificacion_monedas.png")
    fig2.savefig(fig2_path, dpi=160, bbox_inches="tight")

    # -------------------------------------------------------------------------
    # FIGURA 3 (PUNTO C): Determinación del Valor de la Cara Superior de Dados
    # -------------------------------------------------------------------------
    fig3, axes3 = plt.subplots(2, 5, figsize=(18, 9.0))
    for row, d in enumerate(dados):
        axes3[row, 0].imshow(d["crop_rgb"])
        axes3[row, 0].set_title(f"Dado #{d['id_dado']} - Recorte Original\nBBox={d['bbox']}", fontsize=9.5, fontweight="bold")

        axes3[row, 1].imshow(d["pips_bw"], cmap="gray")
        axes3[row, 1].set_title("1. Umbralado Inverso (T=90)\nPuntos + Fondo Exterior", fontsize=9.5, fontweight="bold")

        axes3[row, 2].imshow(d["pips_cb"], cmap="gray")
        axes3[row, 2].set_title("2. Limpieza de Borde (imclearborder)\nRemoción del fondo exterior", fontsize=9.5, fontweight="bold")

        axes3[row, 3].imshow(d["pips_clean"], cmap="gray")
        axes3[row, 3].set_title("3. Clausura (9x9) + Apertura (5x5)\nConsolidación de puntos internos", fontsize=9.5, fontweight="bold")

        vis_dado = d["crop_rgb"].copy()
        for p_sup in d["pips_cara_superior"]:
            cv2.drawContours(vis_dado, [p_sup["cnt"]], -1, (0, 230, 60), 3)
            cx_p, cy_p = int(p_sup["centroid"][0]), int(p_sup["centroid"][1])
            cv2.circle(vis_dado, (cx_p, cy_p), 5, (0, 230, 60), -1)
            cv2.putText(vis_dado, f"rho={p_sup['rho']:.2f}", (cx_p - 28, cy_p - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis_dado, f"rho={p_sup['rho']:.2f}", (cx_p - 28, cy_p - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 255, 140), 1, cv2.LINE_AA)

        for p_lat in d["pips_caras_laterales"]:
            cv2.drawContours(vis_dado, [p_lat["cnt"]], -1, (255, 50, 50), 3)
            cx_p, cy_p = int(p_lat["centroid"][0]), int(p_lat["centroid"][1])
            cv2.putText(vis_dado, f"lat({p_lat['rho']:.2f})", (max(2, cx_p - 30), min(vis_dado.shape[0] - 6, cy_p + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 80, 80), 1, cv2.LINE_AA)

        axes3[row, 4].imshow(vis_dado)
        axes3[row, 4].set_title(
            f"4. Clasificación por Factor de Forma\nCara Superior = {d['valor_cara_superior']} (Verde: rho > {RHO_TH:.2f})",
            fontsize=9.5, fontweight="bold"
        )

        for col_idx in range(5):
            axes3[row, col_idx].axis("off")

    plt.suptitle(
        f"PROBLEMA 1 - PUNTO C: Detección de Cara Superior de cada Dado (Dado 1 = {dados[0]['valor_cara_superior']}, "
        f"Dado 2 = {dados[1]['valor_cara_superior']}, Suma Total = {resumen_dados['suma_total']})",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    fig3_path = os.path.join(out_dir, "p1_puntoC_analisis_dados.png")
    fig3.savefig(fig3_path, dpi=160, bbox_inches="tight")

    if mostrar:
        plt.show()
    else:
        plt.close("all")

    return [fig1_path, fig1b_path, fig2_path, fig3_path]


# =============================================================================
# REPORTE POR CONSOLA Y EJECUCIÓN PRINCIPAL
# =============================================================================

def imprimir_reporte_consola(img_shape, objetos, monedas, dados, etapas, resumen_monedas, resumen_dados):
    print("=" * 88)
    print("TRABAJO PRÁCTICO N° 2 - PROCESAMIENTO DE IMÁGENES I (TUIA - FCEIA - UNR)")
    print("PROBLEMA 1: DETECCIÓN Y CLASIFICACIÓN DE MONEDAS Y DADOS (monedas.jpg)")
    print("=" * 88)
    print(f"Dimensiones de la imagen : {img_shape[1]} x {img_shape[0]} px ({img_shape[2]} canales)")
    print(f"Umbral Otsu directo (gris): {etapas['th_otsu_global']:.1f} (falla por fondo de intensidad no uniforme)")
    print(
        f"Umbral Gradiente Sobel    : {etapas['th_grad']:.1f} "
        f"(dinámico: μ={etapas['grad_mean']:.2f} + {etapas['k_opt']:.1f}·σ={etapas['grad_std']:.2f})"
    )
    print(f"Área mínima filtro (AREA_MIN): {etapas['area_min']:.1f} px² (0.14% de H×W)")

    print("\n" + "-" * 88)
    print("PUNTO A: SEGMENTACIÓN AUTOMÁTICA DE MONEDAS Y DADOS (Factor de Forma Fp = A / P^2)")
    print("-" * 88)
    print("Referencia teórica círculo: Fp = 1/(4*pi) = 0.0796 | 1/Fp = 12.57 | rho = 4*pi*Fp = 1.00")
    print(f"Criterio de clasificación : Moneda si rho > {RHO_TH:.2f} (1/Fp < {4.0 * np.pi / RHO_TH:.2f}) | Dado si rho <= {RHO_TH:.2f}")
    print(f"{'#':>3} | {'Tipo':<8} | {'Área [px²]':>11} | {'Perímetro [px]':>14} | {'Fp=A/P²':>8} | {'1/Fp':>7} | {'rho=4πFp':>8} | {'Centroide (x,y)':>16}")
    print("-" * 88)
    for idx, obj in enumerate(objetos, 1):
        cx, cy = obj["centroid"]
        print(
            f"{idx:3d} | {obj['tipo_objeto']:<8} | {obj['area']:11.1f} | {obj['perimetro']:14.1f} | "
            f"{obj['fp']:8.4f} | {obj['inv_fp']:7.2f} | {obj['rho']:8.4f} | ({cx:6.1f}, {cy:6.1f})"
        )
    print("-" * 88)
    print(f"Total de objetos segmentados: {len(objetos)} -> {len(monedas)} Monedas y {len(dados)} Dados.")

    print("\n" + "-" * 88)
    print("PUNTO B: CLASIFICACIÓN Y CONTEO AUTOMÁTICO DE MONEDAS")
    print("-" * 88)
    for clase, cant in resumen_monedas["conteo"].items():
        sub_areas = [m["area"] for m in monedas if m["denominacion"] == clase]
        rango_str = f"[{min(sub_areas):.0f} - {max(sub_areas):.0f}] px²" if sub_areas else "N/A"
        print(f" - Monedas de {clase:<12}: {cant:2d} unidades  (Rango de áreas: {rango_str})")
    print(f" -> CONTEO TOTAL DE MONEDAS : {resumen_monedas['total_monedas']} monedas")
    print(f" -> MONTO MONETARIO TOTAL   : ${resumen_monedas['monto_total']:.2f}")

    print("\n" + "-" * 88)
    print("PUNTO C: VALOR DE LA CARA SUPERIOR DE CADA DADO Y CONTEO AUTOMÁTICO")
    print("-" * 88)
    for d in dados:
        print(
            f" - Dado #{d['id_dado']} en BBox={d['bbox']} (rho_dado={d['rho']:.4f}):\n"
            f"     * Puntos circulares en CARA SUPERIOR (rho > {RHO_TH:.2f}) : {d['valor_cara_superior']} "
            f"(áreas={[int(p['area']) for p in d['pips_cara_superior']]}, "
            f"rho={[round(p['rho'], 3) for p in d['pips_cara_superior']]})\n"
            f"     * Puntos elípticos en CARAS LATERALES descartados : {len(d['pips_caras_laterales'])} "
            f"(áreas={[int(p['area']) for p in d['pips_caras_laterales']]}, "
            f"rho={[round(p['rho'], 3) for p in d['pips_caras_laterales']]})"
        )
    print(f" -> CANTIDAD TOTAL DE DADOS        : {resumen_dados['total_dados']} dados")
    print(f" -> VALORES DE CARAS SUPERIORES    : {resumen_dados['valores']}")
    print(f" -> SUMA TOTAL DE CARAS SUPERIORES : {resumen_dados['suma_total']}")
    print("=" * 88)


def main():
    parser = argparse.ArgumentParser(
        description="TP2 PDI - Problema 1: Detección y clasificación de monedas y dados."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Ruta a la imagen monedas.jpg (por defecto se busca en el directorio del script).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Muestra interactivamente las figuras en pantalla además de guardarlas.",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = args.image if args.image else os.path.join(base_dir, "monedas.jpg")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"No se encontró el archivo de imagen en: {img_path}")

    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Error al leer la imagen: {img_path}")

    # Punto A: Segmentación automática
    objetos, monedas, dados, etapas = segmentar_objetos(img_bgr)

    # Punto B: Clasificación y conteo de monedas
    monedas, resumen_monedas = clasificar_monedas(monedas)

    # Punto C: Determinación de la cara superior de cada dado y conteo
    dados, resumen_dados = analizar_dados(img_bgr, etapas["gray"], dados)

    # Informe por terminal
    imprimir_reporte_consola(
        img_bgr.shape, objetos, monedas, dados, etapas, resumen_monedas, resumen_dados
    )

    # Generación de figuras de etapas intermedias
    out_dir = os.path.join(base_dir, "resultados_problema1")
    rutas_figuras = generar_figuras_problema1(
        img_bgr, objetos, monedas, dados, etapas, resumen_monedas, resumen_dados, out_dir, mostrar=args.plot
    )
    print("\nFiguras de etapas intermedias guardadas en:")
    for r in rutas_figuras:
        print(f" - {r}")


if __name__ == "__main__":
    main()
