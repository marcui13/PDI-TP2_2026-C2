# Trabajo Práctico N° 2 — Procesamiento de Imágenes I (TUIA - FCEIA - UNR)

**Asignatura:** Procesamiento de Imágenes I (`IA 4.4`)  
**Carrera:** Tecnicatura Universitaria en Inteligencia Artificial (TUIA) — FCEIA — Universidad Nacional de Rosario (UNR)  
**Año Académico:** 2026 — 2° Cuatrimestre (`2026-C2`)  
**Integrantes:**
- **Agustín Marquardt**
- **Fabián Alvarez**

---

## 1. Descripción General del Proyecto

Este repositorio contiene la resolución integral del **Trabajo Práctico N° 2**, desarrollada aplicando estrictamente los conceptos y herramientas de las **Unidades 1 a 7** de la asignatura (filtrado espacial no lineal, operadores diferenciales de primer orden, umbralización estadística, morfología matemática binaria y en niveles de gris, reconstrucción morfológica geodésica `imreconstruct`, `imfillhole` e `imclearborder`, transformada Top-Hat y descriptores geométricos regionales como el Factor de Forma $F_p = A / P^2$ y la relación de aspecto).

El trabajo se estructura en dos problemas principales:

1. **Problema 1 (`problema1_monedas_y_dados.py`) — Detección y clasificación de monedas y dados (`monedas.jpg`):**
   - **Punto A:** Segmentación automática de los **19 objetos** presentes sobre un fondo con fuerte gradiente de iluminación no uniforme mediante filtrado de mediana ($11\times 11$), magnitud del **Gradiente de Sobel** ($\|\nabla I\| = \sqrt{G_x^2 + G_y^2}$), umbralado estadístico ($T=20$), clausura morfológica ($15\times 15$), rellenado geodésico de huecos (`imfillhole`) y apertura ($31\times 31$). Discriminación entre **17 Monedas** ($\rho > 0.80$) y **2 Dados** ($\rho \le 0.80$) mediante el **Factor de Forma** $F_p = A/P^2$ ($\rho = 4\pi F_p$).
   - **Punto B:** Clasificación granulométrica y conteo automático de los tres tipos de monedas según su área de contorno (`cv2.contourArea`): **9 monedas de 10 Centavos**, **5 monedas de 1 Peso** y **3 monedas de 50 Centavos** (**17 monedas en total**, monto acumulado de **\$7.40**).
   - **Punto C:** Determinación automática del valor representado por la **cara superior** de cada dado mediante umbralado inverso ($T=90$), limpieza de bordes por reconstrucción morfológica (`imclearborder`), clausura ($9\times 9$) + apertura ($5\times 5$) y filtrado de los puntos circulares de la cara superior ($\rho > 0.80$) frente a los puntos elípticos escorzados de las caras laterales ($\rho \le 0.80$): **Dado #1 = 5**, **Dado #2 = 3** (**Suma Total = 8**).

2. **Problema 2 (`problema2_deteccion_patentes.py`) — Detección de patentes y segmentación de caracteres (`img01.png` a `img12.png`):**
   - **Punto A:** Detección automática y segmentación de la patente en las 12 imágenes vehiculares mediante **Transformada Top-Hat** en niveles de gris (`cv2.MORPH_TOPHAT` con elemento estructurante rectangular de $21\times 9\text{ px}$), etiquetado de componentes conectadas (`cv2.connectedComponentsWithStats`) y evaluación de las 3 reglas geométricas de dominio: relación `Alto/Ancho` $\in [1.25, 3.5]$, formación de grupos de 3 caracteres colineales de altura uniforme y apareamiento de dos tripletes separados por el espacio central (**12/12 patentes detectadas = 100%**).
   - **Punto B:** Segmentación individual de los 6 caracteres alfanuméricos a partir exclusivamente del recorte de la patente detectada en el Punto A, empleando Top-Hat local ($21\times 9$), binarización, remoción del marco blanco perimetral mediante reconstrucción morfológica desde el borde (`imclearborder`, Unidad 6) y extracción ordenada de izquierda a derecha (**72/72 caracteres segmentados = 100%**).

---

## 2. Requisitos del Entorno y Versiones de Librerías Utilizadas

El código fue desarrollado y validado utilizando **Python 3.9+** y las siguientes versiones de librerías científicas:

| Librería / Paquete | Versión Utilizada | Uso en el Trabajo Práctico |
| :--- | :--- | :--- |
| **Python** | `3.9.6` (compatible con `3.9+`) | Intérprete principal del lenguaje |
| **NumPy** (`numpy`) | `2.0.2` | Operaciones matriciales, vectorización y cálculo de descriptores |
| **OpenCV** (`opencv-python`) | `5.0.0.93` (compatible con `4.x+`) | Filtrado espacial, Sobel, morfología matemática, Top-Hat y componentes conexas |
| **Matplotlib** (`matplotlib`) | `3.9.4` | Generación de figuras de etapas intermedias y del informe en PDF |
| **Pillow** (`pillow`) | `11.3.0` | Soporte auxiliar de lectura y escritura de imágenes |

### Instalación de dependencias

```bash
pip install numpy opencv-python matplotlib pillow
```

---

## 3. Instrucciones de Ejecución de los Scripts

Todos los scripts pueden ejecutarse directamente desde la terminal ubicándose en el directorio `TP2_2026-C2`:

### 3.1. Ejecutar el Problema 1 (Monedas y Dados)

```bash
# Ejecución estándar (imprime reporte completo por consola y guarda las figuras en resultados_problema1/)
python problema1_monedas_y_dados.py

# Ejecución mostrando además las ventanas interactivas de Matplotlib
python problema1_monedas_y_dados.py --plot
```

**Salidas generadas por `problema1_monedas_y_dados.py`:**
- `resultados_problema1/p1_puntoA_etapas_segmentacion.png`: Las 8 etapas del Punto A (Original, falla de Otsu directo, Mediana $11\times 11$, Gradiente de Sobel, Umbralado, Clausura $15\times 15$, Rellenado `imfillhole` + Apertura $31\times 31$, y clasificación Moneda vs. Dado por Factor de Forma $\rho$).
- `resultados_problema1/p1_puntoB_clasificacion_monedas.png`: Mapa espacial de las 17 monedas clasificadas por denominación y diagrama de dispersión `Área vs. Circularidad` con fronteras de decisión.
- `resultados_problema1/p1_puntoC_analisis_dados.png`: Las 5 etapas de procesamiento interno de cada dado y conteo de los puntos circulares de la cara superior ($\rho > 0.80$) descartando puntos elípticos laterales ($\rho \le 0.80$).

---

### 3.2. Ejecutar el Problema 2 (Detección de Patentes y Segmentación de Caracteres)

```bash
# Ejecución cíclica sobre las 12 imágenes de prueba (img01.png a img12.png)
python problema2_deteccion_patentes.py

# Ejecución sobre imágenes específicas y con visualización interactiva
python problema2_deteccion_patentes.py --images img01.png img04.png --plot
```

**Salidas generadas por `problema2_deteccion_patentes.py`:**
- `resultados_problema2/p2_puntoA_deteccion_12_patentes.png`: Mosaico de las 12 imágenes vehiculares con la caja contenedora detectada y vista ampliada del recorte de la patente.
- `resultados_problema2/p2_puntoB_segmentacion_caracteres_12_patentes.png`: Mosaico de las 12 patentes recortadas junto con su máscara libre de marco (`imclearborder`), etiquetado en color y los **72 caracteres individuales segmentados** ($C_1 \dots C_6$).
- `resultados_problema2/p2_etapas_detalle_img01.png`, `p2_etapas_detalle_img04.png`, `p2_etapas_detalle_img06.png`, `p2_etapas_detalle_img08.png`: Desglose detallado de las 8 etapas del pipeline (Puntos A y B) sobre escenas representativas.

---

### 3.3. Regenerar el Informe Técnico en PDF (`Informe_TP2_PDI.pdf`)

```bash
python generar_informe_tp2.py
```

---

## 4. Resumen Cuantitativo de Resultados

### 4.1. Problema 1 (`monedas.jpg`)

| Punto | Categoría / Objeto | Criterio Geométrico | Cantidad Detectada | Rango de Áreas $[\text{px}^2]$ | Circularidad $\rho = 4\pi A/P^2$ | Valor / Monto |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **A / B** | **Monedas de 10 Centavos** | $\rho > 0.80 \;\wedge\; A < 79\,744$ | **9** | $[62\,612 - 67\,606]$ | $[0.8056 - 0.8904]$ | $\$0.90$ |
| **A / B** | **Monedas de 1 Peso** | $\rho > 0.80 \;\wedge\; 79\,744 \le A < 102\,381$ | **5** | $[91\,882 - 97\,154]$ | $[0.8270 - 0.8679]$ | $\$5.00$ |
| **A / B** | **Monedas de 50 Centavos** | $\rho > 0.80 \;\wedge\; A \ge 102\,381$ | **3** | $[107\,607 - 110\,049]$ | $[0.8530 - 0.8739]$ | $\$1.50$ |
| **B** | **Total Monedas** | $\rho > 0.80$ | **17 monedas** | $[62\,612 - 110\,049]$ | $[0.8056 - 0.8904]$ | **$\$7.40$** |
| **A / C** | **Dado #1 (Izquierda)** | $\rho_{\text{dado}} = 0.7742 \le 0.80$ | **5 pts sup.** (3 lat. desc.) | Pts sup.: $[1\,937 - 2\,002]$ | Pts sup.: $[0.871 - 0.900]$ | **Cara sup. = 5** |
| **A / C** | **Dado #2 (Derecha)** | $\rho_{\text{dado}} = 0.6590 \le 0.80$ | **3 pts sup.** (0 lat. desc.) | Pts sup.: $[1\,968 - 2\,127]$ | Pts sup.: $[0.885 - 0.889]$ | **Cara sup. = 3** |
| **C** | **Total Dados** | $\rho \le 0.80$ | **2 dados** | $[66\,617 - 74\,462]$ | $[0.6590 - 0.7742]$ | **Suma = 8** |

---

### 4.2. Problema 2 (`img01.png` a `img12.png`)

| Imagen | Patente Ref. | Punto A: BBox Detectado $(x, y, w, h)$ | Umbral Top-Hat $T_A$ | Punto B: Umbral $T_B$ | Alturas de $C_1 \dots C_6$ $[\text{px}]$ | Caracteres Segmentados |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `img01.png` | `BUF 817` | `(333, 140, 77x33)` | `140` | `110` | `[16, 15, 16, 16, 16, 15]` | **6 / 6 (OK)** |
| `img02.png` | `GXQ 798` | `(320, 196, 73x35)` | `70` | `70` | `[16, 15, 15, 15, 14, 15]` | **6 / 6 (OK)** |
| `img03.png` | `DMH 024` | `(233, 174, 82x41)` | `85` | `85` | `[16, 17, 16, 16, 17, 16]` | **6 / 6 (OK)** |
| `img04.png` | `LHE 986` | `(312, 176, 84x42)` | `105` | `105` | `[16, 18, 18, 17, 16, 16]` | **6 / 6 (OK)** |
| `img05.png` | `FJY 344` | `(250, 108, 88x44)` | `55` | `55` | `[19, 19, 19, 19, 18, 18]` | **6 / 6 (OK)** |
| `img06.png` | `LEH 992` | `(279, 174, 108x46)` | `45` | `45` | `[21, 21, 21, 20, 21, 21]` | **6 / 6 (OK)** |
| `img07.png` | `LOD 123` | `(254, 163, 68x24)` | `70` | `55` | `[14, 14, 13, 13, 13, 13]` | **6 / 6 (OK)** |
| `img08.png` | `JMO 337` | `(185, 203, 77x32)` | `110` | `110` | `[14, 15, 15, 14, 15, 14]` | **6 / 6 (OK)** |
| `img09.png` | `FFD 765` | `(360, 232, 89x30)` | `65` | `65` | `[16, 16, 17, 17, 18, 17]` | **6 / 6 (OK)** |
| `img10.png` | `DOA 322` | `(292, 254, 79x30)` | `105` | `105` | `[15, 15, 16, 15, 16, 16]` | **6 / 6 (OK)** |
| `img11.png` | `GMQ 569` | `(255, 206, 69x28)` | `125` | `125` | `[12, 12, 11, 11, 11, 11]` | **6 / 6 (OK)** |
| `img12.png` | `BQX 387` | `(325, 191, 54x24)` | `60` | `60` | `[12, 11, 11, 11, 11, 11]` | **6 / 6 (OK)** |

---

## 5. Estructura del Repositorio

```text
TP2_2026-C2/
├── README.md                            # Documentación general, versiones e instrucciones de uso
├── Informe_TP2_PDI.pdf                  # Informe técnico completo con capturas de etapas y conclusiones
├── TUIA_PDI_TP2_2026_C2.pdf             # Consigna oficial del Trabajo Práctico N° 2
├── problema1_monedas_y_dados.py         # Script de resolución del Problema 1 (Puntos A, B y C)
├── problema2_deteccion_patentes.py      # Script de resolución del Problema 2 (Puntos A y B)
├── generar_informe_tp2.py               # Script generador del informe técnico en PDF
├── monedas.jpg                          # Imagen de entrada del Problema 1
├── img01.png ... img12.png              # 12 imágenes vehiculares de entrada del Problema 2
├── resultados_problema1/                # Figuras de etapas intermedias y resultados del Problema 1 (.png)
└── resultados_problema2/                # Figuras de etapas intermedias y resultados del Problema 2 (.png)
```
