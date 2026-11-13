# Guía Exhaustiva de Estudio para la Defensa Oral — Trabajo Práctico N° 2
## Procesamiento Digital de Imágenes I (`IA 4.4`) — TUIA — FCEIA — UNR (2026 - C2)

**Autores:** Agustín Marquardt, Damián Turco, Fabián Alvarez  
**Scripts analizados:** [`problema1_monedas_y_dados.py`](file:///Users/agustinmarquardt/Documents/TUIA/pdi/TP2_2026-C2/problema1_monedas_y_dados.py) y [`problema2_deteccion_patentes.py`](file:///Users/agustinmarquardt/Documents/TUIA/pdi/TP2_2026-C2/problema2_deteccion_patentes.py)

---

## Índice de Contenidos
1. [Mapa Conceptual: Vinculación entre el Código y las Unidades Teóricas (U1 a U7)](#1-mapa-conceptual-vinculación-entre-el-código-y-las-unidades-teóricas-u1-a-u7)
2. [Fundamentos Morfológicos de Cátedra Compartidos en Ambos Problemas (`imreconstruct`, `imfillhole`, `imclearborder`)](#2-fundamentos-morfológicos-de-cátedra-compartidos-en-ambos-problemas)
3. [PROBLEMA 1: Detección y Clasificación de Monedas y Dados (`monedas.jpg`)](#3-problema-1-detección-y-clasificación-de-monedas-y-dados-monedasjpg)
   - [3.1. Diagnóstico del Problema y Por Qué Falla Otsu Directo](#31-diagnóstico-del-problema-y-por-qué-falla-otsu-directo)
   - [3.2. Punto A: Tubería Paso a Paso de Segmentación sobre Fondo No Uniforme](#32-punto-a-tubería-paso-a-paso-de-segmentación-sobre-fondo-no-uniforme)
   - [3.3. Punto A: Separación Moneda vs. Dado mediante Factor de Forma ($F_p$ y $\rho$)](#33-punto-a-separación-moneda-vs-dado-mediante-factor-de-forma-f_p-y-rho)
   - [3.4. Punto B: Clasificación Granulométrica y Conteo Automático de Monedas](#34-punto-b-clasificación-granulométrica-y-conteo-automático-de-monedas)
   - [3.5. Punto C: Determinación del Valor de la Cara Superior de Cada Dado](#35-punto-c-determinación-del-valor-de-la-cara-superior-de-cada-dado)
4. [PROBLEMA 2: Detección Automática de Patentes y Segmentación de Caracteres (`img01.png` a `img12.png`)](#4-problema-2-detección-automática-de-patentes-y-segmentación-de-caracteres)
   - [4.1. Diagnóstico de la Problemática en Escenas Vehiculares Reales](#41-diagnóstico-de-la-problemática-en-escenas-vehiculares-reales)
   - [4.2. Punto A: Transformada Top-Hat en Niveles de Gris (`cv2.MORPH_TOPHAT`)](#42-punto-a-transformada-top-hat-en-niveles-de-gris-cv2morph_tophat)
   - [4.3. Punto A: Las 3 Reglas Geométricas y Estructurales (`info_patentes.txt`)](#43-punto-a-las-3-reglas-geométricas-y-estructurales-info_patentestxt)
   - [4.4. Punto B: Segmentación de los 6 Caracteres Individuales y Desacople del Marco con `imclearborder`](#44-punto-b-segmentación-de-los-6-caracteres-individuales-y-desacople-del-marco-con-imclearborder)
5. [Diccionario de Funciones de OpenCV (`cv2`) y NumPy (`np`) para la Defensa](#5-diccionario-de-funciones-de-opencv-cv2-y-numpy-np-para-la-defensa)
6. [Preguntas Clásicas de Defensa Oral y Respuestas Fundamentadas](#6-preguntas-clásicas-de-defensa-oral-y-respuestas-fundamentadas)

---

## 1. Mapa Conceptual: Vinculación entre el Código y las Unidades Teóricas (U1 a U7)

En una defensa oral de Procesamiento de Imágenes I, el tribunal evalúa que cada bloque de código tenga un **sustento teórico directo en las Unidades 1 a 7**:

| Unidad de la Materia | Concepto Teórico Aplicado | Dónde y Para Qué se Usa en Nuestro Código |
| :--- | :--- | :--- |
| **Unidad 1 y 2**<br>*(Fundamentos y Filtrado Espacial)* | Conversión de espacio de color a luminancia (`BGR2GRAY`) y **Filtro No Lineal de Mediana** (`cv2.medianBlur`). | **Problema 1 (Punto A):** Filtro de mediana $11\times 11$ antes de derivar. Elimina el grano de la superficie de apoyo preservando intactos los escalones de borde de las monedas y los dados (a diferencia de un filtro Gaussiano que difuminaría el borde). |
| **Unidad 3**<br>*(Segmentación: Bordes, Umbralado y Regiones)* | **Operador Diferencial de Sobel** (`cv2.Sobel`), **Umbralado de Otsu** (`THRESH_OTSU`), **Componentes Conectadas** (`connectedComponentsWithStats`) y **Contornos** (`findContours`). | **Problema 1 (Punto A):** El gradiente de Sobel $\|\nabla I\| = \sqrt{G_x^2 + G_y^2}$ anula la rampa suave de iluminación del fondo.<br>**Problema 2 (Puntos A y B):** `connectedComponentsWithStats` etiqueta candidatos a caracteres y extrae su Bounding Box $(x, y, w, h)$, área y centroide $(c_x, c_y)$. |
| **Unidad 6**<br>*(Morfología Matemática Binaria y en Grises)* | **Dilatación, Erosión, Apertura, Clausura** (`morphologyEx`), **Reconstrucción Morfológica Geodésica** (`imreconstruct`, `imfillhole`, `imclearborder`) y **Transformada Top-Hat en Grises** (`MORPH_TOPHAT`). | **Problema 1 (Puntos A y C):** Clausura $15\times 15$ para cerrar brechas en bordes de Sobel; `imfillhole` para rellenar el interior de monedas y dados; Apertura $31\times 31$ para alisar contornos; `imclearborder` para limpiar el fondo exterior del recorte de cada dado.<br>**Problema 2 (Puntos A y B):** Transformada **Top-Hat** ($21\times 9$) para resaltar los caracteres claros sobre fondo oscuro suprimiendo la carrocería y la calle; `imclearborder` para eliminar el marco blanco de la patente que toca el borde del recorte. |
| **Unidad 7**<br>*(Representación y Descripción Regional)* | **Área** ($A$), **Perímetro** ($P$), **Factor de Forma / Compacidad** ($F_p = A/P^2$, $\rho = 4\pi A/P^2$), **Relación de Aspecto** ($H/W$) y **Descriptores Relacionales**. | **Problema 1 (Puntos A, B y C):** $\rho > 0.80$ separa Monedas (círculos) de Dados (cuadrados) en el Punto A, y separa los puntos circulares de la **cara superior** ($\rho > 0.80$) de los puntos elípticos inclinados de las **caras laterales** ($\rho \le 0.80$) en el Punto C. El **Área** clasifica las 3 denominaciones ($10\text{ ctvs}$, $\$1$, $50\text{ ctvs}$) en el Punto B.<br>**Problema 2 (Puntos A y B):** La relación $H/W \in [1.25, 3.5]$ y la colinealidad/espaciado de dos grupos de 3 caracteres aíslan la patente sin falsos positivos. |

---

## 2. Fundamentos Morfológicos de Cátedra Compartidos en Ambos Problemas

Tanto en [`problema1_monedas_y_dados.py`](file:///Users/agustinmarquardt/Documents/TUIA/pdi/TP2_2026-C2/problema1_monedas_y_dados.py#L64-L109) como en [`problema2_deteccion_patentes.py`](file:///Users/agustinmarquardt/Documents/TUIA/pdi/TP2_2026-C2/problema2_deteccion_patentes.py#L81-L112) implementamos las tres funciones de **Reconstrucción Morfológica Binaria** vistas en la práctica de la Unidad 6 (`PI_U6_Morfologico_p2_Reconstruccion.py`):

### 2.1. Reconstrucción Geodésica Binaria (`imreconstruct`)
- **Definición teórica (Unidad 6):** Dada una imagen **máscara** $G$ (que restringe el crecimiento) y una imagen **marcador** $F \subseteq G$ (que indica desde dónde crecer), la dilatación geodésica de tamaño 1 se define como:
  $$\delta_G^{(1)}(F) = (F \oplus B) \cap G$$
  donde $B$ es el elemento estructurante de conectividad ($3\times 3$). La **reconstrucción morfológica** $R_G(F)$ consiste en repetir esta dilatación geodésica iterativamente hasta alcanzar la convergencia (estabilidad idempotente):
  $$R_G(F) = \delta_G^{(k)}(F) \quad \text{tal que} \quad \delta_G^{(k)}(F) = \delta_G^{(k-1)}(F)$$
- **Qué hace conceptualmente:** Extrae de la máscara $G$ **únicamente aquellas componentes conectadas que contienen al menos un píxel activo del marcador $F$**, sin cambiarles su forma ni su tamaño original (a diferencia de una dilatación común que engrosa los objetos).
- **Implementación en código:**
  ```python
  def imreconstruct(marker, mask, kernel=None):
      if kernel is None:
          kernel = np.ones((3, 3), np.uint8)
      prev = np.zeros_like(marker)
      curr = marker.copy()
      while not np.array_equal(curr, prev):
          prev = curr.copy()
          curr = cv2.min(cv2.dilate(curr, kernel), mask)
      return curr
  ```
  *(Nota: para imágenes binarias `0`/`255`, `cv2.min(dilatado, mask)` es equivalente a la intersección lógica `cv2.bitwise_and`).*

### 2.2. Rellenado Morfológico de Huecos (`imfillhole`)
- **Definición teórica (Unidad 6):** Un **hueco** es una región de fondo (negra) completamente encerrada por un objeto (blanco), es decir, que **no tiene conexión con el borde exterior de la imagen**.
- **Fórmula matemática:** Si inundamos el fondo desde el borde de la imagen complementada $I^c$ usando un marcador $F$ que vale $I^c$ en el borde y $0$ en el interior, la reconstrucción $R_{I^c}(F)$ recupera únicamente el fondo externo conectado al borde. Al volver a complementar ese resultado, todos los huecos internos quedan rellenos:
  $$H = \left[ R_{I^c}(F) \right]^c$$
- **Implementación eficiente en OpenCV (`cv2.floodFill`):** En [`imfillhole`](file:///Users/agustinmarquardt/Documents/TUIA/pdi/TP2_2026-C2/problema1_monedas_y_dados.py#L78-L90), la inundación geodésica desde el píxel de esquina `(0, 0)` sobre la imagen binaria calcula exactamente el fondo externo, y al invertirlo (`cv2.bitwise_not`) y sumarlo con `f | flood_inv` rellena todos los huecos interiores de las monedas y los dados.

### 2.3. Limpieza de Objetos en contacto con el Borde (`imclearborder`)
- **Definición teórica (Unidad 6):** Elimina cualquier componente conectada blanca que toque cualquiera de los 4 bordes de la imagen, dejando intactas todas las componentes que están completamente adentro.
- **Fórmula matemática:** Se define el marcador $F$ copiando únicamente la primera y última fila y la primera y última columna de la imagen $I$ (y poniendo ceros en todo el interior). La reconstrucción $R_I(F)$ recupera íntegramente todos los objetos que tocan el borde. Luego se restan de la imagen original:
  $$I_{\text{sin\_borde}} = I - R_I(F)$$
- **Por qué es clave en este TP:**
  1. En el **Problema 1 (Punto C)**, al invertir el recorte del dado (`THRESH_BINARY_INV`), tanto los puntos negros del dado como el fondo oscuro exterior que rodea al dado se vuelven blancos (`255`). Como el fondo exterior toca el marco del recorte, `imclearborder` lo borra de un solo paso y deja solo los puntos internos del dado.
  2. En el **Problema 2 (Punto B)**, el recorte de la patente Argentina tiene un **marco blanco perimetral** que toca el borde del recorte. Al aplicar `imclearborder`, ese marco blanco se elimina por completo sin tocar los 6 caracteres alfanuméricos centrales.

---

## 3. PROBLEMA 1: Detección y Clasificación de Monedas y Dados (`monedas.jpg`)

### 3.1. Diagnóstico del Problema y Por Qué Falla Otsu Directo
La imagen [`monedas.jpg`](file:///Users/agustinmarquardt/Documents/TUIA/pdi/TP2_2026-C2/monedas.jpg) tiene una resolución de $3871 \times 2681$ píxeles y presenta **tres complejidades simultáneas**:
1. **Fondo de intensidad fuertemente no uniforme:** Según el modelo físico de formación de imágenes (Unidad 1 y 3), $I(x, y) = i(x, y) \cdot r(x, y)$, donde $i(x, y)$ es la iluminación y $r(x, y)$ la reflectancia. Aquí la iluminación crece de izquierda a derecha: el fondo gris vale $I \approx 60$ a la izquierda y llega hasta $I \approx 185$ a la derecha.
   - Cuando aplicamos Otsu global directo (`cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)`), el umbral resultante es $T = 115$.
   - Con $T = 115$, toda la mitad derecha del fondo ($I \in [116, 185]$) se clasifica erróneamente como objeto blanco (`255`), fusionando las monedas derechas con el fondo (como se demuestra en el panel 2 de [`p1_puntoA_etapas_segmentacion.png`](file:///Users/agustinmarquardt/Documents/TUIA/pdi/TP2_2026-C2/resultados_problema1/p1_puntoA_etapas_segmentacion.png)).
2. **Heterogeneidad interna de los objetos:** Las monedas de $\$1$ son bimetálicas (anillo plateado exterior y núcleo bronce oscuro interior con relieves), y los dados son blancos con puntos negros profundos.
3. **Perspectiva oblicua en los dados:** En el Dado #1 (arriba a la izquierda) se ven claramente los **5 puntos de la cara superior** y **3 puntos en la cara lateral derecha**. Si contáramos componentes oscuras sin analizar su geometría, contaríamos $5 + 3 = 8$ en vez de $5$.

---

### 3.2. Punto A: Tubería Paso a Paso de Segmentación sobre Fondo No Uniforme

La función [`segmentar_objetos(img_bgr)`](file:///Users/agustinmarquardt/Documents/TUIA/pdi/TP2_2026-C2/problema1_monedas_y_dados.py#L140-L249) resuelve la iluminación no uniforme aprovechando una propiedad matemática fundamental de los operadores diferenciales (Unidad 3):
> **Propiedad clave para la defensa:** La iluminación no uniforme del fondo varía muy lentamente de píxel a píxel (baja frecuencia espacial), por lo que su derivada espacial es casi nula ($\|\nabla I_{\text{fondo}}(x,y)\| \approx 0$). En cambio, en el perímetro donde termina la mesa y empieza una moneda o un dado, hay un salto brusco de intensidad (alta frecuencia espacial), produciendo un pico elevado en la magnitud del gradiente $\|\nabla I\|$.

Los pasos exactos implementados son:

1. **Conversión a Escala de Grises y Filtro de Mediana ($11\times 11$) — Unidad 2:**
   ```python
   gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
   gray_med = cv2.medianBlur(gray, 11)
   ```
   - **Por qué Mediana y no Gaussiano:** El filtro de mediana ordena las intensidades en la vecindad de $11\times 11$ ($121$ píxeles) y toma el valor central. Suprime la textura granulada del fondo oscuro sin suavizar ni desplazar los bordes abruptos de las monedas y dados.

2. **Magnitud Exacta del Gradiente de Sobel — Unidad 3:**
   ```python
   gx = cv2.Sobel(gray_med, cv2.CV_64F, 1, 0, ksize=3)
   gy = cv2.Sobel(gray_med, cv2.CV_64F, 0, 1, ksize=3)
   grad_mag = cv2.magnitude(gx, gy)
   grad_u8 = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
   ```
   - **Por qué `cv2.CV_64F`:** Las derivadas pueden ser negativas cuando la intensidad baja de izquierda a derecha o de arriba a abajo. Si usáramos `uint8` directamente al llamar a `cv2.Sobel`, OpenCV truncaría las derivadas negativas a `0`, perdiendo la mitad de los bordes. Por eso se calcula en punto flotante de 64 bits (`CV_64F`), se obtiene la magnitud euclídea $\|\nabla I\| = \sqrt{G_x^2 + G_y^2}$ con `cv2.magnitude`, y luego se normaliza a `[0, 255]`.
   - **Anulación de margen de 10 px:** Se pone en cero un marco de 10 píxeles (`grad_u8[:10, :] = 0`, etc.) para asegurar que el borde de la imagen sea `0` puro y que el `floodFill` de `imfillhole` desde `(0,0)` pueda recorrer libremente todo el contorno exterior.

3. **Umbralado Estadístico Automático del Gradiente — Unidad 3:**
   ```python
   mean_g, std_g = float(np.mean(grad_u8)), float(np.std(grad_u8))
   th_grad = max(18.0, round(mean_g + 0.9 * std_g))  # Da exactamente T = 20.0
   _, bw_edges = cv2.threshold(grad_u8, th_grad, 255, cv2.THRESH_BINARY)
   ```
   - Como el fondo ahora tiene gradiente casi cero en toda la imagen (tanto a la izquierda como a la derecha), un único umbral estadístico global $T = \mu_G + 0.9\sigma_G = 20.0$ separa perfectamente los anillos de borde de los 19 objetos.

4. **Clausura Morfológica ($15\times 15$) + Rellenado (`imfillhole`) + Apertura ($31\times 31$) — Unidad 6:**
   ```python
   se_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
   bw_closed = cv2.morphologyEx(bw_edges, cv2.MORPH_CLOSE, se_close)
   bw_filled = imfillhole(bw_closed)
   se_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
   bw_mask = cv2.morphologyEx(bw_filled, cv2.MORPH_OPEN, se_open)
   ```
   - **Clausura ($\bullet$, dilatación seguida de erosión con elipse $15\times 15$):** Sella cualquier micro-interrupción en el anillo del gradiente exterior (especialmente en las esquinas redondeadas de los dados donde el contraste local con el fondo claro disminuye). Es indispensable que el anillo exterior esté 100% cerrado antes de llamar a `imfillhole`.
   - **Rellenado (`imfillhole`):** Convierte los 19 anillos cerrados en **19 discos/regiones sólidas blancas**, tapando por completo todas las cavidades internas del grabado de las monedas y los puntos de los dados.
   - **Apertura ($\circ$, erosión seguida de dilatación con elipse $31\times 31$):** Elimina pequeñas protuberancias periféricas del borde y suaviza el perímetro antes de medir el Factor de Forma.

---

### 3.3. Punto A: Separación Moneda vs. Dado mediante Factor de Forma ($F_p$ y $\rho$)

Extraemos los contornos externos con `cv2.findContours(bw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)` (Unidad 3 y 7) y filtramos regiones con $\text{Área} \ge 10\,000\text{ px}^2$ (los objetos reales miden entre $62\,612$ y $110\,049\text{ px}^2$).

Para cada objeto calculamos su **Área** ($A = \text{cv2.contourArea(cnt)}$) y su **Perímetro** ($P = \text{cv2.arcLength(cnt, True)}$), y aplicamos las dos expresiones equivalentes vistas en la cátedra ([`Factor de Forma.txt`](file:///Users/agustinmarquardt/Documents/TUIA/pdi/TP2_2026-C2/Factor%20de%20Forma.txt) y `u6/clasifico_celulas.py`):

1. **Factor de Forma de [`Factor de Forma.txt`](file:///Users/agustinmarquardt/Documents/TUIA/pdi/TP2_2026-C2/Factor%20de%20Forma.txt):**
   $$F_p = \frac{A}{P^2}$$
   - Para un **círculo de radio $r$**:
     $$F_p = \frac{\pi r^2}{(2\pi r)^2} = \frac{1}{4\pi} \approx 0.07958 \implies \frac{1}{F_p} = 4\pi \approx 12.57$$
   - Para un **cuadrado de lado $a$**:
     $$F_p = \frac{a^2}{(4a)^2} = \frac{1}{16} = 0.06250 \implies \frac{1}{F_p} = 16.00$$

2. **Circularidad Normalizada de `u6/clasifico_celulas.py`:**
   $$\rho = 4\pi \cdot F_p = \frac{4\pi A}{P^2} \in (0, 1]$$
   - En un **círculo perfecto**, $\rho = 1.00$.
   - En un **cuadrado perfecto**, $\rho = \pi / 4 \approx 0.7854$.

Usando el mismo umbral de la cátedra `RHO_TH = 0.80` ($F_p = 0.0637$, $1/F_p = 15.71$):
- **Las 17 Monedas** tienen $\rho \in [0.8056, 0.8904] > 0.80$ ($1/F_p \in [14.11, 15.60]$).
- **Los 2 Dados** (cuya proyección 2D es cuasi-cuadrada/hexagonal por perspectiva) tienen $\rho = 0.7742$ ($1/F_p = 16.23$) y $\rho = 0.6590$ ($1/F_p = 19.07$), ambos $\le 0.80$.

---

### 3.4. Punto B: Clasificación Granulométrica y Conteo Automático de Monedas

En la función [`clasificar_monedas(monedas)`](file:///Users/agustinmarquardt/Documents/TUIA/pdi/TP2_2026-C2/problema1_monedas_y_dados.py#L256-L303):
- Como la cámara está a una altura fija perpendicular a la mesa, el **Área en píxeles cuadrados** ($A = \pi R^2$) es un descriptor de tamaño (granulometría, Unidad 7) que separa limpiamente las tres denominaciones físicas:
  - **10 Centavos (diámetro menor):** 9 monedas con áreas entre $62\,612$ y $67\,606\text{ px}^2$.
  - **1 Peso (diámetro intermedio):** 5 monedas con áreas entre $91\,882$ y $97\,154\text{ px}^2$.
  - **50 Centavos (diámetro mayor):** 3 monedas con áreas entre $107\,607$ y $110\,049\text{ px}^2$.
- **Cómo se calculan los umbrales de área automáticamente en el código:**
  En lugar de hardcodear números mágicos, ordenamos el vector de las 17 áreas `areas_ordenadas = np.sort(areas)` y calculamos las diferencias entre elementos sucesivos `diffs = np.diff(areas_ordenadas)`. Los dos mayores saltos (`np.argsort(diffs)[-2:]`) corresponden exactamente a las brechas vacías entre las 3 clases:
  - Salto 1: entre $67\,606$ y $91\,882\text{ px}^2 \implies$ Punto medio $U_1 = 79\,744\text{ px}^2$.
  - Salto 2: entre $97\,154$ y $107\,607\text{ px}^2 \implies$ Punto medio $U_2 = 102\,381\text{ px}^2$.
- **Resultado del conteo:** **9 de 10 ctvs** ($\$0.90$) + **5 de 1 Peso** ($\$5.00$) + **3 de 50 ctvs** ($\$1.50$) = **17 monedas** y **$\$7.40$** en total.

---

### 3.5. Punto C: Determinación del Valor de la Cara Superior de Cada Dado

La función [`analizar_dados(img_bgr, dados)`](file:///Users/agustinmarquardt/Documents/TUIA/pdi/TP2_2026-C2/problema1_monedas_y_dados.py#L310-L369) resuelve la pregunta más interesante del Problema 1: **¿Cómo distinguir los puntos de la cara superior de los puntos visibles en las caras laterales?**

1. **Recorte (ROI) y Umbralado Inverso ($T = 90$):**
   Tomamos la caja contenedora `bbox` de cada dado sobre la imagen en escala de grises y aplicamos `cv2.threshold(crop_gray, 90, 255, cv2.THRESH_BINARY_INV)`.Como el cuerpo del dado es blanco brillante ($I > 160$) y tanto los puntos (pips) como el fondo exterior son oscuros ($I < 90$), quedan en blanco (`255`) los puntos del dado y el fondo exterior que rodea al dado.
2. **Eliminación del fondo exterior con `imclearborder` (Unidad 6):**
   Al llamar a `pips_cb = imclearborder(pips_bw, conn=8)`, el fondo oscuro exterior (que toca los 4 bordes de la sub-imagen del dado) se reconstruye desde el borde y se suprime por completo. ¡Solo sobreviven las cavidades oscuras que están adentro del cuerpo blanco del dado!
3. **Clausura ($9\times 9$) + Apertura ($5\times 5$) (Unidad 6):**
   Los puntos negros de los dados son concavidades brillantes que reflejan un pequeño punto de luz blanca en su centro (visible en la columna 2 de [`p1_puntoC_analisis_dados.png`](file:///Users/agustinmarquardt/Documents/TUIA/pdi/TP2_2026-C2/resultados_problema1/p1_puntoC_analisis_dados.png)). La clausura con elipse $9\times 9$ rellena esos reflejos internos y la apertura $5\times 5$ limpia cualquier píxel aislado.
4. **Geometría Proyectiva + Factor de Forma ($\rho = 4\pi A/P^2$) (Unidad 7):**
   - La **cara superior** del dado es prácticamente paralela al plano del sensor de la cámara, por lo que sus puntos circulares se proyectan como **círculos** con alta circularidad:
     $$\rho_{\text{cara superior}} \in [0.871, 0.900] > 0.80 \quad (\text{Área} \approx 1950 - 2127\text{ px}^2)$$
   - Las **caras laterales** del dado están inclinadas respecto al eje óptico en un ángulo $\theta \approx 65^\circ$. Por acortamiento proyectivo (escorzo), un círculo de radio $r$ inclinado un ángulo $\theta$ se proyecta como una **elipse** de semiejes $a = r$ y $b = r\cos\theta$, cuya circularidad cae drásticamente:
     $$\rho_{\text{cara lateral}} \in [0.550, 0.590] \le 0.80 \quad (\text{Área} \approx 622 - 649\text{ px}^2)$$
   - Aplicando **exactamente el mismo umbral $\rho_{th} = 0.80$**:
     - **Dado #1:** Se aceptan los **5 puntos circulares** de la cara superior y se descartan automáticamente los 3 puntos elípticos laterales $\implies$ **Valor = 5**.
     - **Dado #2:** Se aceptan los **3 puntos circulares** de la cara superior $\implies$ **Valor = 3**.
     - **Suma Total de los 2 Dados = 8**.

---

## 4. PROBLEMA 2: Detección Automática de Patentes y Segmentación de Caracteres

### 4.1. Diagnóstico de la Problemática en Escenas Vehiculares Reales
Las 12 imágenes ([`img01.png`](file:///Users/agustinmarquardt/Documents/TUIA/pdi/TP2_2026-C2/img01.png) a [`img12.png`](file:///Users/agustinmarquardt/Documents/TUIA/pdi/TP2_2026-C2/img12.png), todas de $640 \times 360\text{ px}$) muestran autos en la vía pública bajo condiciones muy variadas:
- **Cambios de escala (distancia cámara-vehículo):** En `img06.png` (`LEH 992`, muy cerca) los caracteres miden $21\text{ px}$ de alto, mientras que en `img11.png` (`GMQ 569`) e `img12.png` (`BQX 387`, más lejos) miden apenas $11\text{ px}$ de alto.
- **Inclinación y perspectiva:** Patentes rotadas en el plano como `img03.png` (`DMH 024`), `img04.png` (`LHE 986`) e `img11.png` (`GMQ 569`).
- **Oclusiones cercanas y fondos distractores:** En `img04.png` e `img08.png` hay un poste metálico gris justo delante del paragolpes; además hay adoquines en la calle, ópticas brillantes, parrillas de radiador y logos cromados (Peugeot, VW, Renault, Fiat, Toyota).

---

### 4.2. Punto A: Transformada Top-Hat en Niveles de Gris (`cv2.MORPH_TOPHAT`)

¿Por qué no alcanza con umbralizar la imagen en escala de grises directamente? Porque el cielo, la pintura blanca o gris plata de los autos (`img02`, `img06`, `img10`, `img11`, `img12`), las ópticas y el pavimento soleado son tanto o más brillantes que las letras blancas de la patente.

La solución teórica exacta de la **Unidad 6 (`PI_U6_Morfologico_p4_Grises.py`)** es la **Transformada Top-Hat Blanca (White Top-Hat)**:
$$I_{\text{tophat}}(x, y) = I(x, y) - (I \circ B)(x, y)$$
donde $I \circ B = (I \ominus B) \oplus B$ es la **apertura morfológica en niveles de gris** con el elemento estructurante $B$.

- **Cómo funciona físicamente la apertura en grises $(I \circ B)$:** Si imaginamos la imagen en escala de grises como una superficie topográfica 3D donde la altura es el brillo $I(x,y)$, la apertura consiste en empujar el elemento estructurante $B$ **desde abajo** contra la superficie luminosa.
  - Si un pico brillante es **más angosto** que el elemento estructurante $B$ (como los trazos blancos de las letras y números de la patente, que tienen un ancho de $2$ a $13\text{ px}$), el elemento estructurante $B$ **no cabe** dentro del pico y lo aplana por completo en $(I \circ B)$.
  - Si una región brillante es **más grande** que $B$ (como el capó blanco de un auto, el paragolpes o la calle), $B$ cabe perfectamente adentro y la apertura $(I \circ B)$ conserva su brillo original.
- **Al restar $I - (I \circ B)$:** Las grandes superficies claras se cancelan ($I - I \approx 0$, quedan negras) y **sobreviven únicamente las estructuras brillantes más pequeñas que $B$** que están rodeadas de un fondo oscuro (los caracteres blancos de la patente).
- **Por qué elegimos un Elemento Estructurante Rectangular de $21 \times 9$ píxeles (`cv2.getStructuringElement(cv2.MORPH_RECT, (21, 9))`):**
  En todas las imágenes, el ancho de cada carácter individual es $W \in [5, 13]\text{ px}$. Con un ancho de $21\text{ px}$ ($> 13\text{ px}$), garantizamos que ningún carácter quepa horizontalmente dentro del elemento estructurante, por lo que los 6 caracteres siempre producen una respuesta intensa en el Top-Hat.

---

### 4.3. Punto A: Las 3 Reglas Geométricas y Estructurales (`info_patentes.txt`)

Incluso después del Top-Hat, en la escena siguen apareciendo pequeños reflejos brillantes (en la parrilla del radiador, faros o bordes de adoquines). Para descartar el 100% de esos falsos positivos, implementamos en [`filtrar_candidatos_caracter`](file:///Users/agustinmarquardt/Documents/TUIA/pdi/TP2_2026-C2/problema2_deteccion_patentes.py#L119-L146) y [`buscar_constelacion_patente`](file:///Users/agustinmarquardt/Documents/TUIA/pdi/TP2_2026-C2/problema2_deteccion_patentes.py#L149-L233) las **3 reglas geométricas del dominio** indicadas por la cátedra en [`info_patentes.txt`](file:///Users/agustinmarquardt/Documents/TUIA/pdi/TP2_2026-C2/info_patentes.txt):

1. **Regla 1 — Relación `Alto/Ancho` de cada carácter individual:**
   > *"Relación Alto/Ancho de cada caracter de la patente: ~ 1.5 - 3.0"*
   - Para cada componente conectada extraída con `cv2.connectedComponentsWithStats(bw_mask, connectivity=8)`, leemos $(x, y, w, h, \text{area})$ y $(c_x, c_y)$ y exigimos:
     $$9 \le h \le 30\text{ px}, \quad 3 \le w \le 20\text{ px}, \quad 1.25 \le \frac{h}{w} \le 3.5, \quad \text{area} \ge 15\text{ px}^2$$
   - Esto reduce cientos de regiones de ruido de la escena a solo **entre 7 y 12 candidatos** en toda la imagen (ver columna `Cand->Trip` en la salida de consola).

2. **Regla 2 — Agrupamiento en Tripletes (Grupos de 3 caracteres):**
   > *"Los caracteres forman grupos de 3."*
   - Ordenamos los candidatos de izquierda a derecha por coordenada $x$ y buscamos ternas $(c_1, c_2, c_3)$ que verifiquen:
     - **Uniformidad de altura:** Los 3 caracteres de un mismo grupo tienen la misma escala física, por lo que $\max_i |h_i - \bar{h}| \le 3.5\text{ px}$.
     - **Espaciado horizontal regular:** La distancia entre centroides consecutivos $\Delta x_1 = c_{x,2} - c_{x,1}$ y $\Delta x_2 = c_{x,3} - c_{x,2}$ debe ser proporcional a la altura media $\bar{h}$:
       $$0.40\,\bar{h} \le \Delta x_1, \Delta x_2 \le 1.15\,\bar{h} \quad \text{y} \quad |\Delta x_1 - \Delta x_2| \le 0.35\,\bar{h}$$
     - **Alineación cuasi-horizontal:** $|\Delta y_1| \le 0.35\,\bar{h}$ y $|\Delta y_2| \le 0.35\,\bar{h}$ (permite patentes inclinadas como `img04.png` pero rechaza puntos desalineados verticalmente).

3. **Regla 3 — Apareamiento de Dos Grupos de 3 Caracteres (Letras + Dígitos):**
   > *"La patente está formada por dos grupos de 3 caracteres."*
   - En las patentes argentinas del formato `AAA 000`, las 3 letras y los 3 números están separados en el centro por un espacio mayor que el espacio entre letras contiguas.
   - Dados dos tripletes $T_1 = (c_1, c_2, c_3)$ y $T_2 = (c_4, c_5, c_6)$, verificamos:
     - **Igualdad de escala:** $|\bar{h}_1 - \bar{h}_2| \le 3.5\text{ px}$.
     - **Espacio central característico:** Si $\Delta x_{\text{intra}}$ es la separación media dentro de cada triplete, la separación central $\Delta x_{\text{central}} = c_{x,4} - c_{x,3}$ debe cumplir:
       $$1.25\,\Delta x_{\text{intra}} \le \Delta x_{\text{central}} \le 2.80\,\Delta x_{\text{intra}}$$
     - **Colinealidad proyectiva entre ambos tripletes:** Si la patente está inclinada con pendiente $\Delta y_1 / \Delta x_{\text{intra}}$, el segundo triplete debe continuar sobre la misma recta:
       $$\left| \Delta y_{\text{central}} - \Delta y_1 \cdot \frac{\Delta x_{\text{central}}}{\Delta x_{\text{intra}}} \right| \le 4.0\text{ px}$$
- **Barrido de Umbral (`range(35, 165, 5)`) y Función de Puntaje:**
  Como algunas patentes están al sol (`img01`, umbral óptimo $T_A = 140$) y otras en sombra (`img06`, umbral óptimo $T_A = 45$), el algoritmo evalúa umbrales de $35$ a $160$ con paso $5$ y elige la constelación de 6 caracteres que maximiza el puntaje estructural:
  $$\text{Score} = \sum_{k=1}^{6} \text{Área}(c_k) - 10 \cdot \sigma_h$$
  Esto premia caracteres bien definidos (mayor área sólida) y de altura uniforme (mínima desviación estándar de alturas $\sigma_h$).
- **Recorte de la Patente (Bounding Box con margen proporcional):**
  A partir del mínimo y máximo $(x, y)$ de los 6 caracteres detectados, se agrega un margen horizontal de $0.45\,\bar{h}$ y vertical de $0.35\,\bar{h}$ para incluir el cuerpo completo de la chapa patente.

---

### 4.4. Punto B: Segmentación de los 6 Caracteres Individuales y Desacople del Marco con `imclearborder`

El enunciado del **Punto B** pide:
> *"Implementar un algoritmo de procesamiento que permita segmentar los caracteres individuales a partir de la patente detectada en el punto A."*

En [`segmentar_caracteres_patente(recorte_bgr)`](file:///Users/agustinmarquardt/Documents/TUIA/pdi/TP2_2026-C2/problema2_deteccion_patentes.py#L306-L407), la única entrada es el recorte `recorte_bgr` obtenido en el Punto A.

#### ¿Cuál es la trampa técnica del Punto B y cómo la resolvimos?
Las patentes argentinas (`AAA 000`) tienen un **marco blanco pintado alrededor de todo el borde de la chapa**. Cuando recortamos la patente y la binarizamos:
1. El marco blanco aparece como una barra blanca arriba y abajo de las letras (ver panel 6 superior de [`p2_etapas_detalle_img01.png`](file:///Users/agustinmarquardt/Documents/TUIA/pdi/TP2_2026-C2/resultados_problema2/p2_etapas_detalle_img01.png)).
2. Si usamos un único umbral bajo (como Otsu directo sobre el recorte), la difusión óptica hace que la parte superior o inferior de varias letras toque por 1 píxel al marco blanco, uniendo las letras con el marco en una sola componente conectada gigante.

Nuestra solución combina tres herramientas de las **Unidades 3, 6 y 7**:
1. **Top-Hat Local ($21\times 9$) sobre el recorte de la patente:** Resalta los trazos verticales de los caracteres y atenúa las barras horizontales largas del marco.
2. **Limpieza de Bordes por Reconstrucción Morfológica (`imclearborder`, Unidad 6):** Como el marco blanco de la patente se extiende hasta tocar los bordes del recorte mientras que los 6 caracteres están en el centro, `bw_cb = imclearborder(bw_raw, conn=8)` **borra por completo el marco perimetral blanco**.
3. **Selección de umbral guiada por descriptores (`connectedComponentsWithStats`):** Se evalúa el barrido de umbral sobre el Top-Hat local buscando el umbral $T_B$ que, tras aplicar `imclearborder`, deja exactamente **6 componentes conectadas** internas con:
   - Altura entre el $35\%$ y el $92\%$ de la altura del recorte (`0.35 * hc <= h <= 0.92 * hc`).
   - Relación de aspecto `1.25 <= h / w <= 3.5` y `area >= 15`.
   - Diferencia máxima de altura entre el carácter más alto y el más bajo de a lo sumo $4\text{ px}$ (`max(hs) - min(hs) <= 4`).
4. **Ordenamiento Izquierda-Derecha:** Se ordenan las 6 regiones por coordenada $x$ creciente (`sorted(comps, key=lambda c: c["x"])`), obteniendo los 6 caracteres individuales $C_1, C_2, C_3$ (letras) y $C_4, C_5, C_6$ (dígitos) con **100% de éxito en las 12 imágenes (72/72 caracteres)**.

---

## 5. Diccionario de Funciones de OpenCV (`cv2`) y NumPy (`np`) para la Defensa

Esta tabla resume qué hace exactamente cada función llamada en ambos scripts, sus parámetros y qué devuelve, para responder preguntas puntuales de código durante la defensa:

| Función Llamada | Unidad | Parámetros Clave | Qué Devuelve y Por Qué se Usa |
| :--- | :---: | :--- | :--- |
| `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` | U1 / U5 | `COLOR_BGR2GRAY` | Convierte matriz BGR $(H, W, 3)$ a luminancia monocromática $(H, W)$ `uint8` mediante $Y = 0.114 B + 0.587 G + 0.299 R$. |
| `cv2.medianBlur(gray, 11)` | U2 | `ksize=11` (ventana impar $11\times 11$) | Reemplaza cada píxel por la mediana estadística de su entorno $11\times 11$. Elimina ruido/textura preservando bordes abruptos. |
| `cv2.Sobel(gray_med, cv2.CV_64F, dx, dy, ksize=3)` | U3 | `ddepth=cv2.CV_64F`, `dx=1, dy=0` o `dx=0, dy=1` | Calcula las derivadas direccionales horizontales $G_x$ y verticales $G_y$ por convolución con los kernels de Sobel $3\times 3$. Se usa `CV_64F` para no truncar derivadas negativas. |
| `cv2.magnitude(gx, gy)` | U3 | Matrices `gx`, `gy` (`float64`) | Calcula punto a punto la norma euclídea del gradiente $\sqrt{G_x(x,y)^2 + G_y(x,y)^2}$. |
| `cv2.threshold(src, thresh, 255, type)` | U3 | `THRESH_BINARY`, `THRESH_BINARY_INV`, `THRESH_OTSU` | Devuelve `(umbral_usado, img_binaria)`. Genera máscaras binarias `0`/`255`. |
| `cv2.getStructuringElement(shape, ksize)` | U6 | `MORPH_ELLIPSE` (P1) o `MORPH_RECT` (P2) | Construye la matriz del elemento estructurante $B$: elíptico isotrópico para monedas/dados y rectangular $(21, 9)$ para patentes. |
| `cv2.morphologyEx(src, op, kernel)` | U6 | `MORPH_CLOSE`, `MORPH_OPEN`, `MORPH_TOPHAT` | Ejecuta operaciones morfológicas compuestas: Clausura $(A \oplus B) \ominus B$, Apertura $(A \ominus B) \oplus B$ y Top-Hat $I - (I \circ B)$. |
| `cv2.floodFill(img, mask, (0,0), 255)` | U6 | Semilla `(0, 0)` y `mask` de $(H+2, W+2)$ | Inunda por conectividad el fondo exterior desde la esquina superior izquierda; base de nuestra implementación rápida de `imfillhole`. |
| `cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)` | U3 / U7 | `RETR_EXTERNAL` (solo contornos exteriores), `CHAIN_APPROX_NONE` | Extrae la lista de contornos exteriores guardando absolutamente todos los puntos del perímetro (sin comprimir segmentos) para medir con precisión $P$ y $A$. |
| `cv2.contourArea(cnt)` y `cv2.arcLength(cnt, True)` | U7 | `cnt` (contorno), `closed=True` | Calculan el área encerrada $A$ (fórmula de Green/Shoelace) y el perímetro cerrado $P$ para obtener el Factor de Forma $F_p = A/P^2$. |
| `cv2.connectedComponentsWithStats(bw, connectivity=8)` | U3 / U7 | `connectivity=8` (vecindad de 8 píxeles) | Devuelve `(num_labels, labels, stats, centroids)`. Para cada etiqueta $i \ge 1$, `stats[i]` contiene `[x, y, w, h, area]` y `centroids[i]` contiene `[cx, cy]`. |

---

## 6. Preguntas Clásicas de Defensa Oral y Respuestas Fundamentadas

### Pregunta 1: *"En el Problema 1, ¿por qué usaron el Gradiente de Sobel en lugar de hacer una umbralización global con Otsu o elegir otro canal de color?"*
- **Respuesta:** Porque en `monedas.jpg` la iluminación del fondo no es constante: tiene una rampa suave que va desde nivel de gris $\sim 60$ a la izquierda hasta $\sim 185$ a la derecha. Como las monedas y el fondo son acromáticos/metálicos, el gradiente de iluminación afecta a los 3 canales RGB y al canal $V$ de HSV por igual, haciendo que Otsu ($T=115$) sature toda la mitad derecha de la imagen. En cambio, como esa variación de iluminación del fondo es espacialmente suave (baja frecuencia), su primera derivada espacial es casi cero ($\|\nabla b(x,y)\| \approx 0$). Al aplicar el operador de Sobel, el fondo queda uniformemente oscuro en toda la imagen y resaltan únicamente los saltos abruptos de intensidad de los contornos de las 17 monedas y los 2 dados.

### Pregunta 2: *"¿Cómo funciona exactamente el Factor de Forma y por qué les sirvió tanto en el Punto A como en el Punto C del Problema 1?"*
- **Respuesta:** El Factor de Forma $F_p = A / P^2$ (o su versión normalizada $\rho = 4\pi A / P^2$) es un descriptor geométrico adimensional de compacidad (Unidad 7) que es **invariante a la traslación, a la rotación y al cambio de escala**. Por la desigualdad isoperimétrica, el círculo es la figura plana que encierra la máxima área para un perímetro dado ($\rho = 1.00$, $1/F_p = 12.57$), mientras que figuras con esquinas como un cuadrado ($\rho = \pi/4 \approx 0.785$, $1/F_p = 16.0$) o figuras alargadas como una elipse tienen un perímetro mayor en relación a su área y por ende menor $\rho$. Con el umbral de cátedra $\rho = 0.80$:
  1. En el **Punto A**, distingue los contornos circulares de las **monedas** ($\rho \in [0.806, 0.890]$) de los contornos cuasi-cuadrados de los **dados** ($\rho \in [0.659, 0.774]$).
  2. En el **Punto C**, distingue los puntos de la **cara superior** del dado (que se ven como **círculos** de frente a la cámara, $\rho \in [0.871, 0.900]$) de los puntos de las **caras laterales** (que por estar en planos oblicuos sufren escorzo proyectivo y se ven como **elipses alargadas**, $\rho \in [0.550, 0.590]$).

### Pregunta 3: *"En el Punto C de los dados, cuando umbralizan en forma inversa (`THRESH_BINARY_INV`), el fondo oscuro alrededor del dado también queda en blanco. ¿Cómo lo eliminan sin borrar los puntos del dado?"*
- **Respuesta:** Utilizando el algoritmo de **limpieza de bordes por reconstrucción morfológica (`imclearborder`)** de la Unidad 6 (`PI_U6_Morfologico_p2_Reconstruccion.py`). Construimos una imagen marcador que solo conserva los píxeles blancos del borde del recorte del dado y ejecutamos la reconstrucción geodésica $R_I(F)$ usando la imagen binarizada como máscara. Como el fondo oscuro exterior toca los bordes del recorte del dado pero los puntos negros están rodeados por el plástico blanco del cubo, la reconstrucción recupera únicamente el fondo exterior, y al restarlo (`cv2.subtract`) quedan aislados los puntos internos del dado.

### Pregunta 4: *"En el Problema 2 (Patentes), ¿qué hace exactamente la Transformada Top-Hat y cómo eligieron el tamaño $(21, 9)$ del elemento estructurante?"*
- **Respuesta:** La Transformada Top-Hat en niveles de gris se define como $I_{\text{tophat}} = I - (I \circ B)$, es decir, la resta entre la imagen original y su apertura morfológica (Unidad 6). La apertura morfológica $(I \circ B)$ elimina todos los picos brillantes cuyos trazos sean más angostos que el elemento estructurante $B$, conservando únicamente las grandes estructuras claras del fondo (como la carrocería blanca de un auto, el pavimento o el cielo). Al restar esa apertura de la imagen original, el fondo claro desaparece y resaltan exclusivamente las estructuras brillantes pequeñas sobre fondo oscuro. Elegimos un rectángulo de $21\times 9$ píxeles porque en las 12 imágenes el ancho de los caracteres de la patente varía entre $5$ y $13$ píxeles; al ser $21 > 13$, ningún carácter cabe dentro del elemento estructurante durante la apertura, garantizando que los 6 caracteres de la patente siempre produzcan un pico fuerte en la imagen Top-Hat.

### Pregunta 5: *"¿Por qué en el Punto B de las patentes no bastaba con aplicar un umbral de Otsu directo sobre el recorte de la patente para separar los 6 caracteres?"*
- **Respuesta:** Porque las patentes argentinas tienen un **marco blanco perimetral** pintado alrededor de los caracteres. En varias imágenes (como `img01`, `img03` e `img11`), al binarizar con un único umbral bajo como Otsu, el marco blanco queda unido por 1 o 2 píxeles con los extremos de las letras o números, fusionando varios caracteres con el marco en una sola componente conexa. Al aplicar nuevamente `imclearborder` (Unidad 6) sobre el Top-Hat del recorte —cuyo marco blanco toca el borde del recorte— y seleccionar el umbral que verifica la uniformidad de alturas de las 6 componentes conexas internas (`connectedComponentsWithStats`), el marco perimetral se elimina limpiamente y se segmentan los 72 caracteres individuales de las 12 imágenes sin un solo error.
