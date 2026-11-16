# Instructivo Narrativo y Guía de Defensa Oral — Trabajo Práctico N° 2
## Procesamiento de Imágenes I (`IA 4.4`) — TUIA — FCEIA — UNR (2026 - C2)

**Autores:** Agustín Marquardt, Damián Turco, Fabián Alvarez  
**Archivos de código explicados:**
- [`problema1_monedas_y_dados.py`](file:///Users/agustinmarquardt/Documents/TUIA/pdi/TP2_2026-C2/problema1_monedas_y_dados.py)
- [`problema2_deteccion_patentes.py`](file:///Users/agustinmarquardt/Documents/TUIA/pdi/TP2_2026-C2/problema2_deteccion_patentes.py)

---

> **Cómo usar este documento:**  
> Está pensado en **modo narrativa**, paso a paso y método por método, para que puedas leerlo de corrido y explicarle fácilmente a un compañero de grupo (o al tribunal en la defensa oral) **qué hace el código, en qué orden se ejecuta, por qué se eligió cada parámetro y qué concepto teórico de las Unidades 1 a 7 lo respalda**.

---

## 1. Visión General y Mapa Teórico (Unidades 1 a 7)

En este TP2 resolvemos dos problemas reales de Visión por Computadora sin usar redes neuronales ni modelos externos, apoyándonos únicamente en técnicas clásicas de **Unidades 1 a 7**:

| Unidad | Tema Teórico de la Materia | Cómo y Dónde se Aplica en el TP2 |
| :--- | :--- | :--- |
| **Unidad 1 y 2** | Conversión a luminancia (`BGR2GRAY`) y **Filtrado Espacial No Lineal (Mediana)** (`cv2.medianBlur`). | **Problema 1 (Punto A):** Antes de derivar la imagen, aplicamos mediana $11\times 11$ para limpiar la textura granulada de la mesa sin redondear los bordes de monedas y dados. |
| **Unidad 3** | **Gradiente de Sobel** (`cv2.Sobel`), **Umbralado de Otsu** (`THRESH_OTSU`), **Componentes Conectadas** (`cv2.connectedComponentsWithStats`) y **Contornos** (`cv2.findContours`). | **Problema 1 (Punto A):** El gradiente de Sobel elimina la rampa de iluminación del fondo.<br>**Problema 2 (Puntos A y B):** `connectedComponentsWithStats` etiqueta cada región blanca candidata y devuelve su caja $(x,y,w,h)$, área y centroide $(c_x, c_y)$. |
| **Unidad 6** | **Morfología Binaria** (Apertura, Clausura), **Reconstrucción Geodésica** (`imreconstruct`, `imfillhole`, `imclearborder`) y **Transformada Top-Hat en Grises** (`cv2.MORPH_TOPHAT`). | **Problema 1:** Clausura $15\times 15$, `imfillhole` y Apertura $31\times 31$ en el Punto A; `imclearborder` en el Punto C para borrar el fondo externo del recorte de cada dado.<br>**Problema 2:** Top-Hat $21\times 9$ para resaltar letras blancas sobre fondo oscuro e `imclearborder` para eliminar el marco blanco de la patente en el Punto B. |
| **Unidad 7** | **Descriptores Regionales:** Área ($A$), Perímetro ($P$), **Factor de Forma** ($F_p = A/P^2$, $\rho = 4\pi F_p$), **Relación de Aspecto** ($H/W$) y **Momentos**. | **Problema 1:** $\rho > 0.80$ separa Monedas de Dados (Punto A) y puntos de la cara superior de puntos laterales (Punto C); el Área clasifica las monedas en el Punto B.<br>**Problema 2:** $H/W \in [1.25, 3.5]$ y las reglas estructurales de 2 grupos de 3 caracteres aíslan la patente sin falsos positivos. |

---

## 2. Operadores Morfológicos Compartidos (Unidad 6)

Antes de entrar a cada problema, ambos scripts comparten el mismo núcleo de **Reconstrucción Morfológica Binaria** tomado del apunte práctico de la Unidad 6 (`PI_U6_Morfologico_p2_Reconstruccion.py`). Si le explicás esto primero a tu compañero, después entiende al instante por qué desaparecen los huecos y los bordes:

### 2.1. `imreconstruct(marker, mask, kernel=None)`
* **Qué recibe:** Una imagen binaria `marker` (semilla inicial) y una imagen binaria `mask` (límite máximo donde puede crecer).
* **Lógica paso a paso:**
  1. Parte de `marker` y lo dilata un paso con un kernel de $3\times 3$ (`cv2.dilate(marker, kernel)`).
  2. Lo intersecta con `mask` (`cv2.bitwise_and` o `cv2.min`), de modo que **solo puede crecer por encima de píxeles blancos que existan en la máscara**.
  3. Repite el bucle `while` hasta que la imagen ya no cambia (`(marker == expanded_intersection).all()`).
* **Por qué es útil (Teoría U6):** Es una **dilatación geodésica iterativa hasta convergencia**. A diferencia de una dilatación común, no engorda los objetos: simplemente "rescata" de `mask` todas las componentes completas que hayan sido tocadas por al menos un píxel de `marker`.

### 2.2. `imfillhole(img)`
* **Qué hace:** Rellena todos los huecos negros encerrados dentro de objetos blancos.
* **Lógica paso a paso:**
  1. Copia la imagen binaria y ejecuta `cv2.floodFill(img_flood_fill, mask, (0, 0), 255)` empezando desde la esquina superior izquierda `(0, 0)`. Eso pinta de blanco todo el fondo exterior que tiene camino libre hasta el borde, pero **no puede entrar** dentro de contornos cerrados.
  2. Invierte esa inundación (`cv2.bitwise_not`): ahora solo quedan en blanco los huecos internos.
  3. Hace un OR lógico (`img | img_flood_fill_inv`) y devuelve los objetos completamente sólidos.

### 2.3. `imclearborder(img)` (o `imclearborder(f, conn=8)` en el Problema 2)
* **Qué hace:** Borra cualquier objeto blanco que esté tocando alguno de los 4 bordes de la imagen, dejando intactos los objetos del interior.
* **Lógica paso a paso:**
  1. Crea una imagen `marker` igual a `img`, pero pone en `0` todo el interior (`marker[1:-1, 1:-1] = 0`), dejando activos únicamente los píxeles blancos que tocan el marco perimetral.
  2. Llama a `border_objs = imreconstruct(marker, mask=img)`. Esto reconstruye íntegramente todos los objetos que tocan el borde.
  3. Resta esos objetos de la imagen original: `cv2.subtract(img, border_objs)`.

---

## 3. Narrativa del PROBLEMA 1: Monedas y Dados (`problema1_monedas_y_dados.py`)

### Flujo General de Ejecución (`main()`)
Cuando ejecutamos `python problema1_monedas_y_dados.py`, el programa:
1. Carga `monedas.jpg` ($3871 \times 2681\text{ px}$).
2. Llama a **`segmentar_objetos(img_bgr)`** (Punto A) $\rightarrow$ devuelve los 19 objetos separados en `monedas` (17) y `dados` (2).
3. Llama a **`clasificar_monedas(monedas)`** (Punto B) $\rightarrow$ clasifica las 17 monedas en 10 ctvs, 1 Peso y 50 ctvs, y suma el monto total.
4. Llama a **`analizar_dados(img_bgr, etapas["gray"], dados)`** (Punto C) $\rightarrow$ determina qué número marca la cara superior de cada dado.
5. Imprime el reporte tabular por consola y llama a **`generar_figuras_problema1(...)`** para guardar las 3 figuras de diagnóstico en `resultados_problema1/`.

---

### Método 1 (Punto A): `segmentar_objetos(img_bgr)`
**El desafío:** La foto tiene un fondo gris cuya intensidad cambia muchísimo de izquierda ($I \approx 60$) a derecha ($I \approx 185$). Si aplicamos un umbral global como **Otsu** directamente (`T = 115`), toda la mitad derecha del fondo supera ese valor y se vuelve blanca, pegándose con las monedas. *(El script calcula `bw_otsu_global` justamente para graficar esa falla en la Figura 1).*

**Cómo lo resuelve paso a paso:**
1. **Grises + Mediana (`cv2.cvtColor` y `cv2.medianBlur(gray, 11)` — U1 y U2):**  
   Pasamos a escala de grises y aplicamos un filtro de mediana de $11\times 11$. Usamos mediana (no Gaussiano) porque suaviza la textura rugosa de la mesa manteniendo el borde escalón de las monedas bien filoso.
2. **Gradiente de Sobel (`cv2.Sobel` con `cv2.CV_64F` — U3):**  
   Calculamos $G_x$ y $G_y$ con kernels $3\times 3$ en punto flotante de 64 bits (fundamental para no perder derivadas negativas cuando la intensidad baja) y obtenemos la magnitud $\|\nabla I\| = \sqrt{G_x^2 + G_y^2}$.  
   *¿Por qué esto elimina el fondo no uniforme?* Porque la iluminación del fondo cambia muy lento (baja frecuencia espacial), así que su derivada vale casi $0$. En cambio, el borde de cada moneda y dado es un salto brusco (alta frecuencia) que da un gradiente bien alto.  
   Luego anulamos un marco de $10\text{ px}$ alrededor de `grad_u8` para asegurar que el `(0, 0)` sea negro puro antes de usar `imfillhole`.
3. **Umbralado estadístico del gradiente (`U3`):**  
   Calculamos `th_grad = np.round(grad_u8.mean() + 0.9 * grad_u8.std())` (que da `T = 20.0`) y binarizamos con `cv2.threshold`. Así obtenemos los anillos blancos de contorno de los 19 objetos.
4. **Clausura + `imfillhole` + Apertura (`U6`):**  
   - **Clausura (`MORPH_CLOSE`, elipse $15\times 15$):** Sella cualquier micro-brecha en el anillo del borde para que no se escape el `floodFill`.
   - **Rellenado (`imfillhole`):** Rellena el interior de los 19 contornos cerrados, tapando los relieves internos de las monedas bimetálicas y los puntos de los dados.
   - **Apertura (`MORPH_OPEN`, elipse $31\times 31$):** Elimina pelusas o pequeñas irregularidades externas y deja bordes suaves.
5. **Etiquetado y Separación por Factor de Forma (`U3` y `U7`):**  
   Llamamos a `cv2.connectedComponentsWithStats(bw_mask)` y descartamos restos chicos con `AREA_MIN = 15000`. Para cada una de las 19 componentes válidas, extraemos su contorno externo (`cv2.findContours`), medimos **Área** ($A$) y **Perímetro** ($P$), y calculamos el **Factor de Forma** de la cátedra (`Factor de Forma.txt` y `clasifico_celulas.py`):
   $$F_p = \frac{A}{P^2}, \qquad \rho = 4\pi F_p = \frac{4\pi A}{P^2}$$
   - Círculo ideal: $\rho = 1.00$ ($1/F_p = 12.57$).
   - Cuadrado ideal: $\rho = \pi/4 \approx 0.785$ ($1/F_p = 16.00$).
   - Con el umbral de cátedra **`RHO_TH = 0.80`**, `es_moneda = rho > 0.80` separa con 100% de exactitud las **17 Monedas** ($\rho \in [0.8056, 0.8904]$) de los **2 Dados** ($\rho = 0.7742$ y $0.6590$).

---

### Método 2 (Punto B): `clasificar_monedas(monedas)`
* **Lógica:** Como la cámara está a una altura fija respecto de la mesa, el **Área en píxeles²** (`m["area"]`) refleja directamente el tamaño físico de cada denominación argentina:
  - **10 Centavos** ($19.0\text{ mm}$): áreas entre $62\,612$ y $67\,606\text{ px}^2$ $\rightarrow$ **9 unidades** ($\$0.90$).
  - **1 Peso** ($23.0\text{ mm}$): áreas entre $91\,882$ y $97\,154\text{ px}^2$ $\rightarrow$ **5 unidades** ($\$5.00$).
  - **50 Centavos** ($25.2\text{ mm}$): áreas entre $107\,607$ y $110\,049\text{ px}^2$ $\rightarrow$ **3 unidades** ($\$1.50$).
* **Fronteras de decisión:** `UMBRAL_AREA_10C_1P = 80000.0` y `UMBRAL_AREA_1P_50C = 102000.0` (con más de $24\,000\text{ px}^2$ de separación libre entre clases).
* **Resultado:** **17 monedas**, **Monto Total = $\$7.40$**.

---

### Método 3 (Punto C): `analizar_dados(img_bgr, gray, dados)`
**El desafío:** Los dados están en perspectiva. En el **Dado #1** se ven los **5 puntos de la cara superior**, pero también se ven **3 puntos oscuros en la cara lateral derecha**.

**Cómo separa la cara superior de las laterales paso a paso:**
Para cada dado (ordenado de izquierda a derecha por `bbox[0]`):
1. **Recorte ROI y Umbralado Inverso (`T = 90`, `THRESH_BINARY_INV`):**  
   Recortamos la caja contenedora `bbox` del dado en escala de grises. Como el dado es blanco ($I > 160$) y los puntos (y la mesa exterior) son oscuros ($I < 90$), al umbralizar invertido quedan blancos (`255`) tanto los puntos internos como el fondo exterior que rodea al dado.
2. **Limpieza de Borde (`imclearborder(pips_bw)` — U6):**  
   Como el fondo exterior oscuro toca los bordes del recorte mientras que los puntos del dado están rodeados por plástico blanco, `imclearborder` borra el fondo exterior de una sola vez.
3. **Clausura ($9\times 9$) + Apertura ($5\times 5$) (`U6`):**  
   Cada punto negro del dado tiene un pequeño reflejo de luz blanca en el centro. La clausura cierra ese brillo interno y la apertura limpia motas aisladas.
4. **Discriminación por Factor de Forma ($\rho > 0.80$ y $A > 1200\text{ px}^2$ — U7):**  
   Extraemos los contornos de cada punto (`cv2.findContours`) y medimos su circularidad $\rho = 4\pi A/P^2$:
   - **Puntos de la cara superior:** Están casi paralelos a la cámara, así que se ven como **círculos** ($\rho \in [0.871, 0.900] > 0.80$, Área $\approx 1950 - 2127\text{ px}^2$).
   - **Puntos de las caras laterales:** Están inclinados por perspectiva (escorzo), así que se ven como **elipses muy alargadas y chicas** ($\rho \in [0.550, 0.590] \le 0.80$, Área $\approx 622 - 649\text{ px}^2$).
* **Resultado:** **Dado #1 = 5** (descarta los 3 laterales), **Dado #2 = 3** (0 laterales), **Suma Total = 8**.

---

## 4. Narrativa del PROBLEMA 2: Patentes (`problema2_deteccion_patentes.py`)

### Flujo General de Ejecución (`main()`)
El script recorre en bucle las 12 imágenes (`img01.png` a `img12.png`, todas de $640 \times 360\text{ px}$). Para cada imagen ejecuta dos funciones independientes y secuenciales:
1. **`res_A = detectar_y_segmentar_patente(img_bgr)`** (Punto A) $\rightarrow$ detecta la patente en la escena completa y devuelve el recorte `recorte_bgr`.
2. **`res_B = segmentar_caracteres_patente(res_A["recorte_bgr"])`** (Punto B) $\rightarrow$ toma **únicamente** ese recorte y segmenta los 6 caracteres individuales (`C1` a `C6`).

---

### Punto A — Detección de la Patente (3 Métodos Encadenados)

#### 4.1. Método principal: `detectar_y_segmentar_patente(img_bgr)`
1. **Escala de grises + Transformada Top-Hat (`cv2.MORPH_TOPHAT`, EE Rectangular $21\times 9$ — U6):**
   $$I_{\text{tophat}} = I - (I \circ B)$$
   *¿Por qué Top-Hat y no un umbral común?* Porque un auto blanco o plateado, el cielo o la calle brillan tanto como las letras de la patente. En la apertura $(I \circ B)$, como el rectángulo $B$ mide $21\text{ px}$ de ancho y las letras miden entre $5$ y $13\text{ px}$ de ancho, la apertura **borra** los trazos de las letras pero **conserva** las grandes superficies claras (capó, calle, cielo). Al restar $I - (I \circ B)$, todo el fondo claro se apaga y solo quedan encendidos los trazos brillantes pequeños sobre fondo oscuro (como las 6 letras blancas sobre la chapa negra).
2. **Barrido de umbral multiescala (`range(35, 165, 5)`):**  
   Como hay patentes al sol (umbral óptimo `T=140` en `img01`) y patentes en sombra (`T=45` en `img06`), probamos umbrales de $35$ a $160$ de a $5$. En cada umbral llamamos a los dos métodos de filtrado geométrico:

#### 4.2. Método de Regla 1: `filtrar_candidatos_caracter(bw_mask)` (`U3` y `U7`)
Llama a `cv2.connectedComponentsWithStats(bw_mask, connectivity=8)` y aplica la **Regla 1 de `info_patentes.txt`** (*relación Alto/Ancho ~ 1.5 - 3.0*):
- Conserva solo componentes con $9 \le h \le 30\text{ px}$, $3 \le w \le 20\text{ px}$, $\text{area} \ge 15$ y relación de aspecto $1.25 \le h/w \le 3.5$ (el rango amplio contempla inclinación y letras anchas o finas como `1`, `M`, `Q`). Esto reduce cientos de manchas de ruido a apenas **7 a 12 candidatos** por foto.

#### 4.3. Método de Reglas 2 y 3: `buscar_constelacion_patente(candidatos)` (`U7`)
Ordena los candidatos de izquierda a derecha por `x` y busca la estructura exacta de una patente argentina (`AAA 000`):
1. **Regla 2 (Tripletes — grupos de 3 caracteres):** Busca ternas $(c_1, c_2, c_3)$ donde:
   - Los 3 caracteres tengan altura pareja (`max|h - h_mean| <= 3.5`).
   - El espaciado horizontal entre centroides $\Delta x_1, \Delta x_2$ esté entre $0.40\,\bar{h}$ y $1.15\,\bar{h}$ y sea regular (`|dx1 - dx2| <= 0.35 * h_mean`).
   - Estén alineados casi horizontalmente (`|dy1|, |dy2| <= 0.35 * h_mean`, tolerando patentes inclinadas como `img04`).
2. **Regla 3 (Dos grupos de 3 caracteres):** Aparea dos tripletes $T_1$ y $T_2$ que tengan altura similar (`|h1 - h2| <= 3.5`), un **espacio central mayor** entre la 3ª letra y el 1º dígito (`1.25 * dx_intra <= dx_central <= 2.80 * dx_intra`), y que sigan la misma recta inclinada (`|dy_central - dy_esperado| <= 4.0`).
3. **Selección por puntaje y Recorte:** Elige el grupo de 6 que maximiza `suma_areas - 10 * std(alturas)` y calcula la caja contenedora agregando un margen proporcional (`0.45 * hm` en horizontal y `0.35 * hm` en vertical) para encerrar la placa completa en `recorte_bgr`. **Efectividad: 12/12 (100%).**

---

### Punto B — Segmentación de Caracteres Individuales (`segmentar_caracteres_patente(recorte_bgr)`)

**El desafío:** Si tomamos el recorte `recorte_bgr` y lo binarizamos con un umbral simple (como Otsu directo), el **marco blanco perimetral** de la patente argentina toca los bordes de las letras por 1 píxel y fusiona varios caracteres en una sola mancha.

**Cómo lo resuelve paso a paso:**
1. **Top-Hat Local (`21x9`) sobre `plate_gray` (`U6`):** Uniformiza sombras oblicuas sobre la chapa y atenúa las barras horizontales del marco.
2. **Barrido de umbral + `imclearborder(bw_raw, conn=8)` (`U6`):**  
   Este es el paso clave: como el marco blanco perimetral toca los bordes del recorte mientras que los 6 caracteres están en el centro, `imclearborder` elimina por completo el marco blanco sin dañar ninguna letra.
3. **Validación de las 6 Componentes Conexas (`connectedComponentsWithStats` — U3 y U7):**  
   Para cada umbral `th in range(30, 180, 5)`, filtra las componentes internas que miden entre el $35\%$ y el $92\%$ del alto del recorte (`0.35 * hc <= h <= 0.92 * hc`), con `1.25 <= h/w <= 3.5` y `area >= 15`. Selecciona el umbral que produce **exactamente 6 componentes** con alturas uniformes (`max(hs) - min(hs) <= 4`) y máximo puntaje `sum(areas) - 10 * std(hs)`.
4. **Ordenamiento Izquierda-Derecha y Extracción (`C1`..`C6`):**  
   Ordena las 6 componentes por `x` creciente (`C1, C2, C3` = Letras; `C4, C5, C6` = Dígitos) y guarda tanto la máscara limpia como el recorte de cada carácter individual. **Efectividad: 72/72 caracteres segmentados (100%).**
