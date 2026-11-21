# Instructivo Narrativo y Guía de Defensa Oral — Trabajo Práctico N° 2
## Procesamiento de Imágenes I (`IA 4.4`) — TUIA — FCEIA — UNR (2026 - C2)

**Autores:** Agustín Marquardt, Fabián Alvarez  
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

### 4.4. Invarianza a Traslación, Rotación y Cambios de Escala (Problema 2)
* **Invarianza a Traslación:** Top-Hat (`cv2.MORPH_TOPHAT`), `cv2.connectedComponentsWithStats` y `imclearborder` operan de forma local o relativa a los centroides/cajas contenedoras, sin asumir ninguna posición fija en la imagen (`12/12` en Punto A y `12/12` en Punto B bajo desplazamiento).
* **Invarianza a Escala:**
  - El tamaño del elemento estructurante Top-Hat (`21x9` en `640×360`), los límites geométricos de Regla 1 (`h`, `w`, `area_min`) y las tolerancias de altura/colinealidad en Reglas 2 y 3 escalan proporcionalmente con `scale_f = min(H, W) / 360.0` y con la altura media `h_mean` de cada triplete.
  - Asimismo, las distancias inter-carácter (`dx1, dx2, dx_central`) y los márgenes del recorte (`pad_x, pad_y`) son adimensionales/proporcionales a `h_mean` y `dx_intra`. Esto permite detectar `12/12` patentes y `72/72` caracteres incluso duplicando la resolución (`2.0x` $\rightarrow$ `1280×720`). *(Por debajo de `0.75x`, caracteres originalmente de `5 px` de ancho se fusionan por submuestreo).*
* **Invarianza a Rotación:**
  - La Regla 2 tolera inclinaciones de patente de hasta `|dy| <= 0.35 * h_mean` ($\approx \pm 19^\circ$, como en `img04.png`) y la Regla 3 proyecta la colinealidad entre tripletes sobre la recta inclinada (`dy_esperado = dy1 * dx_central / dx_intra`).
  - Por diseño (`info_patentes.txt`), el descriptor `Alto/Ancho` en `[1.25, 3.5]` y el ordenamiento horizontal asumen vehículos en orientación normal (no patentes verticales a `90°`), logrando `11/12` a `-8°` y `8/12` a `+8°`.

---

## 5. Apuntes Detallados de Defensa — Problema 1 (`problema1_monedas_y_dados.py`)

### 5.1. `segmentar_objetos()` — Preguntas Finas de Diseño y Parámetros
1. **¿Por qué `cv2.medianBlur(gray, 11)` usa el parámetro `11`?**
   - El parámetro `ksize = 11` define una ventana cuadrada de $11\times 11$ píxeles (debe ser impar y $>1$).
   - En una imagen de alta resolución como `monedas.jpg` ($3871\times 2681\text{ px}$), las monedas tienen diámetros de $\sim 280\text{ a }380\text{ px}$, mientras que la rugosidad/grano de la mesa y el ruido impulsivo ocupan entre $3$ y $5\text{ px}$ de radio.
   - Una ventana chica (`3` o `5`) no alcanza a alisar la textura de la mesa, generando falsos gradientes en el fondo. Una ventana muy grande (ej. `25` o `31`) redondearía excesivamente las esquinas de los dados o borraría detalles finos de borde, además de ser computacionalmente más lenta. `11` es el punto justo: suaviza por completo el grano de la mesa preservando los bordes escalón bien nítidos (ventaja no lineal de la mediana frente al filtro Gaussiano).

2. **Justificación de $+0.9\,\sigma$ y cálculo dinámico de `k` (`grad_u8`):**
   - El histograma de `grad_u8` es **fuertemente asimétrico a la derecha** ($\mu = 6.03$, $\sigma = 15.89$): ~95% de los píxeles pertenecen al fondo o al interior liso de los objetos (gradiente $\approx 0$), y solo ~5% pertenece a bordes reales (cola larga hacia valores altos).
   - Por eso Otsu sobre `grad_u8` falla ($T = 65$, demasiado alto porque busca dos campanas separadas) y Triangle queda demasiado bajo ($T = 7$, mete ruido).
   - **Cálculo dinámico de `k`:** en lugar de fijar `0.9` a mano, el código barre `k` en `[0.5, 1.5]` (paso `0.1`), evalúa `T = round(mu + k * sigma)` seguido de clausura + `imfillhole` + apertura, y cuenta cuántas componentes cerradas válidas (`area >= AREA_MIN`) sobreviven. Elige el **mayor `k`** que alcanza el máximo número de objetos válidos (`k = 0.9 -> T = 20.0`, 19 objetos):
     - **Si `k` es menor** (ej. `k = 0.2 -> T = 9`): pasa ruido/textura de la mesa que fusiona objetos cercanos (baja a 9 componentes).
     - **Si `k` es mayor** (ej. `k = 2.5 -> T = 46`): los tramos de borde con menor contraste se cortan, dejando contornos abiertos que `imfillhole` no puede rellenar (baja a 9 componentes).

3. **¿Por qué en el paso 5 se usa `MORPH_ELLIPSE` y `(15, 15)`?**
   - **Forma `MORPH_ELLIPSE`:** los objetos principales son monedas circulares y los contornos tienen orientaciones en todas las direcciones ($0^\circ$ a $360^\circ$). Un elemento estructurante elíptico/circular es **isotrópico** (mismo radio en todas direcciones), evitando introducir sesgos direccionales o esquinas artificiales que sí produciría un `MORPH_RECT`.
   - **Tamaño `(15, 15)` (radio $\approx 7\text{ px}$):** la clausura (dilatación seguida de erosión) sirve para **puentear micro-cortes** de hasta $\sim 14\text{ px}$ en el anillo de borde antes de aplicar `imfillhole`. Con `(15, 15)` sella perfectamente esas brechas sin fusionar monedas vecinas (que están separadas por decenas de píxeles).

4. **¿Cómo funciona en detalle `imfillhole(img)`?**
   - Basado en reconstrucción morfológica geodésica (`PI_U6_Morfologico_p2_Reconstruccion.py`, U6):
     1. Se copia `img` en `img_flood_fill` y se crea una máscara de `cv2.floodFill` de tamaño `(H+2, W+2)`.
     2. Se lanza `floodFill` desde la semilla `(0, 0)` (garantizada en `0` porque anulamos un marco de `10 px` en `grad_u8`), pintando de blanco (`255`) **todo el fondo exterior conectado al borde** sin poder cruzar los anillos cerrados de los objetos.
     3. Al invertir (`cv2.bitwise_not(img_flood_fill)`), el fondo exterior pasa a `0` y únicamente el **interior de los anillos cerrados** queda en `255`.
     4. Haciendo el `OR` lógico (`img | img_flood_fill_inv`), se obtienen los objetos completamente rellenos conservando intacto su contorno original.

5. **¿Por qué en el paso 7 se usa `MORPH_ELLIPSE` y `(31, 31)`?**
   - La apertura (erosión seguida de dilatación) elimina salientes finos, pelusas o pequeños fragmentos que hayan quedado pegados por fuera de los objetos y suaviza el perímetro.
   - Usa `MORPH_ELLIPSE` por isotropía (respetar la geometría circular de las monedas).
   - Usa `(31, 31)` (radio $\approx 15\text{ px}$) porque los objetos reales tienen diámetros de $\sim 280\text{ a }380\text{ px}$, de modo que un disco de diámetro $31\text{ px}$ cabe holgadamente dentro de cualquier moneda o dado sin borrarlos, pero elimina cualquier espícula o resto externo de hasta $30\text{ px}$ de espesor.

6. **¿Cuáles son los 4 valores retornados por `cv2.connectedComponentsWithStats(bw_mask)` en el paso 8?**
   - `num_labels` (`int`): cantidad total de etiquetas asignadas, incluyendo el fondo (etiqueta `0`). Si hay 19 objetos, `num_labels = 20`.
   - `labels` (`ndarray` `H x W`, `int32`): mapa de etiquetas del mismo tamaño que `bw_mask`, donde cada píxel contiene el ID de la componente conexa a la que pertenece (`0` para fondo, `1..num_labels-1` para objetos).
   - `stats` (`ndarray` `num_labels x 5`, `int32`): tabla de estadísticas de cada etiqueta con 5 columnas:
     - `cv2.CC_STAT_LEFT` (col `0`): coordenada `x` mínima de la caja contenedora.
     - `cv2.CC_STAT_TOP` (col `1`): coordenada `y` mínima de la caja contenedora.
     - `cv2.CC_STAT_WIDTH` (col `2`): ancho de la caja contenedora.
     - `cv2.CC_STAT_HEIGHT` (col `3`): alto de la caja contenedora.
     - `cv2.CC_STAT_AREA` (col `4`): área total en píxeles de la componente.
   - `centroids` (`ndarray` `num_labels x 2`, `float64`): coordenadas `(cx, cy)` del centroide geométrico de cada etiqueta.

7. **¿De dónde sale `AREA_MIN` y cómo se calcula dinámicamente?**
   - Originalmente `AREA_MIN = 15000` se estimó observando que los objetos más chicos (monedas de 10 centavos) tienen área $\approx 62\,612\text{ px}^2$, mientras que cualquier resto residual es mucho menor.
   - Para hacerlo **independiente de la resolución**, se calcula dinámicamente como una fracción fija del área total de la imagen: `AREA_MIN = max(500.0, 0.0014 * H * W)`. En `monedas.jpg` ($3871\times 2681$), da $14\,529.4\text{ px}^2$ ($0.14\%$ de la imagen), descartando ruido pequeño incluso si la foto se reescala a otra resolución.

---

### 5.2. `clasificar_monedas()` — Umbrales Dinámicos de Área
- En lugar de fijar `80000` y `102000` manualmente, se implementa un **criterio de máxima separación en 1D**:
  1. Se ordenan las áreas de las monedas detectadas de menor a mayor (`areas_ord`).
  2. Se calculan las diferencias consecutivas entre áreas vecinas (`np.diff(areas_ord)`).
  3. Se identifican los **2 mayores saltos inter-clase** (`np.argsort(diffs)[-2:]`).
  4. Se colocan los umbrales exactamente en el **punto medio** de cada salto:
     - `UMBRAL_AREA_10C_1P = (areas_ord[i1] + areas_ord[i1+1]) / 2` $\rightarrow$ **$79\,743.8\text{ px}^2$** (salto entre $67\,606.0$ y $91\,881.5\text{ px}^2$).
     - `UMBRAL_AREA_1P_50C = (areas_ord[i2] + areas_ord[i2+1]) / 2` $\rightarrow$ **$102\,380.8\text{ px}^2$** (salto entre $97\,154.5$ y $107\,607.0\text{ px}^2$).
- De este modo, si se cambia la distancia de la cámara o la resolución de la foto, los 3 clústeres de tamaño siguen separándose automáticamente sin tocar constantes.

---

### 5.3. `analizar_dados()` — Umbral `90`, `RHO_PIP_TH` y `AREA_PIP_MIN` Dinámico
1. **¿Por qué `cv2.threshold(crop_gray, 90, 255, cv2.THRESH_BINARY_INV)` usa `90`?**
   - Dentro del recorte del dado hay tres regiones de intensidad bien diferenciadas:
     - **Puntos negros (pips):** cavidades pintadas de negro muy oscuro ($I < 70$).
     - **Plástico blanco del dado:** superficie clara ($I > 160$, percentil 75 $\approx 200$).
     - **Bisel/sombra del dado y mesa:** valores intermedios ($I \in [95, 150]$ en el Dado #2).
   - Si se usara Otsu local sobre el recorte ($T_{\text{Otsu}} \approx 125\text{–}127$), incluiría sombras del plástico lateral del dado. El valor `90` se ubica en el valle limpio entre la tinta negra ($<70$) y las sombras/plástico ($>100$), aislando exclusivamente los puntos negros (y el fondo oscuro exterior en el Dado #1, que luego elimina `imclearborder`).
2. **¿De dónde salen `RHO_PIP_TH` y `AREA_PIP_MIN` (y cómo hacerlos invariantes)?**
   - `RHO_PIP_TH = 0.80`: proviene del criterio de circularidad de la cátedra (`clasifico_celulas.py`, Unidad 7). Como $\rho = 4\pi A / P^2$ es **adimensional**, ya es naturalmente invariante a la resolución y escala. Separa perfectamente los puntos circulares de la cara superior ($\rho \in [0.871, 0.900]$) de las elipses alargadas de las caras laterales ($\rho \in [0.550, 0.590]$).
   - `AREA_PIP_MIN` y el filtro de área mínima de cara superior (`1200.0` original): se calculan **dinámicamente respecto del área del propio dado `d["area"]`**:
     - `area_pip_min = max(20.0, 0.0015 * d["area"])` ($\approx 100\text{–}112\text{ px}^2$ en `monedas.jpg`, filtra motas residuales).
     - `area_pip_top_min = 0.017 * d["area"]` ($\approx 1\,132\text{–}1\,266\text{ px}^2$ en `monedas.jpg`, separa los puntos superiores de $\sim 2.6\text{–}3.2\%$ del dado frente a los laterales de $\sim 0.85\%$, de forma completamente independiente de la resolución de la imagen).

---

### 5.4. Invarianza a Traslación, Rotación y Cambios de Escala
* **Invarianza a Traslación:**
  - Todas las operaciones espaciales (`medianBlur`, `Sobel`, `MORPH_CLOSE`, `MORPH_OPEN`) son **convoluciones/filtros locales invariantes al desplazamiento**.
  - `imfillhole` parte de `(0, 0)` sobre un marco perimetral anulado, por lo que rellena cualquier objeto sin importar en qué posición `(x, y)` de la mesa se encuentre (salvo que toque el borde de la foto).
  - Los descriptores `Area`, `Perímetro`, `Fp` y `rho` dependen únicamente de la geometría del contorno, no de su centroide `(cx, cy)`.
* **Invarianza a Rotación:**
  - La magnitud euclídea del gradiente $\|\nabla I\| = \sqrt{G_x^2 + G_y^2}$ es **isotrópica** (invariante a la orientación del borde).
  - Todos los elementos estructurantes morfológicos son `MORPH_ELLIPSE` (discos circulares isotrópicos), evitando sesgos direccionales.
  - `Area`, `Perímetro`, `Fp` y `rho = 4πA/P²` son **invariantes a rotación**.
  - En `analizar_dados()`, `imclearborder` sobre el recorte `bbox` elimina el fondo exterior cualquiera sea el ángulo del dado.
* **Invarianza a Cambios de Escala (Resolución):**
  - Todos los tamaños de kernel y umbrales de área se calculan **proporcionalmente** a la dimensión de la imagen (`dim_min = min(H, W)`) o del objeto (`d["area"]`):
    - `k_med`, `pad_borde`, `k_close`, `k_open` escalan con `dim_min` (dando `11`, `10`, `15`, `31` en `3871×2681`).
    - `AREA_MIN = 0.0014 * H * W` escala con el área total de la imagen.
    - `k_opt` se selecciona dinámicamente mediante estabilidad morfológica sobre el gradiente.
    - `UMBRAL_AREA_10C_1P` y `UMBRAL_AREA_1P_50C` se calculan dinámicamente por **máxima separación 1D** entre las áreas detectadas.
    - En `analizar_dados()`, `k_close_pip`, `k_open_pip`, `area_pip_min` y `area_pip_top_min` escalan con `d["area"]` del propio dado, y `RHO_PIP_TH = 0.80` es adimensional.
