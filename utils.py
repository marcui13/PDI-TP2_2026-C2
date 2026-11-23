"""
Utilidades morfológicas y auxiliares compartidas para el Trabajo Práctico N° 2.

Centraliza las operaciones de reconstrucción morfológica (Unidad 6) y funciones
auxiliares reutilizadas por `problema1_monedas_y_dados.py` y
`problema2_deteccion_patentes.py`.
"""

import cv2
import numpy as np


def imreconstruct(marker, mask, kernel=None):
    """
    Reconstrucción morfológica por dilatación geodésica iterativa R_mask(marker)
    hasta convergencia (Unidad 6 - PI_U6_Morfologico_p2_Reconstruccion.py).

    Usa `cv2.min` como restricción geodésica, por lo que funciona tanto para
    imágenes binarias (0 / 255) como en escala de grises.
    """
    if kernel is None:
        kernel = np.ones((3, 3), np.uint8)
    curr = marker.copy()
    while True:
        expanded = cv2.dilate(curr, kernel)
        next_state = cv2.min(expanded, mask)
        if np.array_equal(curr, next_state):
            break
        curr = next_state
    return curr


def imfillhole(img):
    """
    Rellenado automático de huecos internos en una imagen binaria (0 / 255).
    Basado en PI_U6_Morfologico_p2_Reconstruccion.py (Unidad 6).
    """
    img_flood_fill = img.copy().astype(np.uint8)
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(img_flood_fill, mask, (0, 0), 255)
    img_flood_fill_inv = cv2.bitwise_not(img_flood_fill)
    return img | img_flood_fill_inv


def imclearborder(img, conn=8):
    """
    Elimina componentes conexas brillantes que tocan el borde de la imagen
    mediante reconstrucción morfológica desde el marco perimetral (Unidad 6).

    Parámetros:
        img: Imagen binaria (0 / 255) o en escala de grises.
        conn: Conectividad del elemento estructurante 3x3 (8 o 4).
    """
    marker = np.zeros_like(img)
    marker[0, :] = img[0, :]
    marker[-1, :] = img[-1, :]
    marker[:, 0] = img[:, 0]
    marker[:, -1] = img[:, -1]
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT if conn == 8 else cv2.MORPH_CROSS, (3, 3)
    )
    border_elements = imreconstruct(marker=marker, mask=img, kernel=kernel)
    return cv2.subtract(img, border_elements)


def impar(val, minimo=3):
    """Redondea `val` al entero impar más cercano mayor o igual a `minimo`."""
    v = max(minimo, int(round(val)))
    return v if v % 2 == 1 else v + 1
