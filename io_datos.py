import os
import numpy as np
from PIL import Image

def cargar_entrenamiento(ruta_base="datos/entrenamiento", size=(64, 64)):
    """
    Paso 1: Carga las imágenes, las convierte a escala de grises,
    las redimensiona y las aplana en una matriz X.
    """
    imagenes_lista = []
    y_labels = []
    nombres_clases = {}
    
    # Leer las subcarpetas dentro de datos/entrenamiento/
    if not os.path.exists(ruta_base):
        return None, None, None
        
    subcarpetas = sorted([d for d in os.listdir(ruta_base) if os.path.isdir(os.path.join(ruta_base, d))])
    
    for idx, nombre_persona in enumerate(subcarpetas):
        nombres_clases[idx] = nombre_persona
        ruta_persona = os.path.join(ruta_base, nombre_persona)
        
        for filename in os.listdir(ruta_persona):
            if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                img_path = os.path.join(ruta_persona, filename)
                try:
                    # Convertir a escala de grises ('L'), redimensionar y normalizar a [0, 1]
                    img = Image.open(img_path).convert('L').resize(size)
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    
                    # Aplanar la imagen en un vector de p píxeles
                    imagenes_lista.append(img_array.flatten())
                    y_labels.append(idx)
                except Exception as e:
                    pass
                    
    X = np.array(imagenes_lista)
    return X, np.array(y_labels), nombres_clases