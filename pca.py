import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def centrar(X):
    """
    Paso 2: Calcula la imagen media y centra los datos.
    X: Matriz de entrenamiento de dimensiones (m, p)
    """
    mu = np.mean(X, axis=0)  # Media de cada columna (píxel)
    Xc = X - mu              # Matriz centrada
    return Xc, mu

def gram(Xc):
    """
    Paso 3: Calcula la matriz de Gram G = (Xc * Xc^T) / (m-1)
    Xc: Matriz centrada de dimensiones (m, p)
    """
    m = Xc.shape[0]
    G = (Xc @ Xc.T) / (m - 1)
    return G

import numpy as np

def espectro(Xc):
    """
    Calcula los valores y vectores propios utilizando la matriz de Gram G = Xc @ Xc.T
    o la covarianza, asegurando ordenamiento descendente y limpieza numérica.
    """
    # 1. Calcular la matriz de Gram (o covarianza reducida)
    # Si usas Gram: G = Xc @ Xc.T
    G = Xc @ Xc.T
    
    # 2. Obtener valores y vectores propios con eigh (devuelve en orden ASCENDENTE)
    lam, V_gram = np.linalg.eigh(G)
    
    # 3. Limpiar errores numéricos (convertir valores negativos microscópicos a 0)
    lam[lam < 0] = 0.0
    
    # 4. Ordenar de MAYOR a MENOR (invertir el arreglo con [::-1])
    idx = np.argsort(lam)[::-1]
    lam = lam[idx]
    V_gram = V_gram[:, idx]
    
    # 5. Recuperar los vectores propios originales (V) proyectando hacia el espacio de los datos
    # V = Xc.T @ V_gram (normalizados)
    V = Xc.T @ V_gram
    
    # Normalizar los vectores propios resultantes para que tengan norma 1
    for i in range(V.shape[1]):
        norma = np.linalg.norm(V[:, i])
        if norma > 0:
            V[:, i] /= norma
            
    return lam, V

def varianza_acumulada(lam):
    """Paso 6: Calcula el porcentaje de varianza acumulada."""
    return np.cumsum(lam) / np.sum(lam) * 100

def proyectar(x, mu, V, k):
    """Paso 7: Proyecta una imagen al subespacio PCA."""
    return V[:, :k].T @ (x - mu)

def reconstruir(t, V, mu, k):
    """Paso 7: Reconstruye la imagen desde el espacio PCA."""
    return (V[:, :k] @ t) + mu

def identificar(t, T, etiquetas, k, umbral=25.0):
    """Paso 8: Calcula distancias y asigna la clase del vecino más cercano."""
    # T tiene los scores de entrenamiento (m, k). t es la prueba (k,)
    distancias = np.linalg.norm(T[:, :k] - t, axis=1)
    idx_min = np.argmin(distancias)
    d_min = distancias[idx_min]
    
    if d_min > umbral:
        return "Desconocido", d_min
    return etiquetas[idx_min], d_min

def calcular_error_reconstruccion(x_prueba, t, V, mu, k):
    """Calcula la norma de la diferencia entre la imagen original y la reconstruida."""
    # Reconstruimos la imagen usando solo k componentes
    x_reconstruida = (V[:, :k] @ t) + mu
    # El error es la distancia euclidiana entre la original y la reconstruida
    error = np.linalg.norm(x_prueba - x_reconstruida)
    return float(error)

def obtener_imagen_reconstruida(x_prueba, V, mu, k):
    """Reconstruye la imagen y la convierte en un objeto PIL Image de 64x64."""
    # Cálculo de la reconstrucción: x_hat = V_k * t + mu
    t = V[:, :k].T @ (x_prueba - mu)
    x_reconstruida = (V[:, :k] @ t) + mu
    
    # Normalizar los valores al rango [0, 255] para que la imagen se vea bien
    x_reconstruida = np.clip(x_reconstruida, 0, 1) * 255.0
    img_array = x_reconstruida.reshape(64, 64).astype(np.uint8)
    
    return Image.fromarray(img_array)

def graficar_matriz_gram(G, ruta_salida="static/matriz_gram.png"):
    """
    Genera y guarda un mapa de calor (heatmap) de la matriz de Gram G (m x m).
    """
    plt.figure(figsize=(8, 6))
    
    # Creamos el mapa de calor con seaborn
    # cmap="viridis" o "coolwarm" le dan un estilo muy profesional de ingeniería
    sns.heatmap(G, cmap="coolwarm", cbar=True, square=True, 
                xticklabels=False, yticklabels=False)
    
    plt.title("Mapa de Calor - Matriz de Gram G ($X_c X_c^T$)", fontsize=12)
    plt.xlabel("Imágenes de entrenamiento ($m$)", fontsize=10)
    plt.ylabel("Imágenes de entrenamiento ($m$)", fontsize=10)
    
    # Guardar la imagen
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300)
    plt.close()