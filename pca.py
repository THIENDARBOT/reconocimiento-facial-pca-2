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

def espectro(Xc):
    """
    Paso 4: Obtiene los valores y vectores propios de G, y recupera los de C.
    """
    G = gram(Xc)
    
    # eigh es la función correcta para matrices simétricas como G
    lam, U = np.linalg.eigh(G)
    
    # Ordenar de mayor a menor
    indices = np.argsort(lam)[::-1]
    lam = lam[indices]
    U = U[:, indices]
    
    # Descartar valores propios nulos o negativos por error numérico
    mascara = lam > 1e-8
    lam = lam[mascara]
    U = U[:, mascara]
    
    # Recuperar los vectores propios V de la matriz de covarianza real
    # V = Xc^T * U, y luego se normalizan sus columnas para que V^T * V = I
    V = Xc.T @ U
    V = V / np.linalg.norm(V, axis=0)
    
    return lam, V
