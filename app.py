from flask import Flask, render_template, request, jsonify
import numpy as np
import pca
import io_datos

app = Flask(__name__)
# Límite de 5MB para las fotos que se suban como pide la rúbrica
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024 

# Variables globales para almacenar el modelo entrenado
X, y_labels, nombres_clases = None, None, None
Xc, mu, lam, V = None, None, None, None

@app.before_first_request
def inicializar_modelo():
    """Ejecuta el entrenamiento PCA una sola vez al iniciar el servidor."""
    global X, y_labels, nombres_clases, Xc, mu, lam, V
    X, y_labels, nombres_clases = io_datos.cargar_entrenamiento()
    
    if X is not None and len(X) > 0:
        # Paso 2, 3 y 4: Calcular variables usando tu archivo pca.py
        Xc, mu = pca.centrar(X)
        lam, V = pca.espectro(Xc)

@app.route("/")
def index():
    """Ruta principal que muestra la página web."""
    # Aquí pasaremos variables a la plantilla HTML para que escribas tus "notas"
    return render_template("index.html")

@app.route("/identify", methods=["POST"])
def identify():
    """Paso 7 y 8: Recibe la foto de prueba y la identifica."""
    # (El código para procesar la foto y comparar irá aquí)
    pass

@app.route("/reconstruct")
def reconstruct():
    """Actualiza la reconstrucción cuando muevas el deslizador de k."""
    k = int(request.args.get('k', 1))
    # (El código de reconstrucción irá aquí)
    pass

if __name__ == "__main__":
    # Nunca ejecutar con debug=True fuera del equipo local según la rúbrica
    app.run(debug=False, port=5000)
