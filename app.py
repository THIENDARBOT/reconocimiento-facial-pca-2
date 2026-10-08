from flask import Flask, render_template, request, jsonify
import numpy as np
import pca
import io_datos
import io
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
# Límite de 5MB para las fotos que se suban como pide la rúbrica
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024 

# Variables globales para almacenar el modelo entrenado y la prueba actual
X, y_labels, nombres_clases = None, None, None
Xc, mu, lam, V = None, None, None, None
T_entrenamiento = None 
x_prueba_actual = None

def inicializar_modelo():
    """Ejecuta el entrenamiento PCA una sola vez al iniciar el servidor."""
    global X, y_labels, nombres_clases, Xc, mu, lam, V, T_entrenamiento
    X, y_labels, nombres_clases = io_datos.cargar_entrenamiento()
    
    if X is not None and len(X) > 0:
        # Paso 2, 3 y 4: Calcular variables usando tu archivo pca.py
        Xc, mu = pca.centrar(X)
        lam, V = pca.espectro(Xc)
        # Paso 5: Calcular scores de entrenamiento globales
        T_entrenamiento = Xc @ V
        print(f"Modelo cargado correctamente. Clases: {list(nombres_clases.values())}")
    else:
        print("Advertencia: No se encontraron imágenes de entrenamiento en la carpeta.")

# Llamamos a la inicialización manualmente antes de arrancar las rutas
with app.app_context():
    inicializar_modelo()

@app.route("/")
def index():
    """Ruta principal que muestra la página web."""
    return render_template("index.html")

@app.route("/identify", methods=["POST"])
def identify():
    global T_entrenamiento, V, mu, Xc, lam, x_prueba_actual
    
    if T_entrenamiento is None:
        T_entrenamiento = Xc @ V

    if 'foto' not in request.files:
        return jsonify({"error": "No se subió ninguna foto"}), 400
        
    foto = request.files['foto']
    img = Image.open(io.BytesIO(foto.read())).convert('L').resize((64, 64))
    x_prueba_actual = np.array(img, dtype=np.float32).flatten() / 255.0
    
    # Obtenemos k que viene del formulario (o 1 por defecto)
    k = int(request.form.get('k', 1))
    t = pca.proyectar(x_prueba_actual, mu, V, k)
    
    umbral_decision = 12.0 
    clase_id, distancia = pca.identificar(t, T_entrenamiento, y_labels, k, umbral_decision)
    
    nombre_asignado = nombres_clases.get(clase_id, "Desconocido")
    if clase_id == "Desconocido":
        nombre_asignado = "Desconocido (Distancia excedida)"
        
    # Calculamos varianza, error actual e imagen reconstruida
    P_k = pca.varianza_acumulada(lam)
    porcentaje = P_k[k-1] if k-1 < len(P_k) else 100.0
    error = pca.calcular_error_reconstruccion(x_prueba_actual, t, V, mu, k)
    
    img_recon_pil = pca.obtener_imagen_reconstruida(x_prueba_actual, V, mu, k)
    buffered = BytesIO()
    img_recon_pil.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
    return jsonify({
        "clase": nombre_asignado,
        "distancia": float(distancia),
        "max_k": len(lam),
        "varianza": float(porcentaje),
        "error": float(error),
        "imagen_reconstruida": img_b64
    })

@app.route("/reconstruct")
def reconstruct():
    global x_prueba_actual, V, mu, lam, T_entrenamiento, y_labels
    k = int(request.args.get('k', 1))
    
    P_k = pca.varianza_acumulada(lam)
    porcentaje = P_k[k-1] if k-1 < len(P_k) else 100.0
    
    error = 0.0
    nombre_asignado = "---"
    img_b64 = ""
    
    if x_prueba_actual is not None:
        t = pca.proyectar(x_prueba_actual, mu, V, k)
        error = pca.calcular_error_reconstruccion(x_prueba_actual, t, V, mu, k)
        
        clase_id, _ = pca.identificar(t, T_entrenamiento, y_labels, k, 35.0)
        nombre_asignado = nombres_clases.get(clase_id, "Desconocido")
        if clase_id == "Desconocido":
            nombre_asignado = "Desconocido (Distancia excedida)"
            
        # Generar la imagen reconstruida en formato base64
        img_recon_pil = pca.obtener_imagen_reconstruida(x_prueba_actual, V, mu, k)
        buffered = BytesIO()
        img_recon_pil.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({
        "clase": nombre_asignado,
        "max_k": len(lam),
        "varianza": float(porcentaje),
        "error": float(error),
        "imagen_reconstruida": img_b64
    })

@app.route("/tabla_varianza")
def tabla_varianza():
    global lam
    if lam is None:
        return jsonify([])
    
    varianza_ind = lam / np.sum(lam) * 100
    varianza_acum = pca.varianza_acumulada(lam)
    
    tabla_datos = []
    for i in range(len(lam)):
        tabla_datos.append({
            "k": i + 1,
            "individual": float(varianza_ind[i]),
            "acumulada": float(varianza_acum[i])
        })
    return jsonify(tabla_datos)

@app.route("/imagen_media")
def imagen_media():
    global mu
    if mu is None:
        return jsonify({"error": "Modelo no inicializado"}), 400
    
    # mu es un vector de 4096. Lo pasamos a 64x64 y lo escalamos a 255
    mu_img_array = (mu.reshape(64, 64) * 255).astype(np.uint8)
    img_pil = Image.fromarray(mu_img_array, mode='L')
    
    buffered = BytesIO()
    img_pil.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return jsonify({"imagen_media": img_b64})

@app.route("/info_galeria")
def info_galeria():
    global X, y_labels, nombres_clases
    if X is None:
        return jsonify({"m": 0, "p": 0, "clases": {}})
    
    m = len(X)
    p = X.shape[1] if len(X) > 1 else 0
    
    # Contar cuántas imágenes hay por cada clase
    conteo_clases = {}
    if y_labels is not None:
        for etiqueta in y_labels:
            nombre = nombres_clases.get(etiqueta, str(etiqueta))
            conteo_clases[nombre] = conteo_clases.get(nombre, 0) + 1
            
    return jsonify({
        "m": m,
        "p": p,
        "clases": conteo_clases
    })

@app.route("/imagen_gram")
def imagen_gram():
    global Xc
    if Xc is None:
        return jsonify({"error": "Modelo no inicializado"}), 400
    
    # Calcular la matriz de Gram G = Xc @ Xc.T
    G = Xc @ Xc.T
    
    # Generar el gráfico y guardarlo en memoria o en la carpeta static
    plt.figure(figsize=(6, 5))
    sns.heatmap(G, cmap="Blues", cbar=True, square=True, xticklabels=False, yticklabels=False)
    plt.title("Matriz de Gram G")
    
    buffered = BytesIO()
    plt.tight_layout()
    plt.savefig(buffered, format="JPEG", dpi=200)
    plt.close()
    
    img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return jsonify({"imagen_gram": img_b64})

@app.route("/valores_propios")
def valores_propios():
    global lam
    if lam is None:
        return jsonify([])
    
    # lam ya viene ordenado y limpio de tu función espectro en pca.py
    lista_vp = []
    for i, val in enumerate(lam):
        lista_vp.append({
            "k": i + 1,
            "lambda": float(val)
        })
    return jsonify(lista_vp)


if __name__ == "__main__":
    app.run(debug=False, port=5000)