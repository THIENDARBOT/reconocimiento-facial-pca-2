from flask import Flask, render_template, request, jsonify
import numpy as np
import pca
import io_datos
import io
import base64
from io import BytesIO
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
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
        Xc, mu = pca.centrar(X)
        lam, V = pca.espectro(Xc)
        T_entrenamiento = Xc @ V
        print(f"Modelo cargado correctamente. Clases: {list(nombres_clases.values())}")
    else:
        print("Advertencia: No se encontraron imágenes de entrenamiento en la carpeta.")

with app.app_context():
    inicializar_modelo()

@app.route("/")
def index():
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
    
    k = int(request.form.get('k', 1))
    t = pca.proyectar(x_prueba_actual, mu, V, k)
    
    umbral_decision = 12.0 
    clase_id, distancia = pca.identificar(t, T_entrenamiento, y_labels, k, umbral_decision)
    
    nombre_asignado = nombres_clases.get(clase_id, "Desconocido")
    if clase_id == "Desconocido":
        nombre_asignado = "Desconocido (Distancia excedida)"
        
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
    distancia = 0.0
    nombre_asignado = "---"
    img_b64 = ""
    
    if x_prueba_actual is not None:
        t = pca.proyectar(x_prueba_actual, mu, V, k)
        error = pca.calcular_error_reconstruccion(x_prueba_actual, t, V, mu, k)
        
        clase_id, distancia = pca.identificar(t, T_entrenamiento, y_labels, k, 35.0)
        nombre_asignado = nombres_clases.get(clase_id, "Desconocido")
        if clase_id == "Desconocido":
            nombre_asignado = "Desconocido (Distancia excedida)"
            
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
    
    fig, ax = plt.subplots(figsize=(6, 5))
    G = Xc @ Xc.T
    sns.heatmap(G, ax=ax, cmap="Blues", cbar=True, square=True, xticklabels=False, yticklabels=False)
    ax.set_title("Matriz de Gram G")
    
    buffered = BytesIO()
    fig.tight_layout()
    fig.savefig(buffered, format="JPEG", dpi=200)
    plt.close(fig) # Cierra y aísla esta figura
    
    img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return jsonify({"imagen_gram": img_b64})

@app.route("/imagen_eigenfaces")
def imagen_eigenfaces():
    global V
    if V is None:
        return jsonify({"error": "Modelo no inicializado"}), 400
    
    try:
        fig, axes = plt.subplots(1, 4, figsize=(10, 3))
        num_eigen = min(4, V.shape[1])
        for i in range(num_eigen):
            vec_propio = V[:, i]
            v_min, v_max = vec_propio.min(), vec_propio.max()
            vec_normalizado = (vec_propio - v_min) / (v_max - v_min) if v_max - v_min > 0 else vec_propio
            eigenface = vec_normalizado.reshape(64, 64)
            
            axes[i].imshow(eigenface, cmap="gray")
            axes[i].set_title(f"Eigenface {i+1}")
            axes[i].axis("off")
            
        buffered = BytesIO()
        fig.tight_layout()
        fig.savefig(buffered, format="JPEG", dpi=200, bbox_inches='tight')
        plt.close(fig) # Cierra y aísla esta figura
        
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return jsonify({"eigenfaces": img_b64})
    
    except Exception as e:
        plt.close('all')
        print("Error en eigenfaces:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/grafica_scores")
def grafica_scores():
    global T_entrenamiento, y_labels, nombres_clases
    if T_entrenamiento is None:
        return jsonify({"error": "Modelo no inicializado"}), 400
    
    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        clases_unicas = np.unique(y_labels)
        for etiqueta in clases_unicas:
            mask = (y_labels == etiqueta)
            nombre_clase = nombres_clases.get(etiqueta, str(etiqueta))
            ax.scatter(T_entrenamiento[mask, 0], T_entrenamiento[mask, 1], label=nombre_clase, s=50)
            
        ax.set_title("Scores de Entrenamiento (PC1 vs PC2)")
        ax.set_xlabel("Componente Principal 1")
        ax.set_ylabel("Componente Principal 2")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        
        buffered = BytesIO()
        fig.savefig(buffered, format="JPEG", dpi=200, bbox_inches='tight')
        plt.close(fig) # Cierra y aísla esta figura
        
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return jsonify({"grafica_scores": img_b64})
    
    except Exception as e:
        plt.close('all')
        print("Error en scores:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/valores_propios")
def valores_propios():
    global lam
    if lam is None:
        return jsonify([])
    
    lista_vp = []
    for i, val in enumerate(lam):
        lista_vp.append({
            "k": i + 1,
            "lambda": float(val)
        })
    return jsonify(lista_vp)

if __name__ == "__main__":
    app.run(debug=False, port=5000)