# Reconocimiento de imágenes con PCA (Eigenfaces) | MATEO MUÑIZ ROQUET 00593092   

Este proyecto implementa un sistema de reconocimiento facial utilizando el Análisis de Componentes Principales (PCA) basado en la técnica de *Autovalores y Autovectores*. El sistema cuenta con una interfaz web interactiva que permite proyectar, reconstruir e identificar imágenes nuevas midiendo su distancia en el subespacio generado.

## 

## Instalación

Para ejecutar este proyecto de manera local, asegúrate de tener instalado Python (se recomienda versión 3.8 o superior). Sigue estos pasos para configurar el entorno:

1. **Clonar o descargar el repositorio:**
Extrae los archivos del proyecto en una carpeta local.

## 

## Ejecución

El proyecto está construido con Flask y se ejecuta de forma local.

## Ejecución

El proyecto está construido con Flask y se ejecuta de forma local. Sigue estos pasos según tu sistema operativo:

### 1. Crear y activar el entorno virtual

* **En Windows (CMD):**
  python -m venv entorno_video |
   Set-ExecutionPolicy Unrestricted -Scope Process |
   entorno_video\Scripts\Activate.ps1
  
* **En macOS o Linux:**
python3 -m venv entorno_video |
 source entorno_video/bin/activate

### 2. Navega hasta la carpeta raíz del proyecto.
### 3. Ejecutar el comando pip install -r requirements.txt
### 4. Inicia el servidor ejecutando: python app.py (Este proceso puede tardar un minuto)
### 5. Una vez inicializado, dirígete en el navegador a la dirección **http://127.0.0.1:5000/** para interactuar con la página.

## Conjunto de Datos

El sistema utiliza una galería de entrenamiento local (`datos/entrenamiento/`) estructurada en carpetas, donde cada carpeta representa una "clase" o identidad distinta (por ejemplo, "Mateo", "Markiplier", etc.).

* **Total de imágenes de entrenamiento:** 36 fotografías.
* **Imágenes de prueba:** El directorio `datos/prueba/` contiene imágenes adicionales para validar el sistema en la interfaz web, incluyendo rostros desconocidos para probar el umbral de rechazo.





## Decisiones de Preprocesamiento y PCA

Para garantizar la viabilidad matemática y el rendimiento computacional del modelo, se tomaron las siguientes decisiones de preprocesamiento:

* **Escala de grises:** Todas las imágenes se convierten a un solo canal (escala de grises) para reducir la complejidad computacional y aislar las variaciones estructurales y de contraste, eliminando la redundancia del color.
* **Redimensionamiento y Recorte:** Las imágenes se estandarizan a una resolución de **64x64 píxeles**. Esto asegura que todas las observaciones tengan la misma longitud vectorial y que los rostros estén espacialmente alineados.
* **Construcción de la Matriz X:** Cada imagen de 64x64 se aplana en un vector fila de 4096 variables. Al apilar las 36 imágenes de entrenamiento, se conforma la matriz de datos estructurada **X pertenece a R 36 x 4096**.
* **Cálculo de Componentes Principales:**

  * Se utiliza el método de la **Matriz de Gram** para calcular los valores y vectores propios de manera eficiente (operando sobre una matriz de 36 x 36 en lugar de 4096 x 4096).
  * Para evitar problemas de inestabilidad, se aplica un filtro de ruido numérico que descarta cualquier valor propio menor a 10^-8.
  * Gracias a este filtrado y a la dependencia lineal natural de los rostros, el sistema determina dinámicamente un máximo de **35 componentes principales efectivos (k = 35)** que capturan el 100% de la varianza útil del conjunto de datos.

