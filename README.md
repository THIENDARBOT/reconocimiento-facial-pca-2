# Reconocimiento de imágenes con PCA (Eigenfaces)

Este proyecto implementa un sistema de reconocimiento facial utilizando el Análisis de Componentes Principales (PCA) basado en la técnica de *Autovalores y Autovectores*. El sistema cuenta con una interfaz web interactiva que permite proyectar, reconstruir e identificar imágenes nuevas midiendo su distancia en el subespacio generado.

## 

## Instalación

Para ejecutar este proyecto de manera local, asegúrate de tener instalado Python (se recomienda versión 3.8 o superior). Sigue estos pasos para configurar el entorno:

1. **Clonar o descargar el repositorio:**
Extrae los archivos del proyecto en una carpeta local.

## 

## Ejecución

El proyecto está construido con Flask y se ejecuta de forma local.

1. Abre una terminal y navega hasta la carpeta raíz del proyecto.
2. Ejecutar el commando pip install -r requirements.txt
3. Inicia el servidor ejecutando:



&#x09;python app.py



4\.  Abre tu navegador web y dirígete a la siguiente dirección para interactuar con la interfaz:
    **http://127.0.0.1:5000/**





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

