# 🤖 Sistema de Segmentación de Clientes con IA (Deep Learning)

Este proyecto es una aplicación web interactiva diseñada para analizar datos de ventas y marketing. Utiliza una arquitectura de **Autoencoders (Redes Neuronales)** para comprimir información compleja y **K-Means Clustering** para agrupar clientes o productos de forma automática en un espacio latente.

## 🌟 Características Clave
- **Pipeline de Datos Automático:** Limpieza, manejo de variables categóricas (Dummies) y escalado (`StandardScaler`).
- **Reducción de Dimensionalidad:** Implementación de una Red Neuronal con Keras/TensorFlow que comprime los datos a 8 dimensiones críticas.
- **Visualización Avanzada:** Gráficos interactivos en 2D y 3D utilizando PCA para interpretar los clusters generados.
- **Interfaz Profesional:** Panel de control con Streamlit que permite ajustar parámetros (K-clusters, épocas) en tiempo real.
- **Exportación:** Opción para descargar el dataset final etiquetado por la IA.

## 🛠️ Estructura del Proyecto
```text
App_Marketing_IA/
├── app.py                # Interfaz de usuario y flujo principal
├── backend_logic.py      # Motor de IA (Preprocesamiento, Autoencoder, K-Means)
├── requirements.txt      # Dependencias del sistema
└── README.md             # Documentación del proyecto
🧠 Metodología y Arquitectura
1. El Autoencoder (Reducción No Lineal)
A diferencia de métodos tradicionales, utilizamos una Red Neuronal para aprender una representación comprimida de los datos:

Encoder: Comprime las columnas originales a través de capas densas (50, 500, 2000 neuronas) hasta un "cuello de botella" de 8 dimensiones.

Espacio Latente: Aquí es donde la IA extrae la verdadera esencia del comportamiento de compra, eliminando el "ruido" de los datos originales.

Decoder: Intenta reconstruir los datos originales para asegurar que la compresión fue precisa (optimizando el error cuadrático medio - MSE).

2. Clustering en Espacio Latente
El algoritmo K-Means se ejecuta sobre las 8 dimensiones generadas por el Autoencoder. Esto permite encontrar grupos mucho más definidos y compactos que si se hiciera sobre los datos crudos.

📦 Instalación y Configuración
Clonar o descargar la carpeta App_Marketing_IA.

Abrir una terminal en dicha carpeta.

Crear y activar un entorno virtual:

Bash
python3 -m venv .venv
# En Windows: .venv\Scripts\activate
# En Linux/Mac: source .venv/bin/activate
Instalar las librerías necesarias:

Bash
pip install -r requirements.txt
🖥️ Cómo Ejecutar la Aplicación
Para iniciar el servidor local, escribe el siguiente comando:

Bash
streamlit run app.py
Se abrirá una pestaña en tu navegador (usualmente en http://localhost:8501). Sube tu archivo CSV, ajusta los parámetros en la barra lateral y presiona "Comenzar Procesamiento".

Proyecto desarrollado por: María Lucero Rodea Martínez 🚀
Institución: CAET


---

### Notas finales para tu entrega:
* **requirements.txt:** Asegúrate de que contenga: `streamlit`, `pandas`, `numpy`, `tensorflow`, `scikit-learn`, `plotly`, `matplotlib`, `seaborn`.
* **Presentación:** Si te preguntan por la arquitectura, recuerda mencionar que el **Autoencoder** captura relaciones **no lineales** que el simple PCA no puede ver.

¡Con esto tu proyecto está listo para una calificación excelente!