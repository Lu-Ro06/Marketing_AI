import streamlit as st
import pandas as pd
import backend_logic as bl 
import plotly.express as px

# Configuración de la página
st.set_page_config(
    page_title="Analizador de Clusters IA",
    page_icon="🐢",
    layout="wide"
)

# Estilos personalizados (Corregido)
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Título principal
st.title(" Marketing AI 🍞")
st.subheader("Entrenamiento de Autoencoders y Clustering en Tiempo Real")

# --- BARRA LATERAL ---
st.sidebar.header("Configuración del Modelo")
n_clusters = st.sidebar.slider("Número de Clusters (K)", min_value=2, max_value=10, value=3)
epochs_input = st.sidebar.number_input("Épocas de Entrenamiento", min_value=10, max_value=1000, value=50)

# --- PASO 1: CARGA DE DATOS ---
st.info("Por favor, sube tu dataset de ventas en formato .CSV para comenzar el análisis.")
archivo = st.file_uploader("Subir Dataset", type=["csv"])

if archivo is not None:
    df_raw = pd.read_csv(archivo)
    
    col_pre, col_stats = st.columns([1, 2])
    with col_pre:
        st.write("### Vista previa", df_raw.head(5))
    with col_stats:
        st.write("### Estadísticas Rápidas", df_raw.describe())

    # --- BOTÓN DE ACCIÓN ---
    if st.button("Comenzar Procesamiento y Entrenamiento"):
        
        try:
            # 1. Preprocesamiento
            with st.status("Ejecutando pipeline de datos...", expanded=True) as status:
                st.write("Limpiando y escalando datos...")
                scaled_data, df_processed, scaler = bl.preprocess_sales_data(df_raw)
                
                # 2. Construcción y Entrenamiento del Autoencoder
                st.write(f"Entrenando Autoencoder por {epochs_input} épocas...")
                input_dim = scaled_data.shape[1]
                autoencoder, encoder = bl.build_autoencoder(input_dim)
                bl.train_autoencoder(autoencoder, scaled_data, epochs=epochs_input)
                
                # 3. Reducción de dimensionalidad
                st.write("Calculando espacio latente...")
                encoded_data = encoder.predict(scaled_data)
                
                # 4. Clustering
                st.write(f"Aplicando K-Means (K={n_clusters})...")
                labels, centers = bl.apply_kmeans(encoded_data, n_clusters=n_clusters)
                
                status.update(label="¡Análisis completado!", state="complete", expanded=False)

            # --- RESULTADOS ---
            st.divider()
            st.header("📊 Visualización de Resultados")
            
            row1_col1, row1_col2 = st.columns(2)
            with row1_col1:
                st.plotly_chart(bl.find_optimal_clusters(encoded_data), use_container_width=True)
            with row1_col2:
                st.plotly_chart(bl.plot_pca_2d(scaled_data, labels), use_container_width=True)

            st.plotly_chart(bl.plot_pca_3d(scaled_data, labels), use_container_width=True)

            # Descarga
            df_final = df_raw.copy()
            df_final['Cluster_Asignado'] = labels
            csv = df_final.to_csv(index=False).encode('utf-8')
            st.download_button("Descargar CSV con Clusters", csv, "ventas_segmentadas.csv", "text/csv")
            
            

        except Exception as e:
            st.error(f"Error: {e}")

else:
    st.write("Esperando archivo...")