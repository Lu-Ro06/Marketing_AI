import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import io

# --- 1. PREPROCESAMIENTO ---
def preprocess_sales_data(df):
    """Limpia, transforma y escala el dataset de ventas."""
    df_processed = df.copy()

    # Convertir fecha
    if 'ORDERDATE' in df_processed.columns:
        df_processed['ORDERDATE'] = pd.to_datetime(df_processed['ORDERDATE'])

    # Función interna para dummies
    def get_dummies_and_drop(dataframe, column_name):
        if column_name in dataframe.columns:
            dummy = pd.get_dummies(dataframe[column_name], prefix=column_name, dtype=bool)
            dataframe = dataframe.drop(columns=[column_name])
            dataframe = pd.concat([dataframe, dummy], axis=1)
        return dataframe

    # Columnas a eliminar (según tu notebook)
    df_drop_list = ['ADDRESSLINE1', 'ADDRESSLINE2', 'POSTALCODE', 'CITY', 'TERRITORY', 
                    'PHONE', 'STATE', 'CONTACTFIRSTNAME', 'CONTACTLASTNAME', 
                    'CUSTOMERNAME', 'ORDERNUMBER', 'STATUS', 'ORDERDATE', 'QTR_ID']
    
    for col in df_drop_list:
        if col in df_processed.columns:
            df_processed.drop(columns=[col], inplace=True)

    # Aplicar dummies
    for col in ['COUNTRY', 'PRODUCTLINE', 'DEALSIZE']:
        df_processed = get_dummies_and_drop(df_processed, col)

    # Codificación de PRODUCTCODE
    if 'PRODUCTCODE' in df_processed.columns:
        df_processed['PRODUCTCODE'] = pd.Categorical(df_processed['PRODUCTCODE']).codes

    # Asegurar tipo float
    if 'ORDERLINENUMBER' in df_processed.columns:
        df_processed['ORDERLINENUMBER'] = df_processed['ORDERLINENUMBER'].astype(float)

    # Escalado
    scaler = StandardScaler()
    sales_df_scaled = scaler.fit_transform(df_processed)

    return sales_df_scaled, df_processed, scaler


# --- 2. AUTOENCODER ---
def build_autoencoder(input_dim):
    """Construye la arquitectura del Autoencoder."""
    input_df = Input(shape=(input_dim,))
    
    # Encoder
    x = Dense(50, activation='relu')(input_df)
    x = Dense(500, activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Dense(500, activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Dense(2000, activation='relu', kernel_initializer='glorot_uniform')(x)
    encoded = Dense(8, activation='relu', kernel_initializer='glorot_uniform')(x)
    
    # Decoder
    x = Dense(2000, activation='relu', kernel_initializer='glorot_uniform')(encoded)
    x = Dense(500, activation='relu', kernel_initializer='glorot_uniform')(x)
    decoded = Dense(input_dim, kernel_initializer='glorot_uniform')(x)

    autoencoder = Model(input_df, decoded)
    encoder = Model(input_df, encoded)

    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder, encoder

def train_autoencoder(autoencoder_model, scaled_data, epochs=50, batch_size=120):
    """Entrena el modelo."""
    autoencoder_model.fit(scaled_data, scaled_data, batch_size=batch_size, epochs=epochs, verbose=0)
    return autoencoder_model


# --- 3. CLUSTERING ---
def find_optimal_clusters(data_to_cluster, max_clusters=20):
    """Genera el gráfico del método del codo."""
    scores = []
    range_values = range(1, max_clusters + 1)
    for i in range_values:
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(data_to_cluster)
        scores.append(kmeans.inertia_)

    fig = go.Figure(data=go.Scatter(x=list(range_values), y=scores, mode='lines+markers'))
    fig.update_layout(title='Método del Codo', xaxis_title='Clusters', yaxis_title='Inercia')
    return fig

def apply_kmeans(data_to_cluster, n_clusters=3):
    """Aplica K-Means."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(data_to_cluster)
    return kmeans.labels_, kmeans.cluster_centers_


# --- 4. VISUALIZACIÓN ---
def plot_pca_2d(scaled_data, labels):
    """Gráfico 2D interactivo."""
    pca = PCA(n_components=2)
    principal_comp = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(data=principal_comp, columns=['pca1', 'pca2'])
    pca_df['cluster'] = labels.astype(str)

    fig = px.scatter(pca_df, x='pca1', y='pca2', color='cluster', 
                     title='Clusters 2D (PCA)', color_discrete_sequence=px.colors.qualitative.Safe)
    return fig

def plot_pca_3d(scaled_data, labels):
    """Gráfico 3D interactivo."""
    pca = PCA(n_components=3)
    principal_comp = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(data=principal_comp, columns=['pc1', 'pc2', 'pc3'])
    pca_df['cluster'] = labels.astype(str)

    fig = px.scatter_3d(pca_df, x='pc1', y='pc2', z='pc3', color='cluster', 
                        title='Clusters 3D (PCA)', opacity=0.7)
    return fig