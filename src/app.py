import time
import torch
import faiss
import pathlib
from PIL import Image
import numpy as np
import pandas as pd
import os

import streamlit as st
from streamlit_cropper import st_cropper

import pickle
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights

# Importar las funciones de extracción de características desde functions.py
from functions import extract_color_histogram, extract_texture_lbp, extract_hu_moments, extract_cnn_features, load_cnn_model

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

st.set_page_config(layout="wide")

device = torch.device('cpu')

FILES_PATH = str(pathlib.Path().resolve())

# Rutas
IMAGES_PATH = os.path.join(FILES_PATH, '')  # Directorio con las imágenes
DB_PATH = os.path.join(FILES_PATH, '../Data')  # Directorio con los archivos de la base de datos

DB_FILE = 'df.csv'  # Nombre del archivo de la base de datos
BASE_DIR = "../Data/preprocessed"
FEATURES_DIR = '../Data/processed/npy'
IDX_DIR = 'indices'
SCALERS_DIR = '../Data/processed/scalers'

def get_image_list():
    df = pd.read_csv(os.path.join(DB_PATH, DB_FILE))
    image_list = list(df['filepath'].values)
    return image_list

def retrieve_image(img_query, feature_extractor, n_imgs=11):
    if feature_extractor == 'Histograma de Color':
        # Extraer características de la imagen de consulta
        query_features = extract_color_histogram(img_query)
        # Cargar scaler
        with open(os.path.join(SCALERS_DIR, 'scaler_color.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        # Cargar índice
        index = faiss.read_index(os.path.join(IDX_DIR, 'index_color.index'))
    elif feature_extractor == 'Textura LBP':
        query_features = extract_texture_lbp(img_query)
        with open(os.path.join(SCALERS_DIR, 'scaler_texture.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        index = faiss.read_index(os.path.join(IDX_DIR, 'index_texture.index'))
    elif feature_extractor == 'Momentos de Hu':
        query_features = extract_hu_moments(img_query)
        with open(os.path.join(SCALERS_DIR, 'scaler_shape.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        index = faiss.read_index(os.path.join(IDX_DIR, 'index_shape.index'))
    elif feature_extractor == 'Características CNN':
        model = load_cnn_model()
        query_features = extract_cnn_features(img_query, model)
        with open(os.path.join(SCALERS_DIR, 'scaler_cnn.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        index = faiss.read_index(os.path.join(IDX_DIR, 'index_cnn.index'))
    elif feature_extractor == 'Características Combinadas':
        model = load_cnn_model()
        # Extraer y combinar todas las características
        query_color_hist = extract_color_histogram(img_query)
        query_texture_hist = extract_texture_lbp(img_query)
        query_hu_moments = extract_hu_moments(img_query)
        query_cnn_feat = extract_cnn_features(img_query, model)
        query_features = np.hstack((query_color_hist, query_texture_hist, query_hu_moments, query_cnn_feat))
        with open(os.path.join(SCALERS_DIR, 'scaler_combined.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        index = faiss.read_index(os.path.join(IDX_DIR, 'index_combined.index'))

    # Normalizar características usando el scaler
    query_features = scaler.transform([query_features])[0]

    # Convertir a float32
    query_features = query_features.astype('float32')

    # Preparar el índice (necesario para índices IVFPQ)
    d = query_features.shape[0]
    quantizer = faiss.IndexFlatL2(d)  # Debe ser el mismo utilizado durante la creación
    if not index.is_trained:
        index.train(np.array([query_features]))
    index.nprobe = 10  # Puedes ajustar nprobe para equilibrar velocidad y precisión

    # Buscar en el índice
    distances, indices = index.search(query_features.reshape(1, -1), k=n_imgs)

    # Retornar índices
    return indices[0]


def main():
    st.title('BUSCADOR DE IMÁGENES CBIR')

    col1, col2 = st.columns(2)

    with col1:
        st.header('CONSULTA')

        st.subheader('Elige el extractor de características')
        option = st.selectbox('.', (
        'Histograma de Color', 'Textura LBP', 'Momentos de Hu', 'Características CNN', 'Características Combinadas'))

        st.subheader('Sube una imagen')
        img_file = st.file_uploader(label='.', type=['png', 'jpg', 'jpeg'])

        if img_file:
            img = Image.open(img_file).convert('RGB')
            # Obtener imagen recortada desde el frontend
            cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0004')

            # Mostrar vista previa
            st.write("Vista Previa")
            _ = cropped_img.thumbnail((300, 300))
            st.image(cropped_img)

    with col2:
        st.header('RESULTADO')
        if img_file:
            st.markdown('**Buscando .......**')
            start = time.time()

            retriev_indices = retrieve_image(cropped_img, option, n_imgs=11)
            image_list = get_image_list()

            end = time.time()
            st.markdown('**Finalizado en ' + str(round(end - start, 2)) + ' segundos**')

            # Obtener rutas de las imágenes recuperadas
            retrieved_image_paths = [os.path.join(IMAGES_PATH, image_list[i]) for i in retriev_indices]

            col3, col4 = st.columns(2)

            with col3:
                image = Image.open(retrieved_image_paths[0])
                st.image(image, use_column_width='always', caption='Imagen más similar')

            with col4:
                image = Image.open(retrieved_image_paths[1])
                st.image(image, use_column_width='always', caption='Segunda imagen más similar')

            col5, col6, col7 = st.columns(3)

            with col5:
                for u in range(2, 11, 3):
                    if u < len(retrieved_image_paths):
                        image = Image.open(retrieved_image_paths[u])
                        st.image(image, use_column_width='always')

            with col6:
                for u in range(3, 11, 3):
                    if u < len(retrieved_image_paths):
                        image = Image.open(retrieved_image_paths[u])
                        st.image(image, use_column_width='always')

            with col7:
                for u in range(4, 11, 3):
                    if u < len(retrieved_image_paths):
                        image = Image.open(retrieved_image_paths[u])
                        st.image(image, use_column_width='always')


if __name__ == '__main__':
    main()
