import os
import numpy as np
import pandas as pd
import torch
import faiss
from sklearn.preprocessing import StandardScaler
import pickle

BASE_DIR = "../Data/preprocessed"
FEATURES_DIR = '../Data/processed/npy'
IDX_DIR = 'indices'
SCALERS_DIR = '../Data/processed/scalers'

# Cargar las características guardadas
color_features = np.load(os.path.join(FEATURES_DIR, 'color_features.npy'))
texture_features = np.load(os.path.join(FEATURES_DIR, 'texture_features.npy'))
shape_features = np.load(os.path.join(FEATURES_DIR, 'shape_features.npy'))
cnn_features = np.load(os.path.join(FEATURES_DIR, 'cnn_features.npy'))

# Cargar el DataFrame con los datos de las imágenes
df = pd.read_csv('../Data/df.csv')

# Crear un scaler para cada tipo de característica
scaler_color = StandardScaler()
scaler_texture = StandardScaler()
scaler_shape = StandardScaler()
scaler_cnn = StandardScaler()
scaler_combined = StandardScaler()

# Normalizar las características individuales
color_features_norm = scaler_color.fit_transform(color_features)
texture_features_norm = scaler_texture.fit_transform(texture_features)
shape_features_norm = scaler_shape.fit_transform(shape_features)
cnn_features_norm = scaler_cnn.fit_transform(cnn_features)

# Convertir las características a float32 para FAISS
color_features_norm = color_features_norm.astype('float32')
texture_features_norm = texture_features_norm.astype('float32')
shape_features_norm = shape_features_norm.astype('float32')
cnn_features_norm = cnn_features_norm.astype('float32')

# Concatenar las características para el índice combinado
combined_features = np.hstack((color_features_norm, texture_features_norm, shape_features_norm, cnn_features_norm))
combined_features_norm = scaler_combined.fit_transform(combined_features)
combined_features_norm = combined_features_norm.astype('float32')

def find_appropriate_M(d, max_M=16):
    divisors = [i for i in range(1, max_M+1) if d % i == 0]
    if not divisors:
        raise ValueError(f"No se encontró un divisor de {d} menor o igual a {max_M}.")
    return max(divisors)

# Función para crear un índice IVFPQ con M adecuado
def create_ivfpq_index(features, d, nlist=100, M=None):
    if M is None:
        M = find_appropriate_M(d)
    else:
        if d % M != 0:
            raise ValueError(f"La dimensión {d} no es divisible por M={M}.")
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, M, 8)
    index.train(features)
    index.add(features)
    return index


# Índice para características de color
d_color = color_features_norm.shape[1]
M_color = find_appropriate_M(d_color)
index_color = create_ivfpq_index(color_features_norm, d_color, M=M_color)

# Índice para características de textura
d_texture = texture_features_norm.shape[1]
M_texture = find_appropriate_M(d_texture)
index_texture = create_ivfpq_index(texture_features_norm, d_texture, M=M_texture)

# Índice para características de forma
d_shape = shape_features_norm.shape[1]
M_shape = find_appropriate_M(d_shape)
index_shape = create_ivfpq_index(shape_features_norm, d_shape, M=M_shape)

# Índice para características CNN
d_cnn = cnn_features_norm.shape[1]
M_cnn = find_appropriate_M(d_cnn)
index_cnn = create_ivfpq_index(cnn_features_norm, d_cnn, M=M_cnn)

# Índice para características combinadas
d_combined = combined_features_norm.shape[1]
M_combined = find_appropriate_M(d_combined)
index_combined = create_ivfpq_index(combined_features_norm, d_combined, M=M_combined)

# Guardar los índices en archivos para su uso posterior (con extensión .index)
faiss.write_index(index_color, f'{IDX_DIR}/index_color.index')
faiss.write_index(index_texture, f'{IDX_DIR}/index_texture.index')
faiss.write_index(index_shape, f'{IDX_DIR}/index_shape.index')
faiss.write_index(index_cnn, f'{IDX_DIR}/index_cnn.index')
faiss.write_index(index_combined, f'{IDX_DIR}/index_combined.index')

with open(f'{SCALERS_DIR}/scaler_color.pkl', 'wb') as f:
    pickle.dump(scaler_color, f)
with open(f'{SCALERS_DIR}/scaler_texture.pkl', 'wb') as f:
    pickle.dump(scaler_texture, f)
with open(f'{SCALERS_DIR}/scaler_shape.pkl', 'wb') as f:
    pickle.dump(scaler_shape, f)
with open(f'{SCALERS_DIR}/scaler_cnn.pkl', 'wb') as f:
    pickle.dump(scaler_cnn, f)
with open(f'{SCALERS_DIR}/scaler_combined.pkl', 'wb') as f:
    pickle.dump(scaler_combined, f)