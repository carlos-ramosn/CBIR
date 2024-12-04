import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch
import torchvision.models as models
import numpy as np
from PIL import Image
import hashlib
from skimage import feature
from skimage.color import rgb2gray
from torchvision.models import ResNet50_Weights
import streamlit as st

# Definición de las transformaciones para imágenes
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Inicialización del modelo pre-entrenado y extractor de características
model = models.resnet50(pretrained=True)
model.eval()
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])


def load_dataset(base_dir):
    """Cargar datos del dataset en un DataFrame."""
    data = []
    for breed in os.listdir(base_dir):
        breed_dir = os.path.join(base_dir, breed)
        if os.path.isdir(breed_dir):
            for img_file in os.listdir(breed_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    data.append({
                        "filepath": os.path.join(breed_dir, img_file),
                        "breed": breed
                    })
    return pd.DataFrame(data)


def visualize_samples(df, total_samples=5):
    """Visualizar un subconjunto de imágenes del dataset."""
    sample_images = df.sample(total_samples, random_state=1).reset_index(drop=True)
    fig, axes = plt.subplots(1, total_samples, figsize=(15, 5))
    if total_samples == 1:
        axes = [axes]
    for i, (_, row) in enumerate(sample_images.iterrows()):
        img = cv2.imread(row['filepath'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img)
        axes[i].set_title(row['breed'], fontsize=10)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()


def analyze_class_distribution(df):
    """Analizar y graficar distribución de clases."""
    class_counts = df['breed'].value_counts()
    plt.figure(figsize=(12, 6))
    class_counts.plot(kind='bar')
    plt.title("Distribución de Clases")
    plt.xlabel("Raza")
    plt.ylabel("Número de Imágenes")
    plt.xticks(rotation=90, fontsize=5)
    plt.show()


def verify_images(directory):
    """
    Verifica la integridad de las imágenes en un directorio y elimina las corruptas.
    Args:
        directory (str): Ruta al directorio principal que contiene las imágenes.
    """
    num_corrupt = 0
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(root, filename)
                try:
                    with Image.open(filepath) as img:
                        img.verify()
                except (IOError, SyntaxError) as e:
                    print(f"Imagen corrupta encontrada y eliminada: {filepath}")
                    os.remove(filepath)
                    num_corrupt += 1
    print(f"Total de imágenes corruptas eliminadas: {num_corrupt}")


def hash_image(image_path):
    """
    Genera un hash MD5 para una imagen dada.
    Args:
        image_path (str): Ruta al archivo de imagen.
    Returns:
        str: Hash MD5 de la imagen.
    """
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        img_bytes = img.tobytes()
        return hashlib.md5(img_bytes).hexdigest()


def remove_duplicates(directory):
    """
    Elimina imágenes duplicadas en un directorio basándose en el hash de las imágenes.
    Args:
        directory (str): Ruta al directorio principal que contiene las imágenes.
    """
    hashes = {}
    num_duplicates = 0
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(root, filename)
                file_hash = hash_image(filepath)
                if file_hash in hashes:
                    print(f"Imagen duplicada encontrada y eliminada: {filepath}")
                    os.remove(filepath)
                    num_duplicates += 1
                else:
                    hashes[file_hash] = filepath
    print(f"Total de imágenes duplicadas eliminadas: {num_duplicates}")


def extract_color_histogram(image, bins=(8, 8, 8)):
    """
    Extracts a color histogram from an image and normalizes it.

    Args:
        image (PIL.Image.Image): Image to process.
        bins (tuple): Number of bins for each color channel.

    Returns:
        np.array: Normalized feature vector of the color histogram.
    """
    image = image.convert('RGB')
    image = np.array(image)
    # Calculate the histogram for each color channel
    hist = np.histogramdd(
        image.reshape(-1, 3),
        bins=bins,
        range=((0, 256), (0, 256), (0, 256))
    )[0]
    hist = hist.flatten()
    hist = hist / np.sum(hist)
    return hist


def extract_texture_lbp(image, numPoints=24, radius=8):
    """
    Extracts an LBP (Local Binary Patterns) histogram from an image and normalizes it.

    Args:
        image (PIL.Image.Image): Image to process.
        numPoints (int): Number of points in the LBP circle.
        radius (int): Radius of the LBP circle.

    Returns:
        np.array: Normalized feature vector of the LBP histogram.
    """
    image = image.convert('RGB')
    image = np.array(image)
    gray = rgb2gray(image)
    lbp = feature.local_binary_pattern(gray, numPoints, radius, method="uniform")
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist


def extract_hu_moments(image):
    """
    Extracts the Hu Moments from an image.

    Args:
        image (PIL.Image.Image): Image to process.

    Returns:
        np.array: Feature vector with the 7 Hu Moments.
    """
    image = image.convert('L')  # Convert to grayscale
    image = np.array(image)
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    moments = cv2.moments(thresh)
    huMoments = cv2.HuMoments(moments)
    # Apply logarithmic transformation
    for i in range(0, 7):
        huMoments[i] = -np.sign(huMoments[i]) * np.log10(abs(huMoments[i]) + 1e-10)
    return huMoments.flatten()


def extract_cnn_features(img, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=ResNet50_Weights.IMAGENET1K_V1.transforms().mean,
            std=ResNet50_Weights.IMAGENET1K_V1.transforms().std
        )
    ])
    # Aplicar transformaciones
    image = transform(img).unsqueeze(0)
    # Extraer características
    with torch.no_grad():
        features = model(image)
    # Aplanar características
    features = features.numpy().flatten()
    return features


def imagefolder_to_dataframe(dataset):
    """
    Convierte un objeto ImageFolder a un DataFrame de pandas.
    Args:
        dataset (ImageFolder)
    Returns:
        pd.DataFrame: DataFrame con columnas de 'filepath' y 'label'.
    """
    filepaths = [item[0] for item in dataset.samples]
    labels = [dataset.classes[item[1]] for item in dataset.samples]
    df = pd.DataFrame({'filepath': filepaths, 'label': labels})
    return df


def load_cnn_model():
    modelo = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # Eliminar la última capa
    modelo = torch.nn.Sequential(*list(modelo.children())[:-1])
    modelo.eval()
    return modelo
