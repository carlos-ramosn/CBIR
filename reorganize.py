import os
import shutil


def reorganizar_datos(base_dir):
    # Rutas de las nuevas carpetas
    processed_dir = os.path.join(base_dir, "processed")
    raw_dir = os.path.join(base_dir, "raw")

    # Subcarpetas en processed
    processed_train_dir = os.path.join(processed_dir, "train")
    npy_dir = os.path.join(processed_dir, "npy")
    scalers_dir = os.path.join(processed_dir, "scalers")

    # Crear carpetas si no existen
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_train_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(scalers_dir, exist_ok=True)

    # Mover datos crudos a raw
    for folder in ["train", "test", "valid"]:
        folder_path = os.path.join(base_dir, folder)
        if os.path.exists(folder_path):
            shutil.move(folder_path, raw_dir)

    print(f"Datos crudos movidos a {raw_dir}")

    # Coloca aquí la lógica de mover npy y scalers a sus respectivas carpetas
    for file in os.listdir(base_dir):
        if file.endswith(".npy"):
            shutil.move(os.path.join(base_dir, file), npy_dir)
        elif file.endswith(".pkl"):
            shutil.move(os.path.join(base_dir, file), scalers_dir)

    print(f"Archivos procesados organizados en {processed_dir}")


# Define la ruta base de tu proyecto
base_dir = "C:/Users/34676/PycharmProjects/CBIR/Data"

# Ejecutar la reorganización
reorganizar_datos(base_dir)
