import os
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog

def register_campus_dataset():
    DRIVE_PATH = "/content/drive/MyDrive/Projeto_IA"
    IMG_DIR = os.path.join(DRIVE_PATH, "imagens")
    JSON_PATH = os.path.join(DRIVE_PATH, "anotacoes", "seu_arquivo.json") # Ajuste o nome do arquivo

    DATASET_NAME = "campus_people"

    if DATASET_NAME not in DatasetCatalog.list():
        register_coco_instances(DATASET_NAME, {}, JSON_PATH, IMG_DIR)
        
        MetadataCatalog.get(DATASET_NAME).set(thing_classes=["person"])
        print(f" Dataset '{DATASET_NAME}' registrado com sucesso!")
    else:
        print(f"Dataset '{DATASET_NAME}' já estava registrado.")

if __name__ == "__main__":
    register_campus_dataset()
