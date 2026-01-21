import os
import torch
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator

DRIVE_PATH = "/content/drive/MyDrive/Projeto_IA"
IMG_DIR = os.path.join(DRIVE_PATH, "imagens")
ANNOT_DIR = os.path.join(DRIVE_PATH, "anotacoes")
JSON_FILE = os.path.join(ANNOT_DIR, "_annotations.coco.json")

register_coco_instances("campus_people", {}, JSON_FILE, IMG_DIR)
print(" Dataset 'campus_people' registrado com sucesso!")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ("campus_people",)
cfg.DATASETS.TEST = () 

cfg.DATALOADER.NUM_WORKERS = 2

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

cfg.SOLVER.IMS_PER_BATCH = 2  
cfg.SOLVER.BASE_LR = 0.00025  
cfg.SOLVER.MAX_ITER = 500     
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 

if torch.cuda.is_available():
    cfg.MODEL.DEVICE = "cuda"
    print(" Usando GPU para treinamento!")
else:
    cfg.MODEL.DEVICE = "cpu"
    print(" AVISO: GPU não encontrada. O treinamento será LENTO na CPU.")

cfg.OUTPUT_DIR = "./training/output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    
    print(f"Iniciando treinamento com {cfg.SOLVER.MAX_ITER} iterações...")
    trainer.train()
    
    print(f" Treinamento concluído! O modelo foi salvo em: {cfg.OUTPUT_DIR}/model_final.pth")
