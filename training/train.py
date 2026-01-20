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

try:
    register_coco_instances("campus_people", {}, JSON_FILE, IMG_DIR)
    print(" Dataset 'campus_people' registrado com sucesso!")
except Exception as e:
    print(f" Aviso: {e}")

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

cfg.MODEL.DEVICE = "cpu" 
cfg.OUTPUT_DIR = "./training/output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

class TrainerWithEval(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

if __name__ == "__main__":
    trainer = TrainerWithEval(cfg) 
    trainer.resume_or_load(resume=False)
    print("Iniciando o treinamento...")
    trainer.train()
    print(f" Treinamento concluído! O modelo foi salvo em: {cfg.OUTPUT_DIR}/model_final.pth")
