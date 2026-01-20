import os
import cv2
from google.colab.patches import cv2_imshow
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

cfg.MODEL.WEIGHTS = "./training/output/model_final.pth" 

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50 # Confiança mínima de 50%
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1        # Apenas 1 classe: pessoa
cfg.MODEL.DEVICE = "cpu" 

predictor = DefaultPredictor(cfg)

PATH_INPUT = "/content/drive/MyDrive/Projeto_IA/imagens"
PATH_OUTPUT = "./results/images"
os.makedirs(PATH_OUTPUT, exist_ok=True)

arquivos = [f for f in os.listdir(PATH_INPUT) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f" Analisando {len(arquivos)} imagens com o modelo treinado...\n")

for img_name in arquivos:
    img_path = os.path.join(PATH_INPUT, img_name)
    img = cv2.imread(img_path)
    
    if img is None: continue

    outputs = predictor(img)
    vips = outputs["instances"].to("cpu")
    
    vips = vips[vips.pred_classes == 0]
    
    v = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("campus_people"), scale=0.8)
    out = v.draw_instance_predictions(vips)
    
    result_img = out.get_image()[:, :, ::-1]
    
    print(f"Resultado para: {img_name}")
    cv2_imshow(result_img)
    
    cv2.imwrite(os.path.join(PATH_OUTPUT, f"detec_{img_name}"), result_img)

print(f" Concluído! Imagens salvas em {PATH_OUTPUT}")
