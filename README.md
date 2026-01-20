# projeto-deteccao-pessoas
Esse projeto é referente ao detectron2 para a disciplina de IA. Os autores desse projeto são Alexandre Barroso Lima 570524 e João Pedro Lima de Moura 570446.

O projeto tem como objetivo a implementação do detectron2 para a monitoração, nesse modelo avaliamos a ocupação da sala de estudo. O sistema identifica e classifica a sala como vazio ou ocupado, assim, permitindo uma melhora na gestão desse espaço.
A relação dele com a segurança da informação é simples, com a implementação do detectron2 pode ser realizado logs de presenças, pois estaria sendo monitorado, prevenção de incidentes, se por acaso expancionados para objetos perigosos o sistema emitir um alerta avisando o risco, e resposta a incidentes já que facilitaria a determinar o que ou quem causou o dano.

passo um : INSTALAÇÃO DO DETECTRON2
!pip install -U torch torchvision torchaudio
!pip install -U 'git+https://github.com/facebookresearch/detectron2.git'

passo dois : AUTORIZAÇÃO PARA O DRIVE, IMPORTANTE POR CONTA DAS FOTOS
from google.colab import drive
drive.mount('/content/drive')

passo três : validação
def avaliar_dataset(dataset_path, predictor):
    y_true = []
    y_pred = []

    for classe_real in ["vazia", "ocupada"]:
        pasta = os.path.join(dataset_path, classe_real)

        for img_name in os.listdir(pasta):
            img_path = os.path.join(pasta, img_name)
            imagem = cv2.imread(img_path)

            if imagem is None:
                continue

            outputs = predictor(imagem)
            instances = outputs["instances"]
            num_pessoas = (instances.pred_classes == 0).sum().item()

            estado_detectado = "vazia" if num_pessoas == 0 else "ocupada"

            rotulo_real = 0 if classe_real == "vazia" else 1
            rotulo_pred = 0 if estado_detectado == "vazia" else 1

            y_true.append(rotulo_real)
            y_pred.append(rotulo_pred)

    return y_true, y_pred

passo quatro : teste
import os
import cv2
import torch
from google.colab.patches import cv2_imshow
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50 
cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)

PATH_INPUT = "/content/drive/MyDrive/Projeto_IA/imagens"
PATH_OUTPUT = "./results/images"
os.makedirs(PATH_OUTPUT, exist_ok=True)

arquivos = [f for f in os.listdir(PATH_INPUT) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"Encontradas {len(arquivos)} imagens. Iniciando análise...\n")

for img_name in arquivos:
    img_path = os.path.join(PATH_INPUT, img_name)
    img = cv2.imread(img_path)
    
    if img is None: 
        continue

    outputs = predictor(img)
    instances = outputs["instances"]
    
    vips = instances[instances.pred_classes == 0]
    num_pessoas = len(vips)
    status = "OCUPADA" if num_pessoas > 0 else "VAZIA"
    
    print("-" * 30)
    print(f"Arquivo: {img_name}")
    print(f"Status: {status} ({num_pessoas} pessoas detectadas)")


    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.8)
    out = v.draw_instance_predictions(vips.to("cpu"))
    
    result_img = out.get_image()[:, :, ::-1]
    
    cv2_imshow(result_img)
    
    save_path = os.path.join(PATH_OUTPUT, f"result_{img_name}")
    cv2.imwrite(save_path, result_img)

print("\nProcessamento concluído. As imagens foram salvas em './results/images'.")

Resultado :<img width="409" height="409" alt="download" src="https://github.com/user-attachments/assets/d0ca45f4-7946-4803-9d0e-a4748ac10132" />

<img width="409" height="409" alt="download" src="https://github.com/user-attachments/assets/a4cc240e-f51c-490b-a7a3-623b3d28cca2" />

<img width="409" height="409" alt="download" src="https://github.com/user-attachments/assets/177e9e10-c273-4bf6-aec7-f66597c8af0a" />

