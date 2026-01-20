# projeto-deteccao-pessoas
Esse projeto é referente ao detectron2 para a disciplina de IA. Os autores desse projeto são Alexandre Barroso Lima 570524 e João Pedro Lima de Moura 570446.

O projeto tem como objetivo detectar pessoas dentro do ambiente universitário que é a sala de estudos com a implementação do detectron2.
As ferramentas que estão sendo utilizadas vão ser o google colab para o código e o roboflow para o dataset.
A detecção de pessoas de maneira automática com o propósito de acompanhamento dentro do espaço escolhido, também com o propósito para ter um determinado controle sobre as pessoas que utilizam esse espaço. Para a segurança da informação essa utilização da IA é importante para fazer logs caso algum acidente ou incidente aconteça. Como estamos nos referindo a um ambiente compartilhado pode acontecer uma ocorrência e facilitaria muito se reconhecido quem ou o que causou. Um exemplo poderia ser um incêndio na sala de estudos, se o detectron2 for ensinado a reconhecer pessoas com fogo, ele emitiria um alerta para caso isso acontecesse agindo como uma prevenção. 

passo um : INSTALAÇÃO DO DETECTRON2
!pip install -U torch torchvision torchaudio
!pip install -U 'git+https://github.com/facebookresearch/detectron2.git'

passo dois : AUTORIZAÇÃO PARA O DRIVE, IMPORTANTE POR CONTA DAS FOTOS
from google.colab import drive
drive.mount('/content/drive')

passo três : PARTE DO CÓDIGO ONDE ACONTECE O TREINAMENTO
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

passo quatro : VERIFICAÇÃO DA PASTA COM AS FOTOS
    import os
from google.colab import drive
drive.mount('/content/drive')

!ls "/content/drive/MyDrive/Projeto_IA/imagens"

passo cinco : CÓDIGO DE DETECÇÃO DE PESSOAS +OBJETOS
import os
import cv2
import torch
from google.colab.patches import cv2_imshow # Para mostrar imagens no Colab
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50 
cfg.MODEL.DEVICE = "cpu" # Forçando CPU conforme seu log anterior
predictor = DefaultPredictor(cfg)

PATH = "/content/drive/MyDrive/Projeto_IA/imagens"

arquivos = [f for f in os.listdir(PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"Encontradas {len(arquivos)} imagens. Iniciando análise...\n")

for img_name in arquivos:
    img_path = os.path.join(PATH, img_name)
    img = cv2.imread(img_path)
    
    if img is None: continue

    outputs = predictor(img)
    instances = outputs["instances"]
    
    # Filtra apenas classe 0 (pessoa)
    vips = instances[instances.pred_classes == 0]
    num_pessoas = len(vips)
    status = "OCUPADA" if num_pessoas > 0 else "VAZIA"
    
    print(f" {img_name} | Resultado: {status} ({num_pessoas} pessoas detectadas)")

    # OPCIONAL: Mostrar a primeira imagem com o desenho da detecção para conferir
    if img_name == arquivos[0]:
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.8)
        out = v.draw_instance_predictions(instances.to("cpu"))
        print("\nExemplo visual da primeira imagem:")
        cv2_imshow(out.get_image()[:, :, ::-1])
