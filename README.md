# projeto-deteccao-pessoas

Esse projeto é referente ao detectron2 para a disciplina de IA. Os autores desse projeto são Alexandre Barroso Lima 570524 e João Pedro Lima de Moura 570446.
O projeto tem como objetivo a implementação do detectron2 para a monitoração, nesse modelo avaliamos a ocupação da sala de estudo. O sistema identifica e classifica a sala como vazio ou ocupado, assim, permitindo uma melhora na gestão desse espaço.
A relação dele com a segurança da informação é simples, com a implementação do detectron2 pode ser realizado logs de presenças, pois estaria sendo monitorado, prevenção de incidentes, se por acaso expancionados para objetos perigosos o sistema emitir um alerta avisando o risco, e resposta a incidentes já que facilitaria a determinar o que ou quem causou o dano.

---
# AMBIENTE

O ambiente foi a sala de estudos,para visualizar se a sala está vazia ou ocupada. Mas para o desenvolvimento do projeto o trabalho foi feito no colab, com o ambiente de execução na "gpu:t4"
com algumas outras ferramentas. 
    Linguagem : Python
    Dataset : roboflow
    Modelo : faster r cnn
    
---
# DETECTRON2
1. INSTALAÇÃO
'''  
!pip install -U torch torchvision torchaudio
!pip install -U 'git+https://github.com/facebookresearch/detectron2.git'

---
2. PERMISSÃO PARA USAR O DRIVE
from google.colab import drive
drive.mount('/content/drive')
---
3.VALIDAÇÃO 
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
---
4. VERIFICAÇÃO DO DRIVE
import os
from google.colab import drive
drive.mount('/content/drive')

!ls "/content/drive/MyDrive/Projeto_IA/imagens"
---
5. PROCESSAMENTO DOS DADOS
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

print("\nProcessamento concluído. As imagens foram salvas em 'drive/Mydrive/Projeto_IA/results/images'.")
---
6. TREINAMENTO
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
cfg.DATASETS.TEST = () # Deixe vazio se não tiver um JSON separado para teste

cfg.DATALOADER.NUM_WORKERS = 2

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

cfg.SOLVER.IMS_PER_BATCH = 2  # Depende da memória da GPU (2 é seguro para Colab gratuito)
cfg.SOLVER.BASE_LR = 0.00025  # Taxa de aprendizado
cfg.SOLVER.MAX_ITER = 500     # Número de iterações (aumente para 1000 ou 3000 para resultados melhores)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # IMPORTANTE: Garanta que seu JSON tem apenas 1 classe (ex: "pessoa")

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
---

7. DIAGNÓSTICO
import os
import cv2
from google.colab import drive
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

print("Conectando ao Google Drive...")
drive.mount('/content/drive', force_remount=True)

PATH_INPUT = "/content/drive/MyDrive/Projeto_IA/imagens"
PATH_OUTPUT = "./results/images"
os.makedirs(PATH_OUTPUT, exist_ok=True)

if not os.path.exists(PATH_INPUT):
    print(f"\n ERRO: A pasta '{PATH_INPUT}' não foi encontrada!")

    path_projeto = "/content/drive/MyDrive/Projeto_IA"
    if os.path.exists(path_projeto):
        print(f" A pasta 'Projeto_IA' existe, mas 'imagens' não.")
        print(f"   Conteúdo de Projeto_IA: {os.listdir(path_projeto)}")
        print("   Verifique se a pasta se chama 'Imagens' (com I maiúsculo) ou outro nome.")
    else:
        print(" A pasta 'Projeto_IA' também não foi encontrada no MyDrive.")
else:
    print(f"\n Pasta encontrada: {PATH_INPUT}")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50
    cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)

    arquivos = [f for f in os.listdir(PATH_INPUT) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f" Encontradas {len(arquivos)} imagens. Iniciando análise...\n")

    if len(arquivos) == 0:
        print(" A pasta existe, mas está vazia! Coloque as fotos .jpg ou .png lá dentro.")

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

        # v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.8)
        # out = v.draw_instance_predictions(vips.to("cpu"))
        # result_img = out.get_image()[:, :, ::-1]

        # cv2_imshow(result_img) 

    print("\nProcessamento concluído.")
---
8. MÉTRICAS
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np

dataset_path = "/content/drive/MyDrive/Projeto_IA/imagens"

classe_assumida = "ocupada"

y_true = []
y_pred = []
contagem_pessoas = []

print(f" Iniciando análise na pasta: {dataset_path}")
print(f" AVISO: Assumindo que TODAS as imagens são da classe '{classe_assumida}' para teste.")

if not os.path.exists(dataset_path):
    print(" ERRO: A pasta 'imagens' não existe no caminho informado.")
else:
    arquivos = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"--> Processando {len(arquivos)} imagens...")

    for img_name in arquivos:
        img_path = os.path.join(dataset_path, img_name)
        img = cv2.imread(img_path)

        if img is None: continue

        outputs = predictor(img)
        instances = outputs["instances"]

       (classe 0)
        pessoas_detectadas = len(instances[instances.pred_classes == 0])
        contagem_pessoas.append(pessoas_detectadas)

        
        estado_predito = "vazia" if pessoas_detectadas == 0 else "ocupada"

        label_real = 1 # 1 = ocupada (Estamos assumindo que tudo é ocupada)

        label_pred = 0 if estado_predito == "vazia" else 1

        y_true.append(label_real)
        y_pred.append(label_pred)

    if len(y_true) > 0:
        print("\n Processamento concluído!")

        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0],
                    xticklabels=['Vazia', 'Ocupada'], yticklabels=['Vazia', 'Ocupada'])
        ax[0].set_title('Matriz de Confusão')
        ax[0].set_xlabel('Predição da IA')
        ax[0].set_ylabel('Realidade (Assumida)')

        acc = accuracy_score(y_true, y_pred)
        ax[1].bar(['Acurácia'], [acc], color='#4CAF50')
        ax[1].set_title(f'Acurácia: {acc:.2%}')
        ax[1].set_ylim(0, 1.1)

        plt.tight_layout()
        plt.show()

        print(classification_report(y_true, y_pred, target_names=['Vazia', 'Ocupada'], labels=[0, 1], zero_division=0))
    else:
        print(" Nenhuma imagem encontrada na pasta.")
---

##RESULTADOS
 :<img width="409" height="409" alt="download" src="https://github.com/user-attachments/assets/d0ca45f4-7946-4803-9d0e-a4748ac10132" />

<img width="409" height="409" alt="download" src="https://github.com/user-attachments/assets/a4cc240e-f51c-490b-a7a3-623b3d28cca2" />

<img width="409" height="409" alt="download" src="https://github.com/user-attachments/assets/177e9e10-c273-4bf6-aec7-f66597c8af0a" />

