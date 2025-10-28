import torch
import torchvision.transforms as transforms
import os

# --- General Settings ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = 'saidas'
LOG_FILE = 'execution_log.txt'
EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, 'saved_models')

# --- Output subdirectories ---
CONFUSION_MATRIX_DIR = os.path.join(OUTPUT_DIR, 'confusion_matrices')
TTA_DIR = os.path.join(CONFUSION_MATRIX_DIR, 'TTA')
ERROR_ANALYSIS_DIR = os.path.join(OUTPUT_DIR, 'error_analysis')
DISTRIBUTION_DIR = os.path.join(OUTPUT_DIR, 'class_distributions')
CROSS_DATASET_DIR = os.path.join(OUTPUT_DIR, 'cross_dataset')
ENSEMBLE_DIR = os.path.join(OUTPUT_DIR, 'ensemble_results')

# --- Output Format ---
PLOT_FORMAT = 'pdf'  # 'pdf' or 'png'
NORMALIZE_CM = True   # Normalize confusion matrices

# --- Training Strategy ---
# 'individual': Train separate models per dataset
# 'merged': Train single model on combined dataset
# 'both': Train both individual and merged models
TRAINING_STRATEGY = 'both'

# --- Dataset Configuration ---
ACTIVE_DATASETS = {
    'RAF-DB': 'load_rafdb',
    'ExpW': 'load_expw',
    'FER2013': 'load_fer2013',
}

# --- Model & Training Configuration ---
'''
O tamanho do batch é ajustado com base na arquitetura do modelo, e o seu hardware, para otimizar o uso da memória GPU.
Os valores utilizados abaixo foram valores que funcionaram bem em uma GPU NVIDIA RTX 5070 ti com 16GB de VRAM.
Teste bem antes de utilizar esses valores em seu próprio hardware, pois eles influenciam muito o tempo de treinamento.

Recomendamos utilizar um CMD rodando o comando "nvidia-smi -l 1" para monitorar o uso da memória GPU durante o treinamento.
A pratica que utilizamos para monitorar se estava adequado era observar o uso de energia da GPU, que deveria estar próximo do 
máximo permitido pela GPU. Se o uso de energia caísse consideravelmente, era um indicativo de que o batch estava muito grande, 
e a GPU estava utilizando um mecanismo de "swapping" para lidar com a falta de memória, o que reduz consideravelmente o desempenho.
'''
MODEL_CONFIG = {
    'densenet121': {'batch_size': 96},
    'resnet50': {'batch_size': 128},
    'efficientnet_b0': {'batch_size': 128},
}

# --- Hyperparameters ---
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 5

# --- Data Augmentation Transforms ---
TRANSFORM_TRAIN_HEAVY = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

TRANSFORM_TEST = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_tta_transforms():
    return [
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(200),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    ]