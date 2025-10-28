import os
import logging
import numpy as np
from PIL import Image
from collections import Counter

import torch
from torch.utils.data import Dataset
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

# Mapping from config.py
emotion_map = {
    'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6,
    'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Sad': 4, 'Surprise': 5, 'Neutral': 6
}

class LazyLoadDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except (IOError, FileNotFoundError) as e:
            logging.error(f"Error loading image {img_path}: {e}")
            # Return a dummy tensor and the label
            return torch.zeros((3, 224, 224)), label


def balance_data_oversampling(images_or_paths, labels):
    logging.info(f"Balanceando dados com RandomOversampler...")
    
    if labels is None or labels.size == 0:
        logging.warning("Aviso: o array de rótulos está vazio. Pulando oversampling.")
        return images_or_paths, np.array(labels) # Return an empty numpy array for labels

    # Convert to integer type before Counter to avoid issues with numpy types
    int_labels = labels.astype(int)
    logging.info(f"Distribuição de classes original: {sorted(Counter(int_labels).items())}")

    is_paths = isinstance(images_or_paths, list)
    if is_paths:
        X = np.array(images_or_paths).reshape(-1, 1)
    else:
        original_shape = images_or_paths.shape
        X = images_or_paths.reshape(original_shape[0], -1)

    y = np.array(labels)
    if X.shape[0] == 0 or y.shape[0] == 0:
        logging.warning("Aviso: Encontradas 0 amostras em X ou y. Pulando oversampling.")
        return images_or_paths, y

    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    
    # Also convert resampled labels to int for Counter
    logging.info(f"Distribuição de classes reamostrada: {sorted(Counter(y_resampled.astype(int)).items())}")

    if is_paths:
        images_balanced = X_resampled.flatten().tolist()
    else:
        images_balanced = X_resampled.reshape(-1, *original_shape[1:])
        
    return images_balanced, y_resampled

def load_fer2013():
    train_dir, test_dir = './fer2013/train', './fer2013/test'
    fer_emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    fer_map = {name: emotion_map[name.capitalize()] for name in fer_emotion_classes}

    def load_from_dir(directory):
      paths, labels = [], []
      for emotion_name in fer_emotion_classes:
          emotion_dir = os.path.join(directory, emotion_name)
          if os.path.exists(emotion_dir):
              for img_name in os.listdir(emotion_dir):
                  paths.append(os.path.join(emotion_dir, img_name))
                  labels.append(fer_map[emotion_name])
      return paths, np.array(labels)

    train_paths, train_labels = load_from_dir(train_dir)
    test_paths, test_labels = load_from_dir(test_dir)
    return (train_paths, train_labels), (test_paths, test_labels)

def load_rafdb():
    rafdb_map = {'1': 'surprise', '2': 'fear', '3': 'disgust', '4': 'happy', '5': 'sad', '6': 'angry', '7': 'neutral'}
    train_dir, test_dir = "./rafdb/DATASET/train", "./rafdb/DATASET/test"

    def load_from_dir(directory):
        paths, labels = [], []
        if not os.path.exists(directory): return paths, np.array(labels)
        for folder, emotion in rafdb_map.items():
            emotion_dir = os.path.join(directory, folder)
            if not os.path.exists(emotion_dir): continue
            label = emotion_map[emotion.capitalize()]
            for img in os.listdir(emotion_dir):
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    paths.append(os.path.join(emotion_dir, img))
                    labels.append(label)
        return paths, np.array(labels)

    train_paths, train_labels = load_from_dir(train_dir)
    test_paths, test_labels = load_from_dir(test_dir)
    return (train_paths, train_labels), (test_paths, test_labels)

def load_expw():
    base_dir = './expw/Expw-F' 
    
    expw_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    image_paths = []
    labels = []

    if not os.path.exists(base_dir):
        logging.error(f"FATAL: ExpW directory not found at the expected path: {base_dir}")
        logging.error("Please ensure the 'Expw-F' folder is inside the 'expw' folder.")
        return ([], np.array([])), ([], np.array([]))

    logging.info(f"Loading images from ExpW dataset at: {base_dir}")
    for emotion_name in expw_classes:
        emotion_dir = os.path.join(base_dir, emotion_name)
        if os.path.exists(emotion_dir):
            label = emotion_map.get(emotion_name.capitalize())
            if label is None:
                logging.warning(f"Emotion '{emotion_name}' not in emotion_map. Skipping.")
                continue
            
            count = 0
            for img_name in os.listdir(emotion_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(emotion_dir, img_name))
                    labels.append(label)
                    count += 1
            logging.info(f"  Found {count} images for '{emotion_name}'.")
        else:
            logging.warning(f"Directory for emotion '{emotion_name}' not found at: {emotion_dir}")

    if not image_paths:
        logging.error("No images were loaded from the ExpW dataset. Check paths and folder structure.")
        return ([], np.array([])), ([], np.array([]))

    logging.info(f"Total images loaded from ExpW: {len(image_paths)}")
    
    # Create a stratified train/test split because the dataset doesn't provide one.
    # This ensures both train and test sets have a similar class distribution.
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, 
        test_size=0.2,       # Using 20% of the data for testing
        random_state=42,     # For reproducibility
        stratify=labels      # CRITICAL for imbalanced datasets
    )
    
    logging.info(f"ExpW split into {len(train_paths)} training images and {len(test_paths)} testing images.")
    
    return (train_paths, np.array(train_labels)), (test_paths, np.array(test_labels))