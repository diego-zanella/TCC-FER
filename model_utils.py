import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, DenseNet121_Weights
import timm
import numpy as np
from collections import Counter

import config

def create_model(model_name, num_classes):
    if model_name == 'resnet50':
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'densenet121':
        model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'convnext_tiny':
        model = timm.create_model('convnext_tiny', pretrained=True, num_classes=num_classes)
    elif model_name == 'efficientnet_b0':
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f"Model '{model_name}' not recognized.")

    return model.to(config.DEVICE)

class EnsembleModel:
    def __init__(self, models, strategy='soft'):
        self.models = models
        self.strategy = strategy

    def predict(self, data_loader):
        all_probabilities = []
        true_labels = []

        for model in self.models:
            model.eval()
            model_probs = []
            with torch.no_grad():
                # Only collect true_labels once
                if not true_labels:
                    for data, target in data_loader:
                        data = data.to(config.DEVICE)
                        outputs = model(data)
                        probs = torch.softmax(outputs, dim=1)
                        model_probs.extend(probs.cpu().numpy())
                        true_labels.extend(target.numpy())
                else: # Subsequent models
                    idx = 0
                    for data, target in data_loader:
                        data = data.to(config.DEVICE)
                        outputs = model(data)
                        probs = torch.softmax(outputs, dim=1)
                        model_probs.extend(probs.cpu().numpy())
                        idx += len(target)

            all_probabilities.append(model_probs)

        if self.strategy == 'soft':
            avg_probs = np.mean(all_probabilities, axis=0)
            ensemble_predictions = np.argmax(avg_probs, axis=1)
        else: # hard voting
            all_preds = [np.argmax(probs, axis=1) for probs in all_probabilities]
            all_preds_t = np.array(all_preds).T
            ensemble_predictions = [Counter(preds).most_common(1)[0][0] for preds in all_preds_t]

        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(true_labels, ensemble_predictions)
        return accuracy, ensemble_predictions, true_labels