import logging
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from PIL import Image

import config
from utils import EarlyStopping

def tensor_to_pil(tensor, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    t = tensor.clone()
    for c in range(t.shape[0]):
        t[c] = t[c] * std[c] + mean[c]
    t = torch.clamp(t, 0.0, 1.0)
    npimg = (t.permute(1, 2, 0).numpy() * 255.0).astype('uint8')
    return Image.fromarray(npimg)

def train_model(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=config.PATIENCE)
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        for data, target in train_loader:
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        
        val_acc, _, _ = evaluate_model(model, val_loader, use_tta=False)
        val_acc *= 100 # Convert to percentage for logging
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        logging.info(f'Epoch {epoch+1}/{config.EPOCHS} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%')
        scheduler.step(val_acc)
        early_stopping(val_acc)
        if early_stopping.early_stop:
            logging.info(f"Early stopping triggered at epoch {epoch+1}")
            break
            
    return history

def evaluate_model(model, test_loader, use_tta=False):
    model.eval()
    predictions, true_labels = [], []

    if use_tta:
        logging.info("Using Test-Time Augmentation...")
        tta_transforms = config.get_tta_transforms()
        with torch.no_grad():
            for data, target in test_loader:
                batch_size = data.size(0)
                probs_per_aug = []
                for transform in tta_transforms:
                    augmented_list = [transform(tensor_to_pil(data[i])) for i in range(batch_size)]
                    augmented_batch = torch.stack(augmented_list).to(config.DEVICE)
                    outputs = model(augmented_batch)
                    probs_per_aug.append(torch.softmax(outputs, dim=1).cpu())
                
                mean_probs = torch.mean(torch.stack(probs_per_aug), dim=0)
                _, predicted = torch.max(mean_probs, 1)
                predictions.extend(predicted.numpy())
                true_labels.extend(target.numpy())
    else:
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(config.DEVICE), target.to(config.DEVICE)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(target.cpu().numpy())
                
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy, predictions, true_labels