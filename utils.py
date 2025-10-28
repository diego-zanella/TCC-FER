# utils.py

import os
import logging
import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import seaborn as sns
from sklearn.metrics import confusion_matrix

import config

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_acc):
        if self.best_score is None or val_acc > self.best_score + self.min_delta:
            self.best_score = val_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def setup_logging(log_dir, log_file):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Log file at: {log_path}")

def save_figure(save_path):
    base_path = os.path.splitext(save_path)[0]
    final_path = f"{base_path}.{config.PLOT_FORMAT}"
    plt.savefig(final_path, dpi=300, bbox_inches='tight', format=config.PLOT_FORMAT)
    plt.close()
    logging.info(f"Plot saved to: {final_path}")

def plot_class_distribution(labels, class_names, title, save_path):
    if labels is None or labels.size == 0:
        logging.warning(f"No labels provided for plot '{title}'. Skipping.")
        return
        
    label_counts = Counter(labels)
    sorted_labels = sorted([int(k) for k in label_counts.keys()])
    counts = [label_counts[key] for key in sorted_labels]
    names = [class_names[key] if key < len(class_names) else f"Label {key}" for key in sorted_labels]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=names, y=counts, palette='viridis')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for i, v in enumerate(counts):
        plt.text(i, v + max(counts)*0.01, str(v), ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    save_figure(save_path)

def plot_confusion_matrix(true_labels, predictions, class_names, title, save_path, normalize=None):

    if normalize is None:
        normalize = 'true' if config.NORMALIZE_CM else None
    
    cm = confusion_matrix(true_labels, predictions)
    
    if normalize == 'true':
        # Normalize by row (true labels)
        # Avoid division by zero
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Prevent division by zero
        cm_display = cm.astype('float') / row_sums
        fmt = '.3f'  # Format as decimal between 0-1
        vmax = 1.0
    elif normalize == 'pred':
        # Normalize by column (predicted labels)
        col_sums = cm.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        cm_display = cm.astype('float') / col_sums
        fmt = '.3f'
        vmax = 1.0
    elif normalize == 'all':
        # Normalize by total
        total = cm.sum()
        if total == 0:
            total = 1
        cm_display = cm.astype('float') / total
        fmt = '.3f'
        vmax = 1.0
    else:
        fmt = 'd'
        cm_display = cm
        vmax = None
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm_display, annot=True, fmt=fmt, 
                xticklabels=class_names, yticklabels=class_names, 
                cmap='Blues', 
                cbar_kws={'label': 'Proportion' if normalize else 'Count'},
                vmin=0, vmax=vmax,
                linewidths=0.5, linecolor='gray')
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    save_figure(save_path)

def plot_error_analysis(true_labels, predictions, class_names, title, save_path):
    cm = confusion_matrix(true_labels, predictions)
    n_classes = len(class_names)
    
    # Calculate per-class metrics
    class_totals = cm.sum(axis=1)
    class_correct = np.diag(cm)
    class_errors = class_totals - class_correct
    error_rates = (class_errors / class_totals) * 100
    
    # Find most confused pairs (excluding diagonal)
    confusion_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                # Normalized by true class total
                conf_rate = (cm[i, j] / class_totals[i]) * 100
                confusion_pairs.append((class_names[i], class_names[j], cm[i, j], conf_rate))
    
    confusion_pairs.sort(key=lambda x: x[3], reverse=True)
    top_confusions = confusion_pairs[:10]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Per-class error rates
    ax1 = fig.add_subplot(gs[0, :])
    colors = ['#d62728' if er > 30 else '#ff7f0e' if er > 15 else '#2ca02c' for er in error_rates]
    bars = ax1.bar(class_names, error_rates, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Error Rate (%)', fontsize=11)
    ax1.set_title('Per-Class Error Rates', fontsize=13, fontweight='bold')
    ax1.axhline(y=np.mean(error_rates), color='red', linestyle='--', label=f'Mean: {np.mean(error_rates):.1f}%')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, er in zip(bars, error_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{er:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Top confused pairs
    ax2 = fig.add_subplot(gs[1, :])
    if top_confusions:
        pairs_labels = [f"{true}→{pred}" for true, pred, _, _ in top_confusions]
        pairs_rates = [rate for _, _, _, rate in top_confusions]
        
        bars = ax2.barh(pairs_labels, pairs_rates, color='coral', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Confusion Rate (% of True Class)', fontsize=11)
        ax2.set_title('Top 10 Most Confused Class Pairs', fontsize=13, fontweight='bold')
        ax2.invert_yaxis()
        
        # Add value labels
        for bar, rate in zip(bars, pairs_rates):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{rate:.1f}%', ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Confusion matrix heatmap (off-diagonal only)
    ax3 = fig.add_subplot(gs[2, 0])
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_errors = cm_normalized.copy()
    np.fill_diagonal(cm_errors, 0)  # Remove diagonal
    
    sns.heatmap(cm_errors * 100, annot=True, fmt='.1f', cmap='Reds',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax3, cbar_kws={'label': 'Error %'})
    ax3.set_title('Error Distribution Matrix', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Predicted', fontsize=10)
    ax3.set_ylabel('True', fontsize=10)
    
    # Summary statistics
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')
    
    overall_acc = (class_correct.sum() / class_totals.sum()) * 100
    avg_error = np.mean(error_rates)
    worst_class = class_names[np.argmax(error_rates)]
    best_class = class_names[np.argmin(error_rates)]
    
    summary_text = f"""
    ERROR ANALYSIS SUMMARY
    {'='*35}
    
    Overall Accuracy: {overall_acc:.2f}%
    Average Error Rate: {avg_error:.2f}%
    
    Best Performing Class:
      • {best_class}: {100-error_rates[np.argmin(error_rates)]:.2f}% accuracy
    
    Worst Performing Class:
      • {worst_class}: {100-error_rates[np.argmax(error_rates)]:.2f}% accuracy
    
    Most Critical Confusions:
    """
    
    for i, (true, pred, count, rate) in enumerate(top_confusions[:3], 1):
        summary_text += f"  {i}. {true} → {pred}: {rate:.1f}%\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    save_figure(save_path)
    
    # Log summary to console
    logging.info(f"\n{'='*60}")
    logging.info(f"ERROR ANALYSIS: {title}")
    logging.info(f"{'='*60}")
    logging.info(f"Overall Accuracy: {overall_acc:.2f}%")
    logging.info(f"Average Error Rate: {avg_error:.2f}%")
    logging.info(f"Best Class: {best_class} ({100-error_rates[np.argmin(error_rates)]:.2f}% acc)")
    logging.info(f"Worst Class: {worst_class} ({100-error_rates[np.argmax(error_rates)]:.2f}% acc)")
    logging.info("\nTop 5 Confusions:")
    for i, (true, pred, count, rate) in enumerate(top_confusions[:5], 1):
        logging.info(f"  {i}. {true} → {pred}: {rate:.1f}% (n={count})")

def plot_cross_dataset_results(cross_results, save_path, training_type='individual'):
    if not cross_results:
        logging.warning("No cross-dataset results to plot.")
        return
    
    train_datasets = list(cross_results.keys())
    test_datasets = list(next(iter(cross_results.values())).keys())
    models = list(next(iter(next(iter(cross_results.values())).values())).keys())
    
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6), squeeze=False)
    
    for idx, model_name in enumerate(models):
        # Create matrix: rows=train_datasets, cols=test_datasets
        data = np.zeros((len(train_datasets), len(test_datasets)))
        
        for i, train_ds in enumerate(train_datasets):
            for j, test_ds in enumerate(test_datasets):
                data[i, j] = cross_results[train_ds][test_ds].get(model_name, 0.0)
        
        ax = axes[0, idx]
        sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', 
                    xticklabels=test_datasets, yticklabels=train_datasets, 
                    ax=ax, vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'},
                    linewidths=0.5, linecolor='gray')
        
        # Highlight diagonal (same dataset train/test)
        for i in range(min(len(train_datasets), len(test_datasets))):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, 
                                      edgecolor='blue', lw=3))
        
        ax.set_title(f'{model_name}\n({training_type.capitalize()} Training)', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Test Dataset', fontsize=11)
        if idx == 0:
            ax.set_ylabel('Training Dataset', fontsize=11)
        else:
            ax.set_ylabel('')
    
    plt.suptitle('Cross-Dataset Evaluation', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(save_path)

def create_comprehensive_results_table(individual_results, merged_results, ensemble_individual, ensemble_merged, save_path):
    data_rows = []
    
    # Section 1: Individual Model Results
    logging.info("\n" + "="*80)
    logging.info("SECTION 1: Individual Dataset Training Results")
    logging.info("="*80)
    
    for dataset, models in individual_results.items():
        for model, metrics in models.items():
            data_rows.append({
                'Section': '1. Individual Training',
                'Training Data': dataset,
                'Model': model,
                'Test Dataset': dataset,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Accuracy (TTA)': f"{metrics['accuracy_tta']:.4f}",
                'Type': 'Individual Model'
            })
    
    # Section 2: Merged Model Results
    logging.info("\n" + "="*80)
    logging.info("SECTION 2: Merged Dataset Training Results")
    logging.info("="*80)
    
    for dataset, models in merged_results.items():
        for model, metrics in models.items():
            data_rows.append({
                'Section': '2. Merged Training',
                'Training Data': 'ALL (Merged)',
                'Model': model,
                'Test Dataset': dataset,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Accuracy (TTA)': f"{metrics['accuracy_tta']:.4f}",
                'Type': 'Merged Model'
            })
    
    # Section 3: Ensemble Results - Individual
    logging.info("\n" + "="*80)
    logging.info("SECTION 3: Ensemble Results (Individual Training)")
    logging.info("="*80)
    
    for dataset, metrics in ensemble_individual.items():
        data_rows.append({
            'Section': '3. Ensemble (Individual)',
            'Training Data': dataset,
            'Model': 'ENSEMBLE (All Models)',
            'Test Dataset': dataset,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Accuracy (TTA)': '-',
            'Type': 'Ensemble'
        })
    
    # Section 4: Ensemble Results - Merged
    logging.info("\n" + "="*80)
    logging.info("SECTION 4: Ensemble Results (Merged Training)")
    logging.info("="*80)
    
    for dataset, metrics in ensemble_merged.items():
        data_rows.append({
            'Section': '4. Ensemble (Merged)',
            'Training Data': 'ALL (Merged)',
            'Model': 'ENSEMBLE (All Models)',
            'Test Dataset': dataset,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Accuracy (TTA)': '-',
            'Type': 'Ensemble'
        })
    
    df = pd.DataFrame(data_rows)
    
    # Save as CSV
    csv_path = save_path.replace('.txt', '.csv')
    df.to_csv(csv_path, index=False)
    logging.info(f"Results table saved to: {csv_path}")
    
    # Log formatted table
    logging.info("\n" + "="*80)
    logging.info("COMPREHENSIVE RESULTS TABLE")
    logging.info("="*80 + "\n")
    logging.info(df.to_string(index=False))
    
    return df