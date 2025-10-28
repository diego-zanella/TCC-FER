import os
import time
import gc
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
import torch

import config
import data_loader
import model_utils
import training
import utils

def train_individual_models(datasets_data, num_classes):
    results = {}
    trained_models = {}
    
    logging.info("\n" + "="*80)
    logging.info("PHASE 1: INDIVIDUAL DATASET TRAINING")
    logging.info("="*80)
    
    for dataset_name, data in datasets_data.items():
        logging.info(f"\n{'='*80}\nProcessing Dataset: {dataset_name}\n{'='*80}")
        
        train_paths, train_labels = data['train']
        test_paths, test_labels = data['test']
        
        # Plot original distribution
        utils.plot_class_distribution(
            train_labels, config.EMOTION_CLASSES, 
            f"Original Distribution - {dataset_name}", 
            os.path.join(config.DISTRIBUTION_DIR, f"{dataset_name}_dist_original")
        )
        
        # Balance data
        train_paths_b, train_labels_b = data_loader.balance_data_oversampling(
            train_paths, train_labels
        )
        
        # Plot balanced distribution
        utils.plot_class_distribution(
            train_labels_b, config.EMOTION_CLASSES, 
            f"Balanced Distribution - {dataset_name}",
            os.path.join(config.DISTRIBUTION_DIR, f"{dataset_name}_dist_balanced")
        )
        
        # Create datasets
        train_dataset = data_loader.LazyLoadDataset(
            train_paths_b, train_labels_b, 
            transform=config.TRANSFORM_TRAIN_HEAVY
        )
        test_dataset = data_loader.LazyLoadDataset(
            test_paths, test_labels, 
            transform=config.TRANSFORM_TEST
        )
        
        results[dataset_name] = {}
        trained_models[dataset_name] = {}
        
        # Train each model
        for model_name, model_params in config.MODEL_CONFIG.items():
            torch.cuda.empty_cache()
            batch_size = model_params['batch_size']
            
            logging.info(f"\n--- Training {model_name} on {dataset_name} (Batch: {batch_size}) ---")
            
            model = model_utils.create_model(model_name, num_classes)
            model_path = os.path.join(
                config.MODEL_SAVE_DIR, 
                f"{dataset_name}_{model_name}.pth"
            )
            
            # Check if model exists
            if os.path.exists(model_path):
                logging.info(f"Loading pre-trained model from {model_path}")
                model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
                training_time = 0
                history = {}
            else:
                logging.info("Training new model...")
                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, 
                    shuffle=True, num_workers=4, pin_memory=True
                )
                val_loader = DataLoader(
                    test_dataset, batch_size=batch_size, 
                    shuffle=False, num_workers=4, pin_memory=True
                )
                
                start_time = time.time()
                history = training.train_model(model, train_loader, val_loader)
                training_time = time.time() - start_time
                
                logging.info(f"Saving model to {model_path}")
                torch.save(model.state_dict(), model_path)
            
            # Evaluate
            eval_loader = DataLoader(
                test_dataset, batch_size=batch_size, 
                shuffle=False, num_workers=4, pin_memory=True
            )
            
            acc, preds, true_lbls = training.evaluate_model(
                model, eval_loader, use_tta=False
            )
            acc_tta, preds_tta, _ = training.evaluate_model(
                model, eval_loader, use_tta=True
            )
            
            logging.info(f"{model_name} | Acc: {acc:.4f} | Acc (TTA): {acc_tta:.4f}")
            
            # Store results
            results[dataset_name][model_name] = {
                'accuracy': acc,
                'accuracy_tta': acc_tta,
                'training_time': training_time,
                'history': history
            }
            trained_models[dataset_name][model_name] = model
            
            # Generate visualizations
            utils.plot_confusion_matrix(
                true_lbls, preds_tta, config.EMOTION_CLASSES,
                f'Confusion Matrix: {model_name} on {dataset_name} (TTA)',
                os.path.join(config.TTA_DIR, f"{dataset_name}_{model_name}_cm_TTA")
            )
            
            utils.plot_error_analysis(
                true_lbls, preds_tta, config.EMOTION_CLASSES,
                f'Error Analysis: {model_name} on {dataset_name}',
                os.path.join(config.ERROR_ANALYSIS_DIR, f"{dataset_name}_{model_name}_error_analysis")
            )
            
            # Classification report
            logging.info("\n" + classification_report(
                true_lbls, preds_tta, 
                target_names=config.EMOTION_CLASSES, 
                digits=3
            ))
        
        del train_dataset, test_dataset
        gc.collect()
        torch.cuda.empty_cache()
    
    return results, trained_models

def train_merged_model(datasets_data, num_classes):
    logging.info("\n" + "="*80)
    logging.info("PHASE 2: MERGED DATASET TRAINING")
    logging.info("="*80)
    
    # Merge all training data
    all_train_paths, all_train_labels = [], []
    for name, data in datasets_data.items():
        train_paths, train_labels = data['train']
        all_train_paths.extend(train_paths)
        all_train_labels.extend(train_labels)
    
    logging.info(f"Total combined training samples: {len(all_train_paths)}")
    
    # Balance merged data
    all_train_paths_b, all_train_labels_b = data_loader.balance_data_oversampling(
        all_train_paths, 
        np.array(all_train_labels)
    )
    
    utils.plot_class_distribution(
        all_train_labels_b, config.EMOTION_CLASSES,
        "Combined Dataset Distribution (Balanced)",
        os.path.join(config.DISTRIBUTION_DIR, "merged_train_distribution")
    )
    
    # Create merged training dataset
    merged_train_dataset = data_loader.LazyLoadDataset(
        all_train_paths_b, all_train_labels_b,
        transform=config.TRANSFORM_TRAIN_HEAVY
    )
    
    if len(merged_train_dataset) == 0:
        logging.error("Merged dataset is empty!")
        return {}, {}
    
    results = {}
    trained_models = {}
    
    # Train each model on merged data
    for model_name, model_params in config.MODEL_CONFIG.items():
        torch.cuda.empty_cache()
        batch_size = model_params['batch_size']
        
        logging.info(f"\n--- Training {model_name} on MERGED dataset (Batch: {batch_size}) ---")
        
        model = model_utils.create_model(model_name, num_classes)
        model_path = os.path.join(config.MODEL_SAVE_DIR, f"merged_{model_name}.pth")
        
        # Check if model exists
        if os.path.exists(model_path):
            logging.info(f"Loading pre-trained merged model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
            training_time = 0
            history = {}
        else:
            logging.info("Training new merged model...")
            train_loader = DataLoader(
                merged_train_dataset, batch_size=batch_size,
                shuffle=True, num_workers=4, pin_memory=True
            )
            
            # Use first dataset's test set for validation
            val_paths, val_labels = datasets_data[list(datasets_data.keys())[0]]['test']
            val_dataset = data_loader.LazyLoadDataset(
                val_paths, val_labels, transform=config.TRANSFORM_TEST
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size,
                shuffle=False, num_workers=4, pin_memory=True
            )
            
            start_time = time.time()
            history = training.train_model(model, train_loader, val_loader)
            training_time = time.time() - start_time
            
            logging.info(f"Saving merged model to {model_path}")
            torch.save(model.state_dict(), model_path)
        
        # Evaluate merged model on each dataset
        for dataset_name, data in datasets_data.items():
            test_paths, test_labels = data['test']
            test_dataset = data_loader.LazyLoadDataset(
                test_paths, test_labels, transform=config.TRANSFORM_TEST
            )
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size,
                shuffle=False, num_workers=4, pin_memory=True
            )
            
            acc, preds, true_lbls = training.evaluate_model(
                model, test_loader, use_tta=False
            )
            acc_tta, preds_tta, _ = training.evaluate_model(
                model, test_loader, use_tta=True
            )
            
            logging.info(f"{model_name} on {dataset_name} | Acc: {acc:.4f} | Acc (TTA): {acc_tta:.4f}")
            
            # Store results per dataset
            if dataset_name not in results:
                results[dataset_name] = {}
            
            results[dataset_name][model_name] = {
                'accuracy': acc,
                'accuracy_tta': acc_tta,
                'training_time': training_time if dataset_name == list(datasets_data.keys())[0] else 0
            }
            
            # Generate visualizations
            utils.plot_confusion_matrix(
                true_lbls, preds_tta, config.EMOTION_CLASSES,
                f'Confusion Matrix: {model_name} (Merged) on {dataset_name} (TTA)',
                os.path.join(config.TTA_DIR, f"merged_{model_name}_{dataset_name}_cm_TTA")
            )
            
            utils.plot_error_analysis(
                true_lbls, preds_tta, config.EMOTION_CLASSES,
                f'Error Analysis: {model_name} (Merged) on {dataset_name}',
                os.path.join(config.ERROR_ANALYSIS_DIR, f"merged_{model_name}_{dataset_name}_error_analysis")
            )
            
            logging.info("\n" + classification_report(
                true_lbls, preds_tta,
                target_names=config.EMOTION_CLASSES,
                digits=3
            ))
            
            del test_dataset, test_loader
            gc.collect()
        
        trained_models[model_name] = model
    
    return results, trained_models

def evaluate_cross_dataset(individual_models, merged_models, datasets_data):
    logging.info("\n" + "="*80)
    logging.info("PHASE 3: CROSS-DATASET EVALUATION")
    logging.info("="*80)
    
    active_datasets = list(datasets_data.keys())
    
    # Cross-dataset for individual models
    cross_individual = {train_ds: {test_ds: {} for test_ds in active_datasets} 
                       for train_ds in active_datasets}
    
    logging.info("\n--- Cross-Dataset: Individual Training ---")
    for train_ds in active_datasets:
        for test_ds in active_datasets:
            logging.info(f"\nEvaluating models trained on {train_ds}, tested on {test_ds}...")
            test_paths, test_labels = datasets_data[test_ds]['test']
            test_dataset = data_loader.LazyLoadDataset(
                test_paths, test_labels, transform=config.TRANSFORM_TEST
            )
            
            for model_name, model in individual_models[train_ds].items():
                batch_size = config.MODEL_CONFIG[model_name]['batch_size']
                test_loader = DataLoader(
                    test_dataset, batch_size=batch_size,
                    shuffle=False, num_workers=4, pin_memory=True
                )
                
                acc, _, _ = training.evaluate_model(model, test_loader, use_tta=False)
                cross_individual[train_ds][test_ds][model_name] = acc
                logging.info(f"  {model_name}: {acc:.4f}")
            
            del test_dataset, test_loader
            gc.collect()
    
    # Cross-dataset for merged models
    cross_merged = {'MERGED': {test_ds: {} for test_ds in active_datasets}}
    
    logging.info("\n--- Cross-Dataset: Merged Training ---")
    for test_ds in active_datasets:
        logging.info(f"\nEvaluating merged models on {test_ds}...")
        test_paths, test_labels = datasets_data[test_ds]['test']
        test_dataset = data_loader.LazyLoadDataset(
            test_paths, test_labels, transform=config.TRANSFORM_TEST
        )
        
        for model_name, model in merged_models.items():
            batch_size = config.MODEL_CONFIG[model_name]['batch_size']
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size,
                shuffle=False, num_workers=4, pin_memory=True
            )
            
            acc, _, _ = training.evaluate_model(model, test_loader, use_tta=False)
            cross_merged['MERGED'][test_ds][model_name] = acc
            logging.info(f"  {model_name}: {acc:.4f}")
        
        del test_dataset, test_loader
        gc.collect()
    
    # Plot cross-dataset results
    utils.plot_cross_dataset_results(
        cross_individual,
        os.path.join(config.CROSS_DATASET_DIR, 'cross_dataset_individual'),
        training_type='individual'
    )
    
    utils.plot_cross_dataset_results(
        cross_merged,
        os.path.join(config.CROSS_DATASET_DIR, 'cross_dataset_merged'),
        training_type='merged'
    )
    
    return cross_individual, cross_merged

def evaluate_ensembles(individual_models, merged_models, datasets_data):
    logging.info("\n" + "="*80)
    logging.info("PHASE 4: ENSEMBLE EVALUATION")
    logging.info("="*80)
    
    ensemble_individual = {}
    ensemble_merged = {}
    
    # Ensemble for individual training (per dataset)
    logging.info("\n--- Ensemble: Individual Training ---")
    for dataset_name in datasets_data.keys():
        logging.info(f"\nEvaluating ensemble for {dataset_name} (individual training)...")
        
        # Create ensemble from models trained on this dataset
        models_list = [model for model in individual_models[dataset_name].values()]
        ensemble = model_utils.EnsembleModel(models_list, strategy='soft')
        
        test_paths, test_labels = datasets_data[dataset_name]['test']
        test_dataset = data_loader.LazyLoadDataset(
            test_paths, test_labels, transform=config.TRANSFORM_TEST
        )
        
        # Use batch size from first model
        batch_size = config.MODEL_CONFIG[list(config.MODEL_CONFIG.keys())[0]]['batch_size']
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=4, pin_memory=True
        )
        
        acc, preds, true_lbls = ensemble.predict(test_loader)
        logging.info(f"Ensemble Accuracy on {dataset_name}: {acc:.4f}")
        
        ensemble_individual[dataset_name] = {'accuracy': acc}
        
        # Generate visualizations
        utils.plot_confusion_matrix(
            true_lbls, preds, config.EMOTION_CLASSES,
            f'Ensemble Confusion Matrix: {dataset_name} (Individual Training)',
            os.path.join(config.ENSEMBLE_DIR, f"ensemble_individual_{dataset_name}_cm")
        )
        
        utils.plot_error_analysis(
            true_lbls, preds, config.EMOTION_CLASSES,
            f'Ensemble Error Analysis: {dataset_name} (Individual Training)',
            os.path.join(config.ENSEMBLE_DIR, f"ensemble_individual_{dataset_name}_error")
        )
        
        logging.info("\n" + classification_report(
            true_lbls, preds,
            target_names=config.EMOTION_CLASSES,
            digits=3
        ))
        
        del test_dataset, test_loader
        gc.collect()
    
    # Ensemble for merged training
    logging.info("\n--- Ensemble: Merged Training ---")
    models_list = [model for model in merged_models.values()]
    ensemble = model_utils.EnsembleModel(models_list, strategy='soft')
    
    for dataset_name, data in datasets_data.items():
        logging.info(f"\nEvaluating merged ensemble on {dataset_name}...")
        
        test_paths, test_labels = data['test']
        test_dataset = data_loader.LazyLoadDataset(
            test_paths, test_labels, transform=config.TRANSFORM_TEST
        )
        
        batch_size = config.MODEL_CONFIG[list(config.MODEL_CONFIG.keys())[0]]['batch_size']
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=4, pin_memory=True
        )
        
        acc, preds, true_lbls = ensemble.predict(test_loader)
        logging.info(f"Merged Ensemble Accuracy on {dataset_name}: {acc:.4f}")
        
        ensemble_merged[dataset_name] = {'accuracy': acc}
        
        # Generate visualizations
        utils.plot_confusion_matrix(
            true_lbls, preds, config.EMOTION_CLASSES,
            f'Ensemble Confusion Matrix: {dataset_name} (Merged Training)',
            os.path.join(config.ENSEMBLE_DIR, f"ensemble_merged_{dataset_name}_cm")
        )
        
        utils.plot_error_analysis(
            true_lbls, preds, config.EMOTION_CLASSES,
            f'Ensemble Error Analysis: {dataset_name} (Merged Training)',
            os.path.join(config.ENSEMBLE_DIR, f"ensemble_merged_{dataset_name}_error")
        )
        
        logging.info("\n" + classification_report(
            true_lbls, preds,
            target_names=config.EMOTION_CLASSES,
            digits=3
        ))
        
        del test_dataset, test_loader
        gc.collect()
    
    return ensemble_individual, ensemble_merged

def main():
    gc.collect()
    torch.cuda.empty_cache()
    
    # Setup
    log_filename = f"execution_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    utils.setup_logging(config.OUTPUT_DIR, log_filename)
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.CONFUSION_MATRIX_DIR, exist_ok=True)
    os.makedirs(config.TTA_DIR, exist_ok=True)
    os.makedirs(config.ERROR_ANALYSIS_DIR, exist_ok=True)
    os.makedirs(config.DISTRIBUTION_DIR, exist_ok=True)
    os.makedirs(config.CROSS_DATASET_DIR, exist_ok=True)
    os.makedirs(config.ENSEMBLE_DIR, exist_ok=True)
    
    logging.info(f"Device: {config.DEVICE}")
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logging.info(f"Training Strategy: {config.TRAINING_STRATEGY}")
    logging.info(f"Output Format: {config.PLOT_FORMAT}")
    logging.info(f"Normalize CM: {config.NORMALIZE_CM}")
    
    # Load datasets
    logging.info("\n" + "="*80)
    logging.info("LOADING DATASETS")
    logging.info("="*80)
    
    datasets_data = {}
    for name, func_name in config.ACTIVE_DATASETS.items():
        logging.info(f"\nLoading {name}...")
        load_func = getattr(data_loader, func_name)
        (train_paths, train_labels), (test_paths, test_labels) = load_func()
        
        if len(train_paths) == 0:
            logging.warning(f"Skipping {name} - empty dataset")
            continue
        
        datasets_data[name] = {
            'train': (train_paths, train_labels),
            'test': (test_paths, test_labels)
        }
        logging.info(f"  Train: {len(train_paths)} | Test: {len(test_paths)}")
    
    if not datasets_data:
        logging.error("No datasets loaded. Exiting.")
        return
    
    num_classes = len(config.EMOTION_CLASSES)
    
    # Execute training based on strategy
    individual_results = {}
    individual_models = {}
    merged_results = {}
    merged_models = {}
    
    if config.TRAINING_STRATEGY in ['individual', 'both']:
        individual_results, individual_models = train_individual_models(
            datasets_data, num_classes
        )
    
    if config.TRAINING_STRATEGY in ['merged', 'both']:
        merged_results, merged_models = train_merged_model(
            datasets_data, num_classes
        )
    
    # Cross-dataset evaluation
    if config.TRAINING_STRATEGY == 'both':
        cross_individual, cross_merged = evaluate_cross_dataset(
            individual_models, merged_models, datasets_data
        )
    
    # Ensemble evaluation
    ensemble_individual = {}
    ensemble_merged = {}
    
    if config.TRAINING_STRATEGY in ['individual', 'both'] and individual_models:
        if config.TRAINING_STRATEGY == 'both':
            ensemble_individual, ensemble_merged = evaluate_ensembles(
                individual_models, merged_models, datasets_data
            )
        else:
            # Only individual ensemble
            for dataset_name in datasets_data.keys():
                models_list = [m for m in individual_models[dataset_name].values()]
                ensemble = model_utils.EnsembleModel(models_list, strategy='soft')
                
                test_paths, test_labels = datasets_data[dataset_name]['test']
                test_dataset = data_loader.LazyLoadDataset(
                    test_paths, test_labels, transform=config.TRANSFORM_TEST
                )
                batch_size = config.MODEL_CONFIG[list(config.MODEL_CONFIG.keys())[0]]['batch_size']
                test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                       shuffle=False, num_workers=4, pin_memory=True)
                
                acc, preds, true_lbls = ensemble.predict(test_loader)
                ensemble_individual[dataset_name] = {'accuracy': acc}
    
    elif config.TRAINING_STRATEGY == 'merged' and merged_models:
        # Only merged ensemble
        models_list = [m for m in merged_models.values()]
        ensemble = model_utils.EnsembleModel(models_list, strategy='soft')
        
        for dataset_name, data in datasets_data.items():
            test_paths, test_labels = data['test']
            test_dataset = data_loader.LazyLoadDataset(
                test_paths, test_labels, transform=config.TRANSFORM_TEST
            )
            batch_size = config.MODEL_CONFIG[list(config.MODEL_CONFIG.keys())[0]]['batch_size']
            test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                   shuffle=False, num_workers=4, pin_memory=True)
            
            acc, preds, true_lbls = ensemble.predict(test_loader)
            ensemble_merged[dataset_name] = {'accuracy': acc}
    
    # Generate comprehensive results table
    logging.info("\n" + "="*80)
    logging.info("GENERATING COMPREHENSIVE RESULTS")
    logging.info("="*80)
    
    utils.create_comprehensive_results_table(
        individual_results,
        merged_results,
        ensemble_individual,
        ensemble_merged,
        os.path.join(config.OUTPUT_DIR, 'comprehensive_results.txt')
    )
    
    logging.info("\n" + "="*80)
    logging.info("EXECUTION COMPLETED SUCCESSFULLY")
    logging.info("="*80)
    logging.info(f"All results saved to: {config.OUTPUT_DIR}/")

if __name__ == '__main__':
    main()