
import logging
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from transformers import get_linear_schedule_with_warmup
except ImportError:
    def get_linear_schedule_with_warmup(*args, **kwargs):
        return None
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from utils.logging_utils import get_logger

logger = get_logger(__name__)

class BaseTrainer:
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        torch.manual_seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)
        
        if config.WANDB_API_KEY:
            try:
                import wandb
                wandb.init(
                    project=config.WANDB_PROJECT,
                    config=vars(config)
                )
            except ImportError:
                logger.warning("wandb not installed, skipping initialization")
    
    def setup_optimizer_and_scheduler(self, model, train_dataloader):
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        logger.info(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.LEARNING_RATE,
            weight_decay=0.01
        )
        
        total_steps = len(train_dataloader) * self.config.NUM_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.WARMUP_STEPS,
            num_training_steps=total_steps
        )
        
        return optimizer, scheduler
    
    def train_epoch(self, model, train_dataloader, optimizer, scheduler, epoch):
        model.train()
        total_loss = 0
        num_batches = len(train_dataloader)
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            outputs = self._forward_pass(model, batch)
            loss = outputs['loss']
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if (step + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            avg_loss = total_loss / (step + 1)
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            
            if self.config.WANDB_API_KEY and step % self.config.LOGGING_STEPS == 0:
                try:
                    import wandb
                    wandb.log({
                        'train_loss': loss.item(),
                        'learning_rate': scheduler.get_last_lr()[0],
                        'epoch': epoch,
                        'step': step
                    })
                except ImportError:
                    pass
        
        return total_loss / num_batches
    
    def evaluate(self, model, eval_dataloader, epoch=None):
        model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self._forward_pass(model, batch)
                loss = outputs['loss']
                logits = outputs['logits']
                
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(eval_dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        metrics = {
            'eval_loss': avg_loss,
            'eval_accuracy': accuracy,
            'eval_precision': precision,
            'eval_recall': recall,
            'eval_f1': f1
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        if self.config.WANDB_API_KEY and epoch is not None:
            try:
                import wandb
                wandb.log({**metrics, 'epoch': epoch})
            except ImportError:
                pass
        
        return metrics
    
    def save_model(self, model, path, metrics=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': vars(self.config),
            'metrics': metrics
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, model, path):
        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")
        
        return checkpoint.get('metrics', {})
    
    def _forward_pass(self, model, batch):
        raise NotImplementedError("Subclasses must implement _forward_pass method")
    
    def train(self, data_loader):
        raise NotImplementedError("Subclasses must implement train method")

class EarlyStopping:
    
    def __init__(self, patience=3, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, current_loss, model):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
            
        return False

class MetricsTracker:
    
    def __init__(self):
        self.metrics_history = {
            'train_loss': [],
            'eval_loss': [],
            'eval_accuracy': [],
            'eval_f1': []
        }
    
    def update(self, metrics):
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
    
    def get_best_metric(self, metric_name):
        if metric_name not in self.metrics_history:
            return None
            
        values = self.metrics_history[metric_name]
        if not values:
            return None
            
        if 'loss' in metric_name:
            return min(values)
        else:
            return max(values)
    
    def get_latest_metrics(self):
        latest = {}
        for key, values in self.metrics_history.items():
            if values:
                latest[key] = values[-1]
        return latest

def calculate_class_weights(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    
    weights = {}
    for label, count in zip(unique_labels, counts):
        weights[label] = total_samples / (len(unique_labels) * count)
    
    return weights

def get_model_size(model):
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    model_size = (param_size + buffer_size) / 1024 / 1024
    return model_size
