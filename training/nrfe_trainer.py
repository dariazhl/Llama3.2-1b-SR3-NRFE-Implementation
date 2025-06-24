
import logging
import torch
from tqdm import tqdm

from .trainer import BaseTrainer, EarlyStopping, MetricsTracker
from models.nrfe_model import EnhancedNRFEModel
from utils.logging_utils import get_logger

logger = get_logger(__name__)

class NRFETrainer(BaseTrainer):
    
    def __init__(self, config):
        super().__init__(config)
        self.metrics_tracker = MetricsTracker()
        
    def _forward_pass(self, model, batch):
        news_text = batch['text'][0] if isinstance(batch['text'], list) else batch['text']
        labels = batch['labels']
        
        positive_reasoning = self._generate_positive_reasoning(news_text)
        negative_reasoning = self._generate_negative_reasoning(news_text)
        
        outputs = model(
            news_text=news_text,
            positive_reasoning=positive_reasoning,
            negative_reasoning=negative_reasoning,
            labels=labels
        )
        
        return outputs
    
    def train(self, data_loader):
        logger.info("Starting Enhanced NRFE training...")
        
        model = EnhancedNRFEModel(self.config).to(self.device)
        
        data_loader.load_dataset()
        
        train_dataloader = data_loader.get_dataloader("train")
        val_dataloader = data_loader.get_dataloader("validation", shuffle=False)
        
        optimizer, scheduler = self.setup_optimizer_and_scheduler(model, train_dataloader)
        early_stopping = EarlyStopping(patience=3)
        
        best_f1 = 0
        
        for epoch in range(self.config.NUM_EPOCHS):
            train_loss = self.train_epoch(model, train_dataloader, optimizer, scheduler, epoch)
            
            val_metrics = self.evaluate(model, val_dataloader, epoch)
            
            self.metrics_tracker.update({
                'train_loss': train_loss,
                **val_metrics
            })
            
            if val_metrics['eval_f1'] > best_f1:
                best_f1 = val_metrics['eval_f1']
                model_path = self.config.MODELS_DIR / f"nrfe_best_model_epoch_{epoch}.pt"
                self.save_model(model, model_path, val_metrics)
                logger.info(f"New best model saved with F1: {best_f1:.4f}")
            
            if early_stopping(val_metrics['eval_loss'], model):
                logger.info("Early stopping triggered")
                break
            
            logger.info(f"Epoch {epoch+1} completed - Loss: {train_loss:.4f}, F1: {val_metrics['eval_f1']:.4f}")
        
        logger.info("Final evaluation...")
        test_dataloader = data_loader.get_dataloader("test", shuffle=False)
        final_metrics = self.evaluate(model, test_dataloader)
        
        final_model_path = self.config.MODELS_DIR / "nrfe_final_model.pt"
        self.save_model(model, final_model_path, final_metrics)
        
        logger.info("Enhanced NRFE training completed!")
        return model, final_metrics
    
    def train_epoch(self, model, train_dataloader, optimizer, scheduler, epoch):
        model.train()
        total_loss = 0
        num_batches = len(train_dataloader)
        
        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            try:
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
                
                postfix = {
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                }
                
                if 'consistency_scores' in outputs:
                    consistency = outputs['consistency_scores']
                    postfix['truth'] = f"{consistency.get('truth_score', 0):.3f}"
                    postfix['reasoning'] = f"{consistency.get('reasoning_score', 0):.3f}"
                    postfix['alignment'] = f"{consistency.get('alignment_score', 0):.3f}"
                
                progress_bar.set_postfix(postfix)
                
                if self.config.WANDB_API_KEY and step % self.config.LOGGING_STEPS == 0:
                    try:
                        import wandb
                        log_dict = {
                            'train_loss': loss.item(),
                            'learning_rate': scheduler.get_last_lr()[0],
                            'epoch': epoch,
                            'step': step
                        }
                        
                        if 'consistency_scores' in outputs:
                            consistency = outputs['consistency_scores']
                            log_dict.update({
                                'train_truth_score': consistency.get('truth_score', 0),
                                'train_reasoning_score': consistency.get('reasoning_score', 0),
                                'train_alignment_score': consistency.get('alignment_score', 0)
                            })
                        
                        wandb.log(log_dict)
                    except ImportError:
                        pass
                
            except Exception as e:
                logger.warning(f"Error in training step {step}: {e}")
                continue
        
        return total_loss / num_batches
    
    def evaluate(self, model, eval_dataloader, epoch=None):
        model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        consistency_metrics = {
            'truth_scores': [],
            'reasoning_scores': [],
            'alignment_scores': []
        }
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                try:
                    outputs = self._forward_pass(model, batch)
                    loss = outputs['loss']
                    logits = outputs['logits']
                    
                    predictions = torch.argmax(logits, dim=-1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(batch['labels'].cpu().numpy())
                    
                    total_loss += loss.item()
                    
                    if 'consistency_scores' in outputs:
                        consistency = outputs['consistency_scores']
                        consistency_metrics['truth_scores'].append(consistency.get('truth_score', 0))
                        consistency_metrics['reasoning_scores'].append(consistency.get('reasoning_score', 0))
                        consistency_metrics['alignment_scores'].append(consistency.get('alignment_score', 0))
                
                except Exception as e:
                    logger.warning(f"Error in evaluation batch: {e}")
                    continue
        
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        avg_loss = total_loss / len(eval_dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        avg_truth_score = sum(consistency_metrics['truth_scores']) / max(1, len(consistency_metrics['truth_scores']))
        avg_reasoning_score = sum(consistency_metrics['reasoning_scores']) / max(1, len(consistency_metrics['reasoning_scores']))
        avg_alignment_score = sum(consistency_metrics['alignment_scores']) / max(1, len(consistency_metrics['alignment_scores']))
        
        metrics = {
            'eval_loss': avg_loss,
            'eval_accuracy': accuracy,
            'eval_precision': precision,
            'eval_recall': recall,
            'eval_f1': f1,
            'eval_truth_consistency': avg_truth_score,
            'eval_reasoning_consistency': avg_reasoning_score,
            'eval_alignment_consistency': avg_alignment_score
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        if self.config.WANDB_API_KEY and epoch is not None:
            try:
                import wandb
                wandb.log({**metrics, 'epoch': epoch})
            except ImportError:
                pass
        
        return metrics
    
    def _generate_positive_reasoning(self, news_text: str) -> str:
        return f"This statement appears credible based on: {news_text[:100]}..."
    
    def _generate_negative_reasoning(self, news_text: str) -> str:
        return f"This statement may be questionable because: {news_text[:100]}..."
    
    def evaluate_cross_attention(self, model, eval_dataloader):
        model.eval()
        attention_stats = {
            'fp_to_x_avg': [],
            'fx_to_p_avg': [],
            'fn_to_x_avg': [],
            'fx_to_n_avg': []
        }
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating cross-attention"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                try:
                    outputs = self._forward_pass(model, batch)
                    
                    if 'cross_attention_weights' in outputs:
                        attention_weights = outputs['cross_attention_weights']
                        
                        for key in attention_stats.keys():
                            attention_key = key.replace('_avg', '')
                            if attention_key in attention_weights:
                                avg_weight = attention_weights[attention_key].mean().item()
                                attention_stats[key].append(avg_weight)
                
                except Exception as e:
                    logger.warning(f"Error in cross-attention evaluation: {e}")
                    continue
        
        final_stats = {}
        for key, values in attention_stats.items():
            if values:
                final_stats[key] = sum(values) / len(values)
            else:
                final_stats[key] = 0.0
        
        logger.info(f"Cross-attention statistics: {final_stats}")
        return final_stats
