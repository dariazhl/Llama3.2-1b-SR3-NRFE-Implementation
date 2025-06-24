
import torch
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, roc_curve
)
from pathlib import Path
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    import seaborn as sns
except ImportError:
    sns = None

from models.nrfe_model import EnhancedNRFEModel, NRFEModel
from models.sr3_model import SR3Model
from models.base_model import FakeNewsClassifier
from utils.logging_utils import get_logger

logger = get_logger(__name__)

class ModelEvaluator:
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def evaluate_model(self, model, dataloader, model_name: str = "Model") -> Dict:
        
        logger.info(f"Evaluating {model_name}...")
        
        model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                try:
                    if isinstance(model, EnhancedNRFEModel):
                        news_text = batch['text'][0] if isinstance(batch['text'], list) else batch['text']
                        positive_reasoning = f"This appears credible: {news_text[:50]}..."
                        negative_reasoning = f"This may be questionable: {news_text[:50]}..."
                        
                        outputs = model(
                            news_text=news_text,  
                            positive_reasoning=positive_reasoning,
                            negative_reasoning=negative_reasoning,
                            labels=batch['labels']
                        )
                    else:
                        outputs = model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['labels']
                        )
                    
                    logits = outputs['logits']
                    loss = outputs.get('loss', torch.tensor(0.0))
                    
                    probabilities = torch.softmax(logits, dim=-1)
                    predictions = torch.argmax(logits, dim=-1)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
                    all_labels.extend(batch['labels'].cpu().numpy())
                    
                    if loss.requires_grad:
                        total_loss += loss.item()
                
                except Exception as e:
                    logger.warning(f"Error evaluating batch: {e}")
                    continue
        
        metrics = self._calculate_metrics(
            all_labels, all_predictions, all_probabilities, model_name
        )
        
        metrics['loss'] = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        
        logger.info(f"{model_name} evaluation completed")
        return metrics
    
    def _calculate_metrics(self, true_labels: List[int], predictions: List[int], 
                          probabilities: List[List[float]], model_name: str) -> Dict:
        
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            true_labels, predictions, average=None
        )
        
        cm = confusion_matrix(true_labels, predictions)
        
        roc_auc = None
        if len(np.unique(true_labels)) == 2 and len(probabilities) > 0:
            try:
                probs_class_1 = [prob[1] for prob in probabilities]
                roc_auc = roc_auc_score(true_labels, probs_class_1)
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
        
        class_report = classification_report(
            true_labels, predictions, 
            target_names=['Real', 'Fake'], 
            output_dict=True
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'support_per_class': support.tolist(),
            'confusion_matrix': cm.tolist(),
            'roc_auc': roc_auc,
            'classification_report': class_report,
            'num_samples': len(true_labels)
        }
        
        return metrics
    
    def evaluate_all_models(self, dataloader) -> Dict:
        
        results = {}
        models_dir = self.config.MODELS_DIR
        
        model_files = {
            'Enhanced NRFE': 'nrfe_final_model.pt',
            'SR3': 'sr3_final_model.pt',
            'Base BERT': 'base_bert_model.pt'
        }
        
        for model_name, filename in model_files.items():
            model_path = models_dir / filename
            
            if model_path.exists():
                try:
                    if model_name == 'Enhanced NRFE':
                        model = EnhancedNRFEModel(self.config)
                    elif model_name == 'SR3':
                        model = SR3Model(self.config)
                    else:
                        model = FakeNewsClassifier(self.config)
                    
                    checkpoint = torch.load(model_path, map_location=self.device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.to(self.device)
                    
                    metrics = self.evaluate_model(model, dataloader, model_name)
                    results[model_name] = metrics
                    
                except Exception as e:
                    logger.error(f"Error loading/evaluating {model_name}: {e}")
                    results[model_name] = {'error': str(e)}
            else:
                logger.warning(f"Model file not found: {model_path}")
                results[model_name] = {'error': 'Model file not found'}
        
        return results
    
    def compare_models(self, dataloader) -> Dict:
        
        results = self.evaluate_all_models(dataloader)
        
        comparison = {
            'models': {},
            'best_accuracy': {'model': None, 'score': 0},
            'best_f1': {'model': None, 'score': 0},
            'best_precision': {'model': None, 'score': 0},
            'best_recall': {'model': None, 'score': 0}
        }
        
        for model_name, metrics in results.items():
            if 'error' not in metrics:
                comparison['models'][model_name] = {
                    'accuracy': metrics['accuracy'],
                    'f1': metrics['f1'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'roc_auc': metrics.get('roc_auc'),
                    'num_samples': metrics['num_samples']
                }
                
                if metrics['accuracy'] > comparison['best_accuracy']['score']:
                    comparison['best_accuracy'] = {'model': model_name, 'score': metrics['accuracy']}
                
                if metrics['f1'] > comparison['best_f1']['score']:
                    comparison['best_f1'] = {'model': model_name, 'score': metrics['f1']}
                
                if metrics['precision'] > comparison['best_precision']['score']:
                    comparison['best_precision'] = {'model': model_name, 'score': metrics['precision']}
                
                if metrics['recall'] > comparison['best_recall']['score']:
                    comparison['best_recall'] = {'model': model_name, 'score': metrics['recall']}
        
        return comparison
    
    def evaluate_cross_attention(self, model, dataloader) -> Dict:
        
        if not isinstance(model, EnhancedNRFEModel):
            return {'error': 'Cross-attention evaluation only available for Enhanced NRFE model'}
        
        model.eval()
        attention_stats = {
            'fp_to_x_weights': [],
            'fx_to_p_weights': [],
            'fn_to_x_weights': [],
            'fx_to_n_weights': []
        }
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                try:
                    news_text = batch['text'][0] if isinstance(batch['text'], list) else batch['text']
                    positive_reasoning = f"This appears credible: {news_text[:50]}..."
                    negative_reasoning = f"This may be questionable: {news_text[:50]}..."
                    
                    outputs = model(
                        news_text=news_text,
                        positive_reasoning=positive_reasoning,
                        negative_reasoning=negative_reasoning,
                        labels=batch['labels']
                    )
                    
                    if 'cross_attention_weights' in outputs:
                        attention_weights = outputs['cross_attention_weights']
                        
                        for key in attention_stats.keys():
                            attention_key = key.replace('_weights', '')
                            if attention_key in attention_weights:
                                weights = attention_weights[attention_key]
                                attention_stats[key].append(weights.mean().item())
                
                except Exception as e:
                    logger.warning(f"Error in cross-attention evaluation: {e}")
                    continue
        
        final_stats = {}
        for key, values in attention_stats.items():
            if values:
                final_stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            else:
                final_stats[key] = {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        
        return final_stats
    
    def generate_evaluation_report(self, results: Dict, output_path: Optional[str] = None) -> str:
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FAKE NEWS DETECTION MODEL EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        for model_name, metrics in results.items():
            if 'error' in metrics:
                report_lines.append(f"{model_name}: ERROR - {metrics['error']}")
                report_lines.append("")
                continue
            
            report_lines.append(f"{model_name.upper()} RESULTS:")
            report_lines.append("-" * 40)
            report_lines.append(f"Accuracy:  {metrics['accuracy']:.4f}")
            report_lines.append(f"Precision: {metrics['precision']:.4f}")
            report_lines.append(f"Recall:    {metrics['recall']:.4f}")
            report_lines.append(f"F1-Score:  {metrics['f1']:.4f}")
            
            if metrics.get('roc_auc'):
                report_lines.append(f"ROC AUC:   {metrics['roc_auc']:.4f}")
            
            report_lines.append(f"Samples:   {metrics['num_samples']}")
            
            if 'f1_per_class' in metrics:
                report_lines.append("")
                report_lines.append("Per-Class Metrics:")
                for i, (f1, prec, rec) in enumerate(zip(
                    metrics['f1_per_class'], 
                    metrics['precision_per_class'], 
                    metrics['recall_per_class']
                )):
                    class_name = 'Real' if i == 0 else 'Fake'
                    report_lines.append(f"  {class_name}: F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}")
            
            if 'confusion_matrix' in metrics:
                report_lines.append("")
                report_lines.append("Confusion Matrix:")
                cm = np.array(metrics['confusion_matrix'])
                report_lines.append(f"           Predicted")
                report_lines.append(f"         Real  Fake")
                report_lines.append(f"Real   {cm[0,0]:6d} {cm[0,1]:6d}")
                report_lines.append(f"Fake   {cm[1,0]:6d} {cm[1,1]:6d}")
            
            report_lines.append("")
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Evaluation report saved to {output_path}")
        
        return report_text
    
    def evaluate_consistency_metrics(self, model, dataloader) -> Dict:
        
        if isinstance(model, EnhancedNRFEModel):
            return self._evaluate_nrfe_consistency(model, dataloader)
        elif isinstance(model, SR3Model):
            return self._evaluate_sr3_consistency(model, dataloader)
        else:
            return {'error': 'Consistency evaluation not available for this model type'}
    
    def _evaluate_nrfe_consistency(self, model, dataloader) -> Dict:
        
        model.eval()
        consistency_scores = {
            'truth_scores': [],
            'reasoning_scores': [],
            'alignment_scores': []
        }
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                try:
                    news_text = batch['text'][0] if isinstance(batch['text'], list) else batch['text']
                    positive_reasoning = f"This appears credible: {news_text[:50]}..."
                    negative_reasoning = f"This may be questionable: {news_text[:50]}..."
                    
                    outputs = model(
                        news_text=news_text,
                        positive_reasoning=positive_reasoning,
                        negative_reasoning=negative_reasoning,
                        labels=batch['labels']
                    )
                    
                    if 'consistency_scores' in outputs:
                        scores = outputs['consistency_scores']
                        consistency_scores['truth_scores'].append(scores.get('truth_score', 0))
                        consistency_scores['reasoning_scores'].append(scores.get('reasoning_score', 0))
                        consistency_scores['alignment_scores'].append(scores.get('alignment_score', 0))
                
                except Exception as e:
                    logger.warning(f"Error in NRFE consistency evaluation: {e}")
                    continue
        
        avg_scores = {}
        for key, values in consistency_scores.items():
            if values:
                avg_scores[key.replace('_scores', '_avg')] = np.mean(values)
                avg_scores[key.replace('_scores', '_std')] = np.std(values)
            else:
                avg_scores[key.replace('_scores', '_avg')] = 0
                avg_scores[key.replace('_scores', '_std')] = 0
        
        return avg_scores
    
    def _evaluate_sr3_consistency(self, model, dataloader) -> Dict:
        
        from models.sr3_model import SR3Evaluator
        
        evaluator = SR3Evaluator(self.config)
        consistency_score = evaluator.evaluate_consistency(model, dataloader)
        
        return {
            'average_consistency': consistency_score,
            'consistency_threshold': self.config.get('CONSISTENCY_THRESHOLD', 0.7)
        }
