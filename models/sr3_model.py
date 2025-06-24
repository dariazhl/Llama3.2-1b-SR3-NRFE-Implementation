
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
import numpy as np

from .base_model import BaseModel, FakeNewsClassifier, RationaleGenerator
from data.preprocessor import ConsistencyChecker

logger = logging.getLogger(__name__)

class SR3Model(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.rationale_generator = RationaleGenerator(config)
        self.classifier = FakeNewsClassifier(config)
        self.consistency_checker = ConsistencyChecker(config)
        
        self.lambda_consistency = config.SR3_LAMBDA_CONSISTENCY
        self.num_rationales = config.SR3_NUM_RATIONALES
        
    def forward(self, input_ids, attention_mask=None, labels=None, 
                training_mode="joint"):
        
        if training_mode == "joint":
            return self._joint_forward(input_ids, attention_mask, labels)
        elif training_mode == "rationale_only":
            return self._rationale_forward(input_ids, attention_mask, labels)
        elif training_mode == "classifier_only":
            return self._classifier_forward(input_ids, attention_mask, labels)
        else:
            raise ValueError(f"Unknown training mode: {training_mode}")
    
    def _joint_forward(self, input_ids, attention_mask, labels):
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        rationales = []
        rationale_logits = []
        
        for i in range(batch_size):
            input_text = self.rationale_generator.tokenizer.decode(
                input_ids[i], skip_special_tokens=True
            )
            
            sample_rationales = []
            for _ in range(self.num_rationales):
                rationale = self.rationale_generator.generate_rationale(
                    input_text, 
                    temperature=self.config.SR3_TEMPERATURE
                )
                sample_rationales.append(rationale)
            
            rationales.append(sample_rationales)
        
        classifier_outputs = self.classifier(input_ids, attention_mask, labels)
        classification_loss = classifier_outputs['loss']
        logits = classifier_outputs['logits']
        
        consistency_loss = self._calculate_consistency_loss(
            rationales, logits, labels
        )
        
        total_loss = classification_loss + self.lambda_consistency * consistency_loss
        
        return {
            'loss': total_loss,
            'classification_loss': classification_loss,
            'consistency_loss': consistency_loss,
            'logits': logits,
            'rationales': rationales
        }
    
    def _rationale_forward(self, input_ids, attention_mask, labels):
        return self.rationale_generator(input_ids, attention_mask, labels)
    
    def _classifier_forward(self, input_ids, attention_mask, labels):
        return self.classifier(input_ids, attention_mask, labels)
    
    def _calculate_consistency_loss(self, rationales, logits, labels):
        batch_size = len(rationales)
        consistency_losses = []
        
        predictions = torch.argmax(logits, dim=-1)
        
        for i in range(batch_size):
            sample_rationales = rationales[i]
            prediction = predictions[i].item()
            
            consistencies = []
            for rationale in sample_rationales:
                consistency = self.consistency_checker.check_semantic_consistency(
                    rationale, prediction
                )
                consistencies.append(consistency)
            
            consistency_tensor = torch.tensor(consistencies, dtype=torch.float32)
            
            consistency_loss = -torch.mean(consistency_tensor)
            consistency_losses.append(consistency_loss)
        
        if consistency_losses:
            return torch.stack(consistency_losses).mean()
        else:
            return torch.tensor(0.0, requires_grad=True)
    
    def generate_and_classify(self, text: str) -> Dict:
        rationales = []
        for _ in range(self.num_rationales):
            rationale = self.rationale_generator.generate_rationale(
                text, temperature=self.config.SR3_TEMPERATURE
            )
            rationales.append(rationale)
        
        inputs = self.rationale_generator.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.MAX_LENGTH
        )
        
        with torch.no_grad():
            outputs = self.classifier(
                inputs['input_ids'],
                inputs['attention_mask']
            )
            logits = outputs['logits']
            prediction = torch.argmax(logits, dim=-1).item()
            confidence = torch.softmax(logits, dim=-1).max().item()
        
        consistency_scores = []
        for rationale in rationales:
            consistency = self.consistency_checker.check_semantic_consistency(
                rationale, prediction
            )
            consistency_scores.append(consistency)
        
        return {
            'text': text,
            'prediction': prediction,
            'confidence': confidence,
            'rationales': rationales,
            'consistency_scores': consistency_scores,
            'average_consistency': np.mean(consistency_scores)
        }
    
    def self_rectify(self, text: str, max_iterations=3) -> Dict:
        current_text = text
        iteration_results = []
        
        for iteration in range(max_iterations):
            result = self.generate_and_classify(current_text)
            iteration_results.append(result)
            
            avg_consistency = result['average_consistency']
            
            if avg_consistency > 0.7:
                break
            
            best_rationale_idx = np.argmax(result['consistency_scores'])
            best_rationale = result['rationales'][best_rationale_idx]
            
            current_text = f"{text} Rationale: {best_rationale}"
        
        return {
            'original_text': text,
            'final_prediction': iteration_results[-1]['prediction'],
            'final_confidence': iteration_results[-1]['confidence'],
            'iterations': iteration_results,
            'num_iterations': len(iteration_results)
        }

class SR3Loss(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lambda_consistency = config.SR3_LAMBDA_CONSISTENCY
        self.cross_entropy = nn.CrossEntropyLoss()
        
    def forward(self, outputs, labels):
        classification_loss = outputs['classification_loss']
        consistency_loss = outputs['consistency_loss']
        
        total_loss = classification_loss + self.lambda_consistency * consistency_loss
        
        return {
            'loss': total_loss,
            'classification_loss': classification_loss,
            'consistency_loss': consistency_loss
        }

class SR3Evaluator:
    
    def __init__(self, config):
        self.config = config
        
    def evaluate_consistency(self, model, dataloader):
        model.eval()
        total_consistency = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                
                outputs = model(input_ids, attention_mask, labels, 
                              training_mode="joint")
                
                rationales = outputs['rationales']
                logits = outputs['logits']
                predictions = torch.argmax(logits, dim=-1)
                
                batch_consistency = 0
                for i, sample_rationales in enumerate(rationales):
                    prediction = predictions[i].item()
                    
                    consistencies = []
                    for rationale in sample_rationales:
                        consistency = model.consistency_checker.check_semantic_consistency(
                            rationale, prediction
                        )
                        consistencies.append(consistency)
                    
                    batch_consistency += np.mean(consistencies)
                
                total_consistency += batch_consistency
                num_samples += len(rationales)
        
        average_consistency = total_consistency / num_samples if num_samples > 0 else 0
        return average_consistency
    
    def evaluate_rectification(self, model, test_texts):
        rectification_results = []
        
        for text in test_texts:
            result = model.self_rectify(text)
            
            initial_consistency = result['iterations'][0]['average_consistency']
            final_consistency = result['iterations'][-1]['average_consistency']
            improvement = final_consistency - initial_consistency
            
            rectification_results.append({
                'text': text,
                'initial_consistency': initial_consistency,
                'final_consistency': final_consistency,
                'improvement': improvement,
                'num_iterations': result['num_iterations']
            })
        
        improvements = [r['improvement'] for r in rectification_results]
        avg_improvement = np.mean(improvements)
        avg_iterations = np.mean([r['num_iterations'] for r in rectification_results])
        
        return {
            'results': rectification_results,
            'average_improvement': avg_improvement,
            'average_iterations': avg_iterations,
            'improvement_rate': sum(1 for imp in improvements if imp > 0) / len(improvements)
        }
