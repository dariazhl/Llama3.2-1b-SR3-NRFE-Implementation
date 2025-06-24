
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np

from .base_model import BaseModel, DualBERTEncoder, CrossAttentionLayer, MLPClassifier
from utils.logging_utils import get_logger

logger = get_logger(__name__)

class EnhancedNRFEModel(BaseModel):
    
    def __init__(self, config):
        super().__init__(config)
        
        self.dual_encoder = DualBERTEncoder(config)
        
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(config) 
            for _ in range(config.CROSS_ATTENTION_LAYERS)
        ])
        
        self.consistency_projector = nn.Linear(config.HIDDEN_SIZE, config.REASONING_HIDDEN_SIZE)
        
        self.classifier = MLPClassifier(
            input_dim=config.HIDDEN_SIZE * 3,
            hidden_dim=config.REASONING_HIDDEN_SIZE,
            output_dim=config.NUM_CLASSES,
            dropout_rate=config.DROPOUT_RATE
        )
        
        self.classification_loss = nn.CrossEntropyLoss()
        self.consistency_loss = nn.MSELoss()
        
    def forward(self, news_text: str, positive_reasoning: str, negative_reasoning: str,
                labels: Optional[torch.Tensor] = None) -> Dict:
        
        news_embeddings = self.dual_encoder.encode_news(news_text)
        pos_embeddings = self.dual_encoder.encode_reasoning(positive_reasoning)
        neg_embeddings = self.dual_encoder.encode_reasoning(negative_reasoning)
        
        cross_attention_weights = {}
        
        attended_pos, fp_to_x = self.cross_attention_layers[0](
            query_embeddings=pos_embeddings,
            key_embeddings=news_embeddings,
            value_embeddings=news_embeddings
        )
        
        attended_news_pos, fx_to_p = self.cross_attention_layers[0](
            query_embeddings=news_embeddings,
            key_embeddings=pos_embeddings,
            value_embeddings=pos_embeddings
        )
        
        attended_neg, fn_to_x = self.cross_attention_layers[0](
            query_embeddings=neg_embeddings,
            key_embeddings=news_embeddings,
            value_embeddings=news_embeddings
        )
        
        attended_news_neg, fx_to_n = self.cross_attention_layers[0](
            query_embeddings=news_embeddings,
            key_embeddings=neg_embeddings,
            value_embeddings=neg_embeddings
        )
        
        cross_attention_weights = {
            'fp_to_x': fp_to_x,
            'fx_to_p': fx_to_p,
            'fn_to_x': fn_to_x,
            'fx_to_n': fx_to_n
        }
        
        news_pooled = torch.mean(attended_news_pos, dim=1)
        pos_pooled = torch.mean(attended_pos, dim=1)
        neg_pooled = torch.mean(attended_neg, dim=1)
        
        combined_features = torch.cat([news_pooled, pos_pooled, neg_pooled], dim=-1)
        
        logits = self.classifier(combined_features)
        
        total_loss = None
        classification_loss = None
        consistency_loss_val = None
        consistency_scores = {}
        
        if labels is not None:
            classification_loss = self.classification_loss(logits, labels)
            
            consistency_scores = self._calculate_consistency_scores(
                news_pooled, pos_pooled, neg_pooled, labels
            )
            
            consistency_loss_val = self._calculate_consistency_loss(consistency_scores)
            
            total_loss = classification_loss + 0.1 * consistency_loss_val
        
        return {
            'loss': total_loss,
            'classification_loss': classification_loss,
            'consistency_loss': consistency_loss_val,
            'logits': logits,
            'cross_attention_weights': cross_attention_weights,
            'consistency_scores': consistency_scores,
            'embeddings': {
                'news': news_pooled,
                'positive_reasoning': pos_pooled,
                'negative_reasoning': neg_pooled
            }
        }
    
    def _calculate_consistency_scores(self, news_emb: torch.Tensor, 
                                    pos_emb: torch.Tensor, neg_emb: torch.Tensor,
                                    labels: torch.Tensor) -> Dict:
        
        news_proj = self.consistency_projector(news_emb)
        pos_proj = self.consistency_projector(pos_emb)
        neg_proj = self.consistency_projector(neg_emb)
        
        news_pos_sim = F.cosine_similarity(news_proj, pos_proj, dim=-1)
        news_neg_sim = F.cosine_similarity(news_proj, neg_proj, dim=-1)
        pos_neg_sim = F.cosine_similarity(pos_proj, neg_proj, dim=-1)
        
        truth_score = torch.where(
            labels == 0,
            news_pos_sim,
            news_neg_sim
        ).mean()
        
        reasoning_score = (1 - torch.abs(pos_neg_sim)).mean()
        
        alignment_score = (news_pos_sim - news_neg_sim).abs().mean()
        
        return {
            'truth_score': truth_score.item(),
            'reasoning_score': reasoning_score.item(),
            'alignment_score': alignment_score.item()
        }
    
    def _calculate_consistency_loss(self, consistency_scores: Dict) -> torch.Tensor:
        
        target_truth = torch.tensor(1.0, device=next(self.parameters()).device)
        target_reasoning = torch.tensor(1.0, device=next(self.parameters()).device)
        target_alignment = torch.tensor(1.0, device=next(self.parameters()).device)
        
        truth_loss = self.consistency_loss(
            torch.tensor(consistency_scores['truth_score']), target_truth
        )
        reasoning_loss = self.consistency_loss(
            torch.tensor(consistency_scores['reasoning_score']), target_reasoning
        )
        alignment_loss = self.consistency_loss(
            torch.tensor(consistency_scores['alignment_score']), target_alignment
        )
        
        return truth_loss + reasoning_loss + alignment_loss

class NRFEModel(BaseModel):
    
    def __init__(self, config):
        super().__init__(config)
        
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(config.MODEL_NAME)
        
        self.classifier = MLPClassifier(
            input_dim=config.HIDDEN_SIZE,
            hidden_dim=config.HIDDEN_SIZE // 2,
            output_dim=config.NUM_CLASSES,
            dropout_rate=config.DROPOUT_RATE
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'last_hidden_state': outputs.last_hidden_state
        }

class TeacherNRFEModel(EnhancedNRFEModel):
    
    def __init__(self, config):
        super().__init__(config)
        
        self.knowledge_projector = nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE)
        self.temperature = config.get('DISTILLATION_TEMPERATURE', 4.0)
    
    def forward_teacher(self, *args, **kwargs):
        
        outputs = self.forward(*args, **kwargs)
        
        soft_predictions = F.softmax(outputs['logits'] / self.temperature, dim=-1)
        
        knowledge_repr = self.knowledge_projector(outputs['embeddings']['news'])
        
        outputs['soft_predictions'] = soft_predictions
        outputs['knowledge_representation'] = knowledge_repr
        
        return outputs

class StudentNRFEModel(BaseModel):
    
    def __init__(self, config):
        super().__init__(config)
        
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained('distilbert-base-uncased')
        
        self.classifier = MLPClassifier(
            input_dim=768,
            hidden_dim=256,
            output_dim=config.NUM_CLASSES,
            dropout_rate=config.DROPOUT_RATE
        )
        
        self.knowledge_adapter = nn.Linear(768, config.HIDDEN_SIZE)
        self.temperature = config.get('DISTILLATION_TEMPERATURE', 4.0)
        
        self.hard_loss = nn.CrossEntropyLoss()
        self.soft_loss = nn.KLDivLoss(reduction='batchmean')
        self.feature_loss = nn.MSELoss()
    
    def forward(self, input_ids, attention_mask=None, labels=None, 
                teacher_outputs=None):
        
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        
        adapted_knowledge = self.knowledge_adapter(pooled_output)
        
        total_loss = None
        
        if labels is not None:
            hard_loss = self.hard_loss(logits, labels)
            
            total_loss = hard_loss
            
            if teacher_outputs is not None and 'soft_predictions' in teacher_outputs:
                student_soft = F.log_softmax(logits / self.temperature, dim=-1)
                teacher_soft = teacher_outputs['soft_predictions']
                
                soft_loss = self.soft_loss(student_soft, teacher_soft) * (self.temperature ** 2)
                
                if 'knowledge_representation' in teacher_outputs:
                    feature_loss = self.feature_loss(
                        adapted_knowledge, 
                        teacher_outputs['knowledge_representation']
                    )
                    
                    total_loss = hard_loss + 0.5 * soft_loss + 0.1 * feature_loss
                else:
                    total_loss = hard_loss + 0.5 * soft_loss
        
        return {
            'loss': total_loss,
            'logits': logits,
            'knowledge_representation': adapted_knowledge,
            'last_hidden_state': outputs.last_hidden_state
        }
