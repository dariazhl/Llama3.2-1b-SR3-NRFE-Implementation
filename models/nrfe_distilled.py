
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np

from .base_model import BaseModel
from utils.logging_utils import get_logger

logger = get_logger(__name__)

class NewsEncoder(nn.Module):
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, 
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        
        final_hidden = torch.cat([forward_hidden, backward_hidden], dim=-1)
        
        return final_hidden

class AttentionPooling(nn.Module):
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        attention_weights = self.attention(hidden_states)
        attention_weights = attention_weights.squeeze(-1)
        
        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(
                attention_mask == 0, -1e9
            )
        
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        pooled = torch.sum(
            hidden_states * attention_weights.unsqueeze(-1), dim=1
        )
        
        return pooled

class NRFE_D(BaseModel):
    
    def __init__(self, config):
        super().__init__(config)
        
        self.vocab_size = config.get('VOCAB_SIZE', 30000)
        self.embed_dim = config.get('EMBED_DIM', 128)
        self.hidden_dim = config.get('DISTILLED_HIDDEN_DIM', 256)
        self.num_layers = config.get('DISTILLED_NUM_LAYERS', 2)
        
        self.news_encoder = NewsEncoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=config.DROPOUT_RATE
        )
        
        self.attention_pooling = AttentionPooling(self.hidden_dim * 2)
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE)
        )
        
        self.classifier = nn.Linear(self.hidden_dim // 2, config.NUM_CLASSES)
        
        self.loss_fn = nn.CrossEntropyLoss()
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0, 0.1)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict:
        
        encoded = self.news_encoder(input_ids, attention_mask)
        
        batch_size, hidden_size = encoded.shape
        seq_len = input_ids.shape[1]
        
        encoded_sequence = encoded.unsqueeze(1).expand(batch_size, seq_len, hidden_size)
        
        pooled = self.attention_pooling(encoded_sequence, attention_mask)
        
        features = self.feature_extractor(pooled)
        
        logits = self.classifier(features)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'features': features,
            'pooled_output': pooled
        }
    
    def predict(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Dict:
        
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            
            logits = outputs['logits']
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            confidence = torch.max(probabilities, dim=-1)[0]
            
            return {
                'predictions': predictions,
                'probabilities': probabilities,
                'confidence': confidence,
                'features': outputs['features']
            }

class NRFE_D_Trainer:
    
    def __init__(self, config, teacher_model=None):
        self.config = config
        self.teacher_model = teacher_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.temperature = config.get('DISTILLATION_TEMPERATURE', 4.0)
        self.alpha = config.get('DISTILLATION_ALPHA', 0.7)
        
        self.hard_loss = nn.CrossEntropyLoss()
        self.soft_loss = nn.KLDivLoss(reduction='batchmean')
        
    def distillation_loss(self, student_logits: torch.Tensor, 
                         teacher_logits: torch.Tensor,
                         labels: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        
        hard_loss = self.hard_loss(student_logits, labels)
        
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        soft_loss = self.soft_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        loss_dict = {
            'total_loss': total_loss,
            'hard_loss': hard_loss,
            'soft_loss': soft_loss
        }
        
        return total_loss, loss_dict
    
    def train_step(self, student_model: NRFE_D, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        student_outputs = student_model(input_ids, attention_mask, labels)
        student_logits = student_outputs['logits']
        
        if self.teacher_model is not None:
            self.teacher_model.eval()
            with torch.no_grad():
                teacher_outputs = self.teacher_model(input_ids, attention_mask)
                teacher_logits = teacher_outputs['logits']
            
            loss, loss_dict = self.distillation_loss(
                student_logits, teacher_logits, labels
            )
        else:
            loss = student_outputs['loss']
            loss_dict = {'total_loss': loss, 'hard_loss': loss, 'soft_loss': torch.tensor(0.0)}
        
        return loss, loss_dict
    
    def evaluate_efficiency(self, model: NRFE_D) -> Dict:
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        model_size_mb = (param_size + buffer_size) / 1024 / 1024
        
        sample_input = torch.randint(0, model.vocab_size, (1, 128))
        sample_mask = torch.ones(1, 128)
        
        embed_flops = 128 * model.embed_dim
        lstm_flops = 128 * model.hidden_dim * model.num_layers * 8
        classifier_flops = model.hidden_dim * model.config.NUM_CLASSES
        
        total_flops = embed_flops + lstm_flops + classifier_flops
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'estimated_flops': total_flops,
            'compression_ratio': getattr(self, 'teacher_params', total_params) / total_params
        }
    
    def compare_with_teacher(self, student_model: NRFE_D, 
                           test_dataloader) -> Dict:
        
        if self.teacher_model is None:
            return {"error": "No teacher model provided"}
        
        student_model.eval()
        self.teacher_model.eval()
        
        student_correct = 0
        teacher_correct = 0
        total = 0
        agreement = 0
        
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                
                student_outputs = student_model.predict(input_ids, attention_mask)
                student_preds = student_outputs['predictions']
                
                teacher_outputs = self.teacher_model(input_ids, attention_mask)
                teacher_preds = torch.argmax(teacher_outputs['logits'], dim=-1)
                
                student_correct += (student_preds == labels).sum().item()
                teacher_correct += (teacher_preds == labels).sum().item()
                agreement += (student_preds == teacher_preds).sum().item()
                total += labels.size(0)
        
        student_accuracy = student_correct / total
        teacher_accuracy = teacher_correct / total
        agreement_rate = agreement / total
        
        return {
            'student_accuracy': student_accuracy,
            'teacher_accuracy': teacher_accuracy,
            'agreement_rate': agreement_rate,
            'accuracy_drop': teacher_accuracy - student_accuracy,
            'total_samples': total
        }
