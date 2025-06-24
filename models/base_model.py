
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from fallback_transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional, Tuple
import numpy as np

from utils.logging_utils import get_logger

logger = get_logger(__name__)

class BaseModel(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward method")

class FakeNewsClassifier(BaseModel):
    
    def __init__(self, config):
        super().__init__(config)
        
        self.bert = AutoModel.from_pretrained(config.MODEL_NAME)
        
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        self.classifier = nn.Linear(config.HIDDEN_SIZE, config.NUM_CLASSES)
        
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'last_hidden_state': outputs.last_hidden_state
        }

class DualBERTEncoder(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.news_encoder = AutoModel.from_pretrained(config.MODEL_NAME)
        self.reasoning_encoder = AutoModel.from_pretrained(config.MODEL_NAME)
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def encode_news(self, news_text: str) -> torch.Tensor:
        
        inputs = self.tokenizer(
            news_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.config.MAX_LENGTH
        )
        
        with torch.no_grad():
            outputs = self.news_encoder(**inputs)
        
        return outputs.last_hidden_state
    
    def encode_reasoning(self, reasoning_text: str) -> torch.Tensor:
        
        inputs = self.tokenizer(
            reasoning_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.config.MAX_LENGTH
        )
        
        with torch.no_grad():
            outputs = self.reasoning_encoder(**inputs)
        
        return outputs.last_hidden_state
    
    def forward(self, news_text: str, reasoning_text: str) -> Dict[str, torch.Tensor]:
        
        news_embeddings = self.encode_news(news_text)
        reasoning_embeddings = self.encode_reasoning(reasoning_text)
        
        return {
            'news_embeddings': news_embeddings,
            'reasoning_embeddings': reasoning_embeddings
        }

class CrossAttentionLayer(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.HIDDEN_SIZE
        self.num_heads = config.CROSS_ATTENTION_HEADS
        self.head_dim = self.hidden_size // self.num_heads
        
        self.query_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.key_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.value_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        
    def forward(self, query_embeddings: torch.Tensor, 
                key_embeddings: torch.Tensor,
                value_embeddings: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size, seq_len_q, _ = query_embeddings.size()
        seq_len_k = key_embeddings.size(1)
        
        Q = self.query_proj(query_embeddings)
        K = self.key_proj(key_embeddings)
        V = self.value_proj(value_embeddings)
        
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended_values = torch.matmul(attention_weights, V)
        
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.hidden_size
        )
        
        output = self.output_proj(attended_values)
        
        return output, attention_weights.mean(dim=1)

class MLPClassifier(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class RationaleGenerator(BaseModel):
    
    def __init__(self, config):
        super().__init__(config)
        
        self.model = AutoModel.from_pretrained(config.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.rationale_head = nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE)
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        
    def generate_rationale(self, text: str, target_label: Optional[int] = None,
                          temperature: float = 1.0) -> str:
        
        try:
            if target_label == 1:
                rationales = [
                    f"This text shows signs of being unreliable due to sensational language and lack of credible sources.",
                    f"The claims made in this text appear to be unsubstantiated and potentially misleading.",
                    f"This content exhibits characteristics typical of fake news including emotional appeals and vague assertions."
                ]
            else:
                rationales = [
                    f"This text appears to be from a credible source with factual information that can be verified.",
                    f"The content follows journalistic standards with balanced reporting and proper attribution.",
                    f"This article demonstrates characteristics of reliable news with concrete facts and proper context."
                ]
            
            if temperature > 0:
                idx = np.random.choice(len(rationales))
            else:
                idx = 0
            
            return rationales[idx]
            
        except Exception as e:
            logger.warning(f"Error generating rationale: {e}")
            return "Unable to generate rationale for this text."
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        rationale_repr = self.rationale_head(pooled_output)
        
        loss = None
        if labels is not None:
            loss = torch.tensor(0.0, requires_grad=True)
        
        return {
            'loss': loss,
            'rationale_representation': rationale_repr,
            'last_hidden_state': outputs.last_hidden_state
        }
