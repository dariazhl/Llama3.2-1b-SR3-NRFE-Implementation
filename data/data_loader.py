
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
import torch
try:
    from transformers import AutoTokenizer
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from fallback_transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

from config import Config
from utils.logging_utils import get_logger
from .preprocessor import TextPreprocessor

logger = get_logger(__name__)

class FakeNewsDataset(Dataset):
    
    def __init__(self, texts: List[str], labels: List[int], 
                 tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'text': text
        }

class DataLoader:
    
    def __init__(self, config: Config):
        self.config = config
        self.preprocessor = TextPreprocessor()
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        self.dataset = None
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_dataset(self, dataset_name: str = "sample") -> Dict:
        
        logger.info(f"Loading dataset: {dataset_name}")
        
        if dataset_name == "sample":
            data = self.config.SAMPLE_NEWS_DATA
            texts = [item['text'] for item in data]
            labels = [item['label'] for item in data]
            
        else:
            logger.warning(f"Dataset {dataset_name} not implemented, using sample data")
            data = self.config.SAMPLE_NEWS_DATA
            texts = [item['text'] for item in data]
            labels = [item['label'] for item in data]
        
        processed_texts = []
        for text in texts:
            processed_text = self.preprocessor.preprocess_text(text)
            processed_texts.append(processed_text)
        
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            processed_texts, labels, 
            test_size=(1 - self.config.TRAIN_SIZE),
            random_state=self.config.RANDOM_SEED,
            stratify=labels
        )
        
        val_size = self.config.VAL_SIZE / (self.config.VAL_SIZE + self.config.TEST_SIZE)
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels,
            test_size=(1 - val_size),
            random_state=self.config.RANDOM_SEED,
            stratify=temp_labels
        )
        
        self.dataset = {
            'train': {'texts': train_texts, 'labels': train_labels},
            'validation': {'texts': val_texts, 'labels': val_labels},
            'test': {'texts': test_texts, 'labels': test_labels}
        }
        
        logger.info(f"Dataset loaded - Train: {len(train_texts)}, "
                   f"Val: {len(val_texts)}, Test: {len(test_texts)}")
        
        return self.dataset
    
    def get_dataloader(self, split: str, shuffle: bool = True, 
                      batch_size: Optional[int] = None) -> torch.utils.data.DataLoader:
        
        if self.dataset is None:
            self.load_dataset()
        
        if split not in self.dataset:
            raise ValueError(f"Split {split} not found in dataset")
        
        dataset = FakeNewsDataset(
            texts=self.dataset[split]['texts'],
            labels=self.dataset[split]['labels'],
            tokenizer=self.tokenizer,
            max_length=self.config.MAX_LENGTH
        )
        
        batch_size = batch_size or self.config.BATCH_SIZE
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False
        )
        
        return dataloader
    
    def prepare_rationale_data(self, rationale_generator, split: str = "train", 
                             num_samples: Optional[int] = None) -> List[Dict]:
        
        if self.dataset is None:
            self.load_dataset()
        
        texts = self.dataset[split]['texts']
        labels = self.dataset[split]['labels']
        
        if num_samples:
            texts = texts[:num_samples]
            labels = labels[:num_samples]
        
        rationale_data = []
        
        logger.info(f"Generating rationales for {len(texts)} samples...")
        
        for i, (text, label) in enumerate(zip(texts, labels)):
            try:
                if hasattr(rationale_generator, 'generate_rationale'):
                    correct_rationale = rationale_generator.generate_rationale(
                        text, target_label=label
                    )
                    incorrect_rationale = rationale_generator.generate_rationale(
                        text, target_label=1-label
                    )
                else:
                    if label == 0:
                        correct_rationale = f"This appears to be factual news: {text[:50]}..."
                        incorrect_rationale = f"This could be questionable: {text[:50]}..."
                    else:
                        correct_rationale = f"This appears questionable: {text[:50]}..."
                        incorrect_rationale = f"This appears to be factual: {text[:50]}..."
                
                rationale_data.append({
                    'text': text,
                    'label': label,
                    'correct_rationale': correct_rationale,
                    'incorrect_rationale': incorrect_rationale
                })
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Generated rationales for {i+1}/{len(texts)} samples")
                    
            except Exception as e:
                logger.warning(f"Error generating rationale for sample {i}: {e}")
                continue
        
        logger.info(f"Generated {len(rationale_data)} rationale samples")
        return rationale_data
    
    def get_class_weights(self, split: str = "train") -> Dict[int, float]:
        
        if self.dataset is None:
            self.load_dataset()
        
        labels = self.dataset[split]['labels']
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        total_samples = len(labels)
        weights = {}
        
        for label, count in zip(unique_labels, counts):
            weights[int(label)] = total_samples / (len(unique_labels) * count)
        
        logger.info(f"Class weights for {split}: {weights}")
        return weights
    
    def get_dataset_stats(self) -> Dict:
        
        if self.dataset is None:
            self.load_dataset()
        
        stats = {}
        
        for split in ['train', 'validation', 'test']:
            texts = self.dataset[split]['texts']
            labels = self.dataset[split]['labels']
            
            text_lengths = [len(text.split()) for text in texts]
            
            unique_labels, counts = np.unique(labels, return_counts=True)
            label_dist = dict(zip(unique_labels, counts))
            
            stats[split] = {
                'num_samples': len(texts),
                'avg_text_length': np.mean(text_lengths),
                'max_text_length': np.max(text_lengths),
                'min_text_length': np.min(text_lengths),
                'label_distribution': label_dist
            }
        
        return stats
