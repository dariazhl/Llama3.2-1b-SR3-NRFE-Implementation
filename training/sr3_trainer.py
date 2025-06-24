
import logging
import os
import torch
from tqdm import tqdm

from .trainer import BaseTrainer, EarlyStopping, MetricsTracker
from models.sr3_model import SR3Model, SR3Loss
from utils.logging_utils import get_logger

logger = get_logger(__name__)

class SR3Trainer(BaseTrainer):
    
    def __init__(self, config):
        super().__init__(config)
        self.loss_fn = SR3Loss(config)
        self.metrics_tracker = MetricsTracker()
        
    def _forward_pass(self, model, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        correct_rationales = batch.get('correct_rationales')
        incorrect_rationales = batch.get('incorrect_rationales')
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            training_mode="joint"
        )
        
        return outputs
    
    def train(self, data_loader):
        logger.info("Starting SR³ training...")
        
        model = SR3Model(self.config).to(self.device)
        
        logger.info("Stage 1: Pre-training rationale generator...")
        self._pretrain_rationale_generator(model, data_loader)
        
        logger.info("Stage 2: Generating rationales...")
        rationale_data = self._generate_training_rationales(model, data_loader)
        
        logger.info("Stage 3: Joint training with consistency learning...")
        self._joint_training(model, rationale_data, data_loader)
        
        logger.info("Final evaluation...")
        eval_dataloader = data_loader.get_dataloader("validation", shuffle=False)
        final_metrics = self.evaluate(model, eval_dataloader)
        
        model_path = self.config.MODELS_DIR / "sr3_final_model.pt"
        self.save_model(model, model_path, final_metrics)
        
        logger.info("SR³ training completed!")
        return model, final_metrics
    
    def _pretrain_rationale_generator(self, model, data_loader):
        logger.info("Pre-training rationale generator...")
        
        train_dataloader = data_loader.get_dataloader("train")
        eval_dataloader = data_loader.get_dataloader("validation", shuffle=False)
        
        rationale_params = list(model.rationale_generator.parameters())
        optimizer = torch.optim.AdamW(
            rationale_params,
            lr=self.config.LEARNING_RATE * 0.5
        )
        
        early_stopping = EarlyStopping(patience=2)
        
        for epoch in range(2):
            model.train()
            total_loss = 0
            
            progress_bar = tqdm(train_dataloader, desc=f"Pretraining Epoch {epoch+1}")
            
            for batch in progress_bar:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = model(
                    batch['input_ids'],
                    batch['attention_mask'],
                    batch['labels'],
                    training_mode="rationale_only"
                )
                
                loss = outputs['loss']
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(rationale_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(train_dataloader)
            logger.info(f"Pretraining epoch {epoch+1}, avg loss: {avg_loss:.4f}")
            
            if early_stopping(avg_loss, model):
                logger.info("Early stopping triggered for pretraining")
                break
    
    def _generate_training_rationales(self, model, data_loader):
        logger.info("Generating rationales for training data...")
        
        rationale_data = data_loader.prepare_rationale_data(
            model.rationale_generator,
            split="train",
            num_samples=min(5000, len(data_loader.dataset['train']))
        )
        
        logger.info(f"Generated rationales for {len(rationale_data)} samples")
        return rationale_data
    
    def _joint_training(self, model, rationale_data, data_loader):
        logger.info("Starting joint training...")
        
        joint_dataloader = self._create_rationale_dataloader(rationale_data)
        eval_dataloader = data_loader.get_dataloader("validation", shuffle=False)
        
        optimizer, scheduler = self.setup_optimizer_and_scheduler(model, joint_dataloader)
        early_stopping = EarlyStopping(patience=3)
        
        best_f1 = 0
        
        for epoch in range(self.config.NUM_EPOCHS):
            train_loss = self.train_epoch(model, joint_dataloader, optimizer, scheduler, epoch)
            
            eval_metrics = self.evaluate(model, eval_dataloader, epoch)
            
            self.metrics_tracker.update({
                'train_loss': train_loss,
                **eval_metrics
            })
            
            if eval_metrics['eval_f1'] > best_f1:
                best_f1 = eval_metrics['eval_f1']
                model_path = self.config.MODELS_DIR / f"sr3_best_model_epoch_{epoch}.pt"
                self.save_model(model, model_path, eval_metrics)
            
            if early_stopping(eval_metrics['eval_loss'], model):
                logger.info("Early stopping triggered")
                break
            
            logger.info(f"Epoch {epoch+1} completed - F1: {eval_metrics['eval_f1']:.4f}")
    
    def _create_rationale_dataloader(self, rationale_data):
        from torch.utils.data import Dataset, DataLoader
        
        class RationaleDataset(Dataset):
            def __init__(self, data, tokenizer, max_length):
                self.data = data
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = self.data[idx]
                
                encoding = self.tokenizer(
                    item['text'],
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(item['label'], dtype=torch.long),
                    'correct_rationales': item['correct_rationale'],
                    'incorrect_rationales': item['incorrect_rationale']
                }
        
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
        
        dataset = RationaleDataset(rationale_data, tokenizer, self.config.MAX_LENGTH)
        
        return DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=2
        )
    
    def evaluate_consistency(self, model, eval_dataloader):
        model.eval()
        consistency_scores = []
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating consistency"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = model(
                    batch['input_ids'],
                    batch['attention_mask'],
                    batch['labels'],
                    training_mode="joint"
                )
                
                rationales = outputs.get('rationales', [])
                logits = outputs['logits']
                predictions = torch.argmax(logits, dim=-1)
                
                for i, sample_rationales in enumerate(rationales):
                    prediction = predictions[i].item()
                    
                    sample_consistencies = []
                    for rationale in sample_rationales:
                        consistency = model.consistency_checker.check_semantic_consistency(
                            rationale, prediction
                        )
                        sample_consistencies.append(consistency)
                    
                    if sample_consistencies:
                        avg_consistency = sum(sample_consistencies) / len(sample_consistencies)
                        consistency_scores.append(avg_consistency)
        
        avg_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0
        
        logger.info(f"Average consistency score: {avg_consistency:.4f}")
        return avg_consistency
