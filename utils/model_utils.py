
import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Union
import json

from utils.logging_utils import get_logger

logger = get_logger(__name__)

def load_model_checkpoint(checkpoint_path: Union[str, Path], 
                          device: Optional[torch.device] = None) -> Dict:
    
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        logger.info("Checkpoint loaded successfully")
        return checkpoint
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        raise

def save_model_checkpoint(model: nn.Module, checkpoint_path: Union[str, Path],
                          config: Optional[Dict] = None, metrics: Optional[Dict] = None,
                          optimizer: Optional[torch.optim.Optimizer] = None,
                          scheduler: Optional[object] = None, epoch: Optional[int] = None):
    
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__
    }
    
    if config:
        checkpoint['config'] = config
    
    if metrics:
        checkpoint['metrics'] = metrics
    
    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if scheduler:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    try:
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")
        raise

def get_model_size(model: nn.Module) -> Dict:
    
    param_size = 0
    buffer_size = 0
    trainable_params = 0
    total_params = 0
    
    for param in model.parameters():
        param_bytes = param.nelement() * param.element_size()
        param_size += param_bytes
        total_params += param.nelement()
        
        if param.requires_grad:
            trainable_params += param.nelement()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'parameter_size_mb': param_size / 1024 / 1024,
        'buffer_size_mb': buffer_size / 1024 / 1024,
        'total_size_mb': total_size_mb
    }

def count_parameters(model: nn.Module) -> Dict:
    
    counts = {}
    
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        counts[name] = {
            'total': module_params,
            'trainable': trainable_params,
            'frozen': module_params - trainable_params
        }
    
    return counts

def freeze_model_parameters(model: nn.Module, modules_to_freeze: Optional[list] = None):
    
    if modules_to_freeze is None:
        for param in model.parameters():
            param.requires_grad = False
        logger.info("Froze all model parameters")
    else:
        for module_name in modules_to_freeze:
            if hasattr(model, module_name):
                module = getattr(model, module_name)
                for param in module.parameters():
                    param.requires_grad = False
                logger.info(f"Froze parameters in module: {module_name}")
            else:
                logger.warning(f"Module {module_name} not found in model")

def unfreeze_model_parameters(model: nn.Module, modules_to_unfreeze: Optional[list] = None):
    
    if modules_to_unfreeze is None:
        for param in model.parameters():
            param.requires_grad = True
        logger.info("Unfroze all model parameters")
    else:
        for module_name in modules_to_unfreeze:
            if hasattr(model, module_name):
                module = getattr(model, module_name)
                for param in module.parameters():
                    param.requires_grad = True
                logger.info(f"Unfroze parameters in module: {module_name}")
            else:
                logger.warning(f"Module {module_name} not found in model")

def initialize_weights(model: nn.Module, initialization_method: str = "xavier_uniform"):
    
    def init_weights(m):
        if isinstance(m, nn.Linear):
            if initialization_method == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight)
            elif initialization_method == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight)
            elif initialization_method == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight)
            elif initialization_method == "kaiming_normal":
                torch.nn.init.kaiming_normal_(m.weight)
            
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, nn.Conv2d):
            if initialization_method == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight)
            elif initialization_method == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight)
            
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    logger.info(f"Initialized model weights using {initialization_method}")

def compare_model_parameters(model1: nn.Module, model2: nn.Module) -> Dict:
    
    comparison = {
        'identical_parameters': 0,
        'different_parameters': 0,
        'total_parameters': 0,
        'parameter_differences': {}
    }
    
    model1_params = dict(model1.named_parameters())
    model2_params = dict(model2.named_parameters())
    
    all_param_names = set(model1_params.keys()).union(set(model2_params.keys()))
    
    for param_name in all_param_names:
        comparison['total_parameters'] += 1
        
        if param_name not in model1_params:
            comparison['parameter_differences'][param_name] = "Only in model2"
            comparison['different_parameters'] += 1
        elif param_name not in model2_params:
            comparison['parameter_differences'][param_name] = "Only in model1"
            comparison['different_parameters'] += 1
        else:
            param1 = model1_params[param_name]
            param2 = model2_params[param_name]
            
            if torch.equal(param1, param2):
                comparison['identical_parameters'] += 1
            else:
                comparison['different_parameters'] += 1
                diff = torch.norm(param1 - param2).item()
                comparison['parameter_differences'][param_name] = f"Difference norm: {diff:.6f}"
    
    return comparison

def save_model_config(config: Dict, config_path: Union[str, Path]):
    
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    serializable_config = {}
    for key, value in config.items():
        if isinstance(value, (str, int, float, bool, list, dict, type(None))):
            serializable_config[key] = value
        else:
            serializable_config[key] = str(value)
    
    try:
        with open(config_path, 'w') as f:
            json.dump(serializable_config, f, indent=2)
        logger.info(f"Model config saved to {config_path}")
    except Exception as e:
        logger.error(f"Error saving model config: {e}")
        raise

def load_model_config(config_path: Union[str, Path]) -> Dict:
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Model config loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading model config: {e}")
        raise
