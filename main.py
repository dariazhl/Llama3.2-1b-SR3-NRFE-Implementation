
import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from config import Config
from utils.logging_utils import setup_logging, get_logger
from data.data_loader import DataLoader
from training.nrfe_trainer import NRFETrainer
from training.sr3_trainer import SR3Trainer
from evaluation.evaluator import ModelEvaluator
from inference.pipeline import InferencePipeline

setup_logging(log_level="INFO", log_file="logs/main.log")
logger = get_logger(__name__)

def train_nrfe(config):
    logger.info("Starting Enhanced NRFE training...")
    
    try:
        data_loader = DataLoader(config)
        
        trainer = NRFETrainer(config)
        
        model, metrics = trainer.train(data_loader)
        
        logger.info(f"NRFE training completed. Final metrics: {metrics}")
        return model, metrics
        
    except Exception as e:
        logger.error(f"Error in NRFE training: {e}")
        raise

def train_sr3(config):
    logger.info("Starting SR³ training...")
    
    try:
        data_loader = DataLoader(config)
        
        trainer = SR3Trainer(config)
        
        model, metrics = trainer.train(data_loader)
        
        logger.info(f"SR³ training completed. Final metrics: {metrics}")
        return model, metrics
        
    except Exception as e:
        logger.error(f"Error in SR³ training: {e}")
        raise

def evaluate_models(config):
    logger.info("Starting model evaluation...")
    
    try:
        data_loader = DataLoader(config)
        evaluator = ModelEvaluator(config)
        
        data_loader.load_dataset()
        test_dataloader = data_loader.get_dataloader("test", shuffle=False)
        
        results = evaluator.evaluate_all_models(test_dataloader)
        
        logger.info("Model evaluation completed")
        for model_name, metrics in results.items():
            logger.info(f"{model_name}: {metrics}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        raise

def run_inference(config, text):
    logger.info(f"Running inference on text: {text[:100]}...")
    
    try:
        pipeline = InferencePipeline(config)
        
        result = pipeline.predict(text)
        
        logger.info("Inference completed")
        return result
        
    except Exception as e:
        logger.error(f"Error in inference: {e}")
        raise

def compare_models(config):
    logger.info("Starting model comparison...")
    
    try:
        evaluator = ModelEvaluator(config)
        
        data_loader = DataLoader(config)
        data_loader.load_dataset()
        test_dataloader = data_loader.get_dataloader("test", shuffle=False)
        
        comparison = evaluator.compare_models(test_dataloader)
        
        logger.info("Model comparison completed")
        return comparison
        
    except Exception as e:
        logger.error(f"Error in model comparison: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Enhanced NRFE Fake News Detection System")
    parser.add_argument("--mode", type=str, required=True,
                       choices=["train_nrfe", "train_sr3", "evaluate", "inference", "compare_models"],
                       help="Operation mode")
    parser.add_argument("--text", type=str, help="Text for inference mode")
    parser.add_argument("--config", type=str, help="Path to config file")
    
    args = parser.parse_args()
    
    config = Config()
    
    logger.info(f"Starting application in {args.mode} mode")
    
    try:
        if args.mode == "train_nrfe":
            model, metrics = train_nrfe(config)
            print(f"\nTraining completed successfully!")
            print(f"Final metrics: {metrics}")
            
        elif args.mode == "train_sr3":
            model, metrics = train_sr3(config)
            print(f"\nTraining completed successfully!")
            print(f"Final metrics: {metrics}")
            
        elif args.mode == "evaluate":
            results = evaluate_models(config)
            print(f"\nEvaluation completed successfully!")
            for model_name, metrics in results.items():
                print(f"{model_name}: {metrics}")
                
        elif args.mode == "inference":
            if not args.text:
                print("Error: --text argument required for inference mode")
                sys.exit(1)
            
            result = run_inference(config, args.text)
            print(f"\nInference result:")
            print(f"Text: {args.text}")
            print(f"Prediction: {'Fake' if result['prediction'] == 1 else 'Real'}")
            print(f"Confidence: {result['confidence']:.4f}")
            
        elif args.mode == "compare_models":
            comparison = compare_models(config)
            print(f"\nModel comparison completed!")
            print(comparison)
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        print("\nTraining stopped by user")
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
