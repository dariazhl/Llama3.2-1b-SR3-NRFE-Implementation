
import os
from pathlib import Path

class Config:
    
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    
    MODEL_NAME = "bert-base-uncased"
    MAX_LENGTH = 512
    HIDDEN_SIZE = 768
    NUM_CLASSES = 2
    DROPOUT_RATE = 0.1
    
    CROSS_ATTENTION_HEADS = 8
    CROSS_ATTENTION_LAYERS = 2
    REASONING_HIDDEN_SIZE = 256
    CONSISTENCY_THRESHOLD = 0.7
    
    SR3_LAMBDA_CONSISTENCY = 0.5
    SR3_NUM_RATIONALES = 3
    SR3_TEMPERATURE = 0.8
    
    BATCH_SIZE = 8
    NUM_EPOCHS = 1
    LEARNING_RATE = 2e-5
    WARMUP_STEPS = 100
    GRADIENT_ACCUMULATION_STEPS = 1
    LOGGING_STEPS = 50
    EVAL_STEPS = 200
    SAVE_STEPS = 500
    
    RANDOM_SEED = 42
    
    WANDB_PROJECT = "enhanced-nrfe-fake-news"
    WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
    
    DEVICE = "cpu"
    
    TRAIN_SIZE = 0.6
    VAL_SIZE = 0.2
    TEST_SIZE = 0.2
    
    SAMPLE_NEWS_DATA = [
        {
            "text": "Scientists at Stanford University have discovered a breakthrough treatment for cancer that shows 95% success rate in clinical trials.",
            "label": 0,
            "explanation": "This appears to be a factual scientific claim that can be verified."
        },
        {
            "text": "Local man discovers aliens living in his backyard shed, government tries to cover it up.",
            "label": 1,
            "explanation": "This claim lacks credible sources and evidence, typical of fake news."
        },
        {
            "text": "New economic policy announced by the Federal Reserve aims to reduce inflation by 2% over the next year.",
            "label": 0,
            "explanation": "Economic policy announcements are typically factual and verifiable."
        },
        {
            "text": "Celebrity spotted drinking magical potion that makes people immortal, doctors hate this one trick.",
            "label": 1,
            "explanation": "Sensational claims about impossible health benefits are typically fake news."
        },
        {
            "text": "Researchers at MIT publish peer-reviewed study on renewable energy efficiency improvements.",
            "label": 0,
            "explanation": "Academic research publications from reputable institutions are typically factual."
        },
        {
            "text": "World leaders secretly replaced by lizard people, exclusive photos leaked online.",
            "label": 1,
            "explanation": "Conspiracy theories without credible evidence are characteristic of fake news."
        },
        {
            "text": "NASA announces successful Mars rover mission with new geological discoveries.",
            "label": 0,
            "explanation": "Official NASA announcements about space missions are typically accurate."
        },
        {
            "text": "Miracle diet pill melts fat overnight, pharmaceutical companies don't want you to know.",
            "label": 1,
            "explanation": "Unrealistic health claims with conspiracy elements are typical fake news patterns."
        }
    ]
