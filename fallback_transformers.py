
class MockAutoTokenizer:
    
    def __init__(self):
        self.pad_token = "[PAD]"
        self.pad_token_id = 0
        self.unk_token = "[UNK]"
        self.unk_token_id = 1
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.vocab_size = 30522
    
    @classmethod
    def from_pretrained(cls, model_name):
        return cls()
    
    def __call__(self, text, max_length=512, padding=True, truncation=True, return_tensors="pt"):
        import torch
        return {
            'input_ids': torch.randint(0, 1000, (1, max_length)),
            'attention_mask': torch.ones(1, max_length)
        }
    
    def decode(self, token_ids):
        return "mock decoded text"

class MockAutoModel:
    
    @classmethod
    def from_pretrained(cls, model_name):
        return cls()
    
    def __call__(self, input_ids, attention_mask=None):
        import torch
        return type('MockOutput', (), {
            'last_hidden_state': torch.randn(1, input_ids.shape[1], 768)
        })()

AutoTokenizer = MockAutoTokenizer
AutoModel = MockAutoModel

class MockBertModel:
    @classmethod
    def from_pretrained(cls, model_name):
        return cls()

BertModel = MockBertModel

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, **kwargs):
    return None