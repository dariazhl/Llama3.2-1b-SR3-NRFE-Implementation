
import torch
import numpy as np
from typing import Dict, List, Optional, Union
try:
    from transformers import AutoTokenizer
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from fallback_transformers import AutoTokenizer
from pathlib import Path

from models.nrfe_model import EnhancedNRFEModel, NRFEModel
from models.sr3_model import SR3Model
from models.base_model import FakeNewsClassifier
from data.preprocessor import TextPreprocessor, ConsistencyChecker
from utils.logging_utils import get_logger

logger = get_logger(__name__)

class InferencePipeline:
    
    def __init__(self, config, model_type: str = "enhanced_nrfe"):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        
        self.preprocessor = TextPreprocessor()
        self.consistency_checker = ConsistencyChecker(config)
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = self._load_model()
        
        self.class_labels = {0: "Real", 1: "Fake"}
        
    def _load_model(self):
        
        models_dir = self.config.MODELS_DIR
        
        model_files = {
            "enhanced_nrfe": "nrfe_final_model.pt",
            "sr3": "sr3_final_model.pt", 
            "base_bert": "base_bert_model.pt",
            "nrfe": "nrfe_model.pt"
        }
        
        model_file = model_files.get(self.model_type, "nrfe_final_model.pt")
        model_path = models_dir / model_file
        
        try:
            if self.model_type == "enhanced_nrfe":
                model = EnhancedNRFEModel(self.config)
            elif self.model_type == "sr3":
                model = SR3Model(self.config)
            elif self.model_type == "nrfe":
                model = NRFEModel(self.config)
            else:
                model = FakeNewsClassifier(self.config)
            
            if model_path.exists():
                logger.info(f"Loading model from {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                logger.warning(f"Model file not found: {model_path}")
                logger.info("Using randomly initialized model for demonstration")
            
            model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Falling back to base BERT classifier")
            model = FakeNewsClassifier(self.config)
            model.to(self.device)
            model.eval()
            return model
    
    def predict(self, text: str, return_reasoning: bool = True, 
                return_attention: bool = True) -> Dict:
        
        try:
            processed_text = self.preprocessor.preprocess_text(text)
            
            if isinstance(self.model, EnhancedNRFEModel):
                return self._predict_enhanced_nrfe(processed_text, return_reasoning, return_attention)
            elif isinstance(self.model, SR3Model):
                return self._predict_sr3(processed_text, return_reasoning)
            else:
                return self._predict_base_model(processed_text, return_reasoning)
                
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {
                'error': str(e),
                'prediction': 0,
                'confidence': 0.5,
                'label': 'Unknown'
            }
    
    def _predict_enhanced_nrfe(self, text: str, return_reasoning: bool, 
                              return_attention: bool) -> Dict:
        
        positive_reasoning = self._generate_positive_reasoning(text)
        negative_reasoning = self._generate_negative_reasoning(text)
        
        with torch.no_grad():
            outputs = self.model(
                news_text=text,
                positive_reasoning=positive_reasoning,
                negative_reasoning=negative_reasoning
            )
            
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1).item()
            confidence = probabilities.max().item()
            
            result = {
                'prediction': prediction,
                'confidence': confidence,
                'label': self.class_labels[prediction],
                'probabilities': {
                    'Real': probabilities[0][0].item(),
                    'Fake': probabilities[0][1].item()
                }
            }
            
            if return_reasoning:
                result['reasoning'] = {
                    'positive': positive_reasoning,
                    'negative': negative_reasoning
                }
                
                if 'consistency_scores' in outputs:
                    result['consistency_scores'] = outputs['consistency_scores']
            
            if return_attention and 'cross_attention_weights' in outputs:
                result['attention_weights'] = {
                    k: v.cpu().numpy().tolist() 
                    for k, v in outputs['cross_attention_weights'].items()
                }
            
            return result
    
    def _predict_sr3(self, text: str, return_reasoning: bool) -> Dict:
        
        result = self.model.generate_and_classify(text)
        
        formatted_result = {
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'label': self.class_labels[result['prediction']],
            'probabilities': {
                'Real': 1 - result['confidence'] if result['prediction'] == 1 else result['confidence'],
                'Fake': result['confidence'] if result['prediction'] == 1 else 1 - result['confidence']
            }
        }
        
        if return_reasoning:
            formatted_result['reasoning'] = {
                'rationales': result['rationales'],
                'consistency_scores': result['consistency_scores'],
                'average_consistency': result['average_consistency']
            }
        
        return formatted_result
    
    def _predict_base_model(self, text: str, return_reasoning: bool) -> Dict:
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.config.MAX_LENGTH
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1).item()
            confidence = probabilities.max().item()
            
            result = {
                'prediction': prediction,
                'confidence': confidence,
                'label': self.class_labels[prediction],
                'probabilities': {
                    'Real': probabilities[0][0].item(),
                    'Fake': probabilities[0][1].item()
                }
            }
            
            if return_reasoning:
                result['reasoning'] = self._generate_simple_reasoning(text, prediction)
            
            return result
    
    def _generate_positive_reasoning(self, text: str) -> str:
        
        features = self.preprocessor.extract_features(text)
        
        reasoning_parts = []
        
        if features['word_count'] > 50:
            reasoning_parts.append("The article has substantial content with detailed information")
        
        if features['avg_sentence_length'] > 15:
            reasoning_parts.append("sentence structure appears professional and well-formed")
        
        if features['upper_ratio'] < 0.1:
            reasoning_parts.append("the tone is measured without excessive capitalization")
        
        if features['exclamation_count'] <= 2:
            reasoning_parts.append("the language is restrained and factual")
        
        if not reasoning_parts:
            reasoning_parts.append("the content follows standard journalistic conventions")
        
        return f"This appears to be credible news because {', '.join(reasoning_parts)}."
    
    def _generate_negative_reasoning(self, text: str) -> str:
        
        features = self.preprocessor.extract_features(text)
        
        reasoning_parts = []
        
        if features['exclamation_count'] > 3:
            reasoning_parts.append("excessive use of exclamation marks suggests sensationalism")
        
        if features['upper_ratio'] > 0.15:
            reasoning_parts.append("high proportion of uppercase text indicates emotional manipulation")
        
        if features['word_count'] < 30:
            reasoning_parts.append("the content is too brief to provide adequate context")
        
        if features['question_count'] > 5:
            reasoning_parts.append("excessive questions may indicate speculative content")
        
        if not reasoning_parts:
            reasoning_parts.append("certain linguistic patterns could suggest bias or manipulation")
        
        return f"This might be questionable because {', '.join(reasoning_parts)}."
    
    def _generate_simple_reasoning(self, text: str, prediction: int) -> str:
        
        features = self.preprocessor.extract_features(text)
        
        if prediction == 0:
            return (f"Classified as real news based on text analysis. "
                   f"The article has {features['word_count']} words with "
                   f"balanced linguistic features suggesting credible journalism.")
        else:
            return (f"Classified as fake news based on text analysis. "
                   f"The content shows {features['exclamation_count']} exclamation marks and "
                   f"{features['upper_ratio']:.1%} uppercase text, which may indicate "
                   f"sensationalism or emotional manipulation.")
    
    def predict_batch(self, texts: List[str], batch_size: int = 8) -> List[Dict]:
        
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = []
            
            for text in batch_texts:
                result = self.predict(text, return_reasoning=False, return_attention=False)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    def self_rectify_prediction(self, text: str, max_iterations: int = 3) -> Dict:
        
        if not isinstance(self.model, SR3Model):
            return {
                'error': 'Self-rectification only available for SR³ model',
                'standard_prediction': self.predict(text)
            }
        
        try:
            result = self.model.self_rectify(text, max_iterations)
            
            return {
                'original_text': result['original_text'],
                'final_prediction': result['final_prediction'],
                'final_confidence': result['final_confidence'],
                'final_label': self.class_labels[result['final_prediction']],
                'iterations': result['iterations'],
                'num_iterations': result['num_iterations'],
                'improvement': (
                    result['iterations'][-1]['average_consistency'] - 
                    result['iterations'][0]['average_consistency']
                ) if len(result['iterations']) > 1 else 0
            }
            
        except Exception as e:
            logger.error(f"Error in self-rectification: {e}")
            return {
                'error': str(e),
                'standard_prediction': self.predict(text)
            }
    
    def get_model_info(self) -> Dict:
        
        from utils.model_utils import get_model_size
        
        return {
            'model_type': self.model_type,
            'model_class': self.model.__class__.__name__,
            'device': str(self.device),
            'model_size': get_model_size(self.model),
            'tokenizer': self.config.MODEL_NAME,
            'max_length': self.config.MAX_LENGTH,
            'num_classes': self.config.NUM_CLASSES
        }
    
    def explain_prediction(self, text: str, method: str = "simple") -> Dict:
        
        prediction_result = self.predict(text)
        
        explanation = {
            'prediction_summary': prediction_result,
            'explanation_method': method
        }
        
        if method == "simple":
            explanation['explanation'] = self._simple_explanation(text, prediction_result)
        elif method == "features":
            explanation['explanation'] = self._feature_based_explanation(text, prediction_result)
        elif method == "attention" and 'attention_weights' in prediction_result:
            explanation['explanation'] = self._attention_based_explanation(text, prediction_result)
        else:
            explanation['explanation'] = "Explanation method not available for this model."
        
        return explanation
    
    def _simple_explanation(self, text: str, prediction: Dict) -> str:
        
        label = prediction['label']
        confidence = prediction['confidence']
        
        if confidence > 0.8:
            certainty = "very confident"
        elif confidence > 0.6:
            certainty = "moderately confident"
        else:
            certainty = "somewhat uncertain"
        
        return (f"The model predicts this news is {label.lower()} and is {certainty} "
               f"in this assessment (confidence: {confidence:.1%}). "
               f"This prediction is based on various linguistic and structural "
               f"features of the text that are associated with {label.lower()} news.")
    
    def _feature_based_explanation(self, text: str, prediction: Dict) -> str:
        
        features = self.preprocessor.extract_features(text)
        label = prediction['label']
        
        key_features = []
        
        if features['exclamation_count'] > 2:
            key_features.append(f"high number of exclamation marks ({features['exclamation_count']})")
        
        if features['upper_ratio'] > 0.1:
            key_features.append(f"elevated uppercase usage ({features['upper_ratio']:.1%})")
        
        if features['avg_word_length'] > 6:
            key_features.append("sophisticated vocabulary")
        elif features['avg_word_length'] < 4:
            key_features.append("simple vocabulary")
        
        if features['word_count'] < 50:
            key_features.append("brief content")
        elif features['word_count'] > 200:
            key_features.append("detailed content")
        
        feature_text = ", ".join(key_features) if key_features else "standard linguistic patterns"
        
        return (f"Predicted as {label.lower()} based on text analysis revealing: {feature_text}. "
               f"The article contains {features['word_count']} words across "
               f"{features['sentence_count']} sentences, with an average word length of "
               f"{features['avg_word_length']:.1f} characters.")
    
    def _attention_based_explanation(self, text: str, prediction: Dict) -> str:
        
        if 'attention_weights' not in prediction:
            return "Attention analysis not available for this prediction."
        
        attention_weights = prediction['attention_weights']
        label = prediction['label']
        
        max_attention_type = max(attention_weights.keys(), 
                               key=lambda k: np.mean(attention_weights[k]))
        
        attention_descriptions = {
            'fp_to_x': 'positive reasoning to news content',
            'fx_to_p': 'news content to positive reasoning',
            'fn_to_x': 'negative reasoning to news content', 
            'fx_to_n': 'news content to negative reasoning'
        }
        
        primary_focus = attention_descriptions.get(max_attention_type, 'content analysis')
        
        return (f"Predicted as {label.lower()} with primary attention focus on {primary_focus}. "
               f"The model's cross-attention mechanism shows strongest activation in "
               f"{max_attention_type} patterns, indicating the key factors driving this "
               f"classification decision.")
