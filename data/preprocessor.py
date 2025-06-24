
import re
import string
import logging
from typing import List, Dict, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np

from utils.logging_utils import get_logger

logger = get_logger(__name__)

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    logger.warning("Failed to download NLTK data")

class TextPreprocessor:
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        
        text = text.lower()
        
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        text = re.sub(r'\S+@\S+', '', text)
        
        text = re.sub(r'\s+', ' ', text)
        
        text = re.sub(r'[^\w\s\.,!?;:]', '', text)
        
        return text.strip()
    
    def tokenize_text(self, text: str) -> List[str]:
        try:
            tokens = word_tokenize(text)
            return tokens
        except:
            return text.split()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        try:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        except:
            return tokens
    
    def preprocess_text(self, text: str, 
                       remove_stopwords: bool = False,
                       lemmatize: bool = False) -> str:
        
        cleaned_text = self.clean_text(text)
        
        if remove_stopwords or lemmatize:
            tokens = self.tokenize_text(cleaned_text)
            
            if remove_stopwords:
                tokens = self.remove_stopwords(tokens)
            
            if lemmatize:
                tokens = self.lemmatize_tokens(tokens)
            
            return ' '.join(tokens)
        
        return cleaned_text
    
    def extract_features(self, text: str) -> Dict:
        
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = len(sent_tokenize(text))
        
        punct_count = sum(1 for char in text if char in string.punctuation)
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        upper_count = sum(1 for char in text if char.isupper())
        upper_ratio = upper_count / max(char_count, 1)
        
        digit_count = sum(1 for char in text if char.isdigit())
        avg_word_length = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': sentence_count,
            'punct_count': punct_count,
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'upper_ratio': upper_ratio,
            'digit_count': digit_count,
            'avg_word_length': avg_word_length,
            'avg_sentence_length': word_count / max(sentence_count, 1)
        }

class ConsistencyChecker:
    
    def __init__(self, config):
        self.config = config
        self.preprocessor = TextPreprocessor()
        
    def check_semantic_consistency(self, rationale: str, prediction: int) -> float:
        
        if not rationale:
            return 0.0
        
        rationale_lower = rationale.lower()
        
        fake_indicators = [
            'questionable', 'doubtful', 'suspicious', 'unverified', 
            'claim', 'alleged', 'rumor', 'misleading', 'false',
            'fabricated', 'hoax', 'conspiracy', 'debunked'
        ]
        
        real_indicators = [
            'verified', 'confirmed', 'official', 'documented',
            'evidence', 'study', 'research', 'expert', 'credible',
            'factual', 'authentic', 'legitimate', 'proven'
        ]
        
        fake_score = sum(1 for indicator in fake_indicators if indicator in rationale_lower)
        real_score = sum(1 for indicator in real_indicators if indicator in rationale_lower)
        
        total_indicators = len(fake_indicators) + len(real_indicators)
        fake_ratio = fake_score / len(fake_indicators)
        real_ratio = real_score / len(real_indicators)
        
        if prediction == 1:
            consistency = fake_ratio / (fake_ratio + real_ratio + 0.001)
        else:
            consistency = real_ratio / (fake_ratio + real_ratio + 0.001)
        
        if len(rationale.split()) < 10:
            consistency *= 0.8
        
        return min(1.0, max(0.0, consistency))
    
    def check_cross_consistency(self, rationales: List[str]) -> Dict:
        
        if not rationales:
            return {'consistency': 0.0, 'agreement': 0.0}
        
        features_list = []
        for rationale in rationales:
            features = self.preprocessor.extract_features(rationale)
            features_list.append(features)
        
        consistency_scores = []
        for feature_name in features_list[0].keys():
            values = [features[feature_name] for features in features_list]
            if len(set(values)) > 1:
                variance = np.var(values)
                consistency = 1.0 / (1.0 + variance)
            else:
                consistency = 1.0
            consistency_scores.append(consistency)
        
        avg_consistency = np.mean(consistency_scores)
        
        semantic_agreement = self._calculate_semantic_agreement(rationales)
        
        return {
            'consistency': avg_consistency,
            'agreement': semantic_agreement,
            'num_rationales': len(rationales)
        }
    
    def _calculate_semantic_agreement(self, rationales: List[str]) -> float:
        
        if len(rationales) < 2:
            return 1.0
        
        all_words = set()
        rationale_words = []
        
        for rationale in rationales:
            words = set(self.preprocessor.tokenize_text(rationale.lower()))
            rationale_words.append(words)
            all_words.update(words)
        
        if not all_words:
            return 0.0
        
        overlaps = []
        for i in range(len(rationale_words)):
            for j in range(i + 1, len(rationale_words)):
                intersection = rationale_words[i].intersection(rationale_words[j])
                union = rationale_words[i].union(rationale_words[j])
                
                if union:
                    overlap = len(intersection) / len(union)
                    overlaps.append(overlap)
        
        return np.mean(overlaps) if overlaps else 0.0
