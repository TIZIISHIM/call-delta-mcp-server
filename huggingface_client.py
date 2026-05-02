import os
import re
from typing import Dict, List

class HuggingFaceClient:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.model_loaded = False
        self.load_finbert()
    
    def load_finbert(self):
        """Load FinBERT model locally - this is the free, self-hosted way"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            print("Loading FinBERT model locally...")
            model_name = "ProsusAI/finbert"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()
            self.model_loaded = True
            print("FinBERT model loaded successfully")
            
        except ImportError:
            print("ERROR: transformers or torch not installed. Run: pip install transformers torch")
            self.model_loaded = False
        except Exception as e:
            print(f"ERROR loading FinBERT: {str(e)}")
            self.model_loaded = False
    
    def analyze_sentiment_with_evidence(self, text: str, max_sentences: int = 150) -> Dict:
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15][:max_sentences]
        
        if not sentences:
            return {
                'sentiment_label': 'neutral',
                'sentiment_score': 0.5,
                'confidence': 0.0,
                'evidence': [],
                'sentence_count': 0,
                'error': 'No valid sentences found'
            }
        
        # Process sentences
        sentence_results = []
        
        for sentence in sentences:
            result = self._analyze_sentence(sentence)
            sentence_results.append(result)
        
        # Calculate overall sentiment
        scores = [r['sentiment_score'] for r in sentence_results if r.get('sentiment_score') is not None]
        if scores:
            overall_score = sum(scores) / len(scores)
        else:
            overall_score = 0.5
            
        label = 'positive' if overall_score > 0.55 else ('negative' if overall_score < 0.45 else 'neutral')
        
        return {
            'sentiment_label': label,
            'sentiment_score': round(overall_score, 3),
            'confidence': round(abs(overall_score - 0.5) * 2, 3),
            'evidence': sentence_results[:50],
            'sentence_count': len(sentence_results)
        }
    
    def _analyze_sentence(self, sentence: str) -> Dict:
        if not self.model_loaded:
            return self._fallback_sentiment(sentence)
        
        try:
            import torch
            
            inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512, padding=True)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                scores = probabilities[0].tolist()
                
                # FinBERT labels: positive, negative, neutral
                # Index mapping depends on the model, typically: 0=positive, 1=negative, 2=neutral
                # Or: 0=negative, 1=neutral, 2=positive
                # Let's determine dynamically
                label_map = {0: "positive", 1: "negative", 2: "neutral"}
                
                # Get the highest probability
                max_idx = torch.argmax(probabilities[0]).item()
                label = label_map.get(max_idx, "neutral")
                score = scores[max_idx]
                
                # Convert to sentiment score (0-1 scale where 0.5 is neutral)
                if label == "positive":
                    sentiment_score = score
                elif label == "negative":
                    sentiment_score = 1 - score
                else:
                    sentiment_score = 0.5
                
                return {
                    'sentence': sentence[:300],
                    'sentiment_label': label,
                    'sentiment_score': round(sentiment_score, 3),
                    'confidence': round(score, 3),
                    'source': 'finbert-local'
                }
                
        except Exception as e:
            print(f"FinBERT local error: {str(e)}")
            return self._fallback_sentiment(sentence)
    
    def _fallback_sentiment(self, sentence: str) -> Dict:
        """Rule-based fallback for when local model fails."""
        sentence_lower = sentence.lower()
        
        positive_words = [
            'growth', 'grew', 'increase', 'increased', 'rising', 'up', 'higher',
            'record', 'strong', 'confidence', 'confident', 'optimistic', 'beat',
            'exceed', 'outperform', 'raise', 'raised', 'guidance', 'momentum',
            'accelerate', 'accelerated', 'expansion', 'expand', 'profit', 'profitable',
            'outlook', 'improve', 'improved', 'improvement', 'better', 'best'
        ]
        
        negative_words = [
            'decline', 'declined', 'decrease', 'decreased', 'fall', 'fell', 'down', 'lower',
            'weak', 'weakness', 'challenge', 'headwind', 'pressure', 'stress', 'risk',
            'miss', 'missed', 'below', 'reduce', 'reduced', 'cut', 'slash', 'drop', 'dropped',
            'collapsed', 'collapse', 'loss', 'lose', 'uncertain', 'uncertainty',
            'deteriorate', 'deterioration', 'worse', 'worsening', 'problem', 'collapsing'
        ]
        
        pos_score = sum(1 for word in positive_words if word in sentence_lower)
        neg_score = sum(1 for word in negative_words if word in sentence_lower)
        
        if pos_score > neg_score:
            sentiment_score = min(0.6 + (pos_score * 0.05), 0.95)
            label = 'positive'
        elif neg_score > pos_score:
            sentiment_score = max(0.4 - (neg_score * 0.05), 0.05)
            label = 'negative'
        else:
            sentiment_score = 0.5
            label = 'neutral'
        
        if abs(pos_score - neg_score) >= 2:
            confidence = 0.8
        elif abs(pos_score - neg_score) >= 1:
            confidence = 0.7
        else:
            confidence = 0.5
        
        return {
            'sentence': sentence[:300],
            'sentiment_label': label,
            'sentiment_score': round(sentiment_score, 3),
            'confidence': round(confidence, 3),
            'source': 'rule-based-fallback'
        }
    
    def compare_with_evidence(self, current_text: str, previous_text: str) -> Dict:
        current = self.analyze_sentiment_with_evidence(current_text)
        previous = self.analyze_sentiment_with_evidence(previous_text)
        
        delta = current['sentiment_score'] - previous['sentiment_score']
        direction = 'more confident' if delta > 0.05 else ('less confident' if delta < -0.05 else 'unchanged')
        materiality = 'high' if abs(delta) > 0.15 else ('moderate' if abs(delta) > 0.08 else 'low')
        
        result = {
            'overall_delta': {
                'current': current['sentiment_score'],
                'previous': previous['sentiment_score'],
                'delta': round(delta, 3),
                'direction': direction,
                'materiality': materiality
            },
            'current_evidence': current.get('evidence', [])[:10],
            'previous_evidence': previous.get('evidence', [])[:10],
            'total_sentences_analyzed': {
                'current': current.get('sentence_count', 0),
                'previous': previous.get('sentence_count', 0)
            }
        }
        
        return result
