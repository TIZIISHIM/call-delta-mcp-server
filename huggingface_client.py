import requests
import os
import re
from typing import Dict, List

class HuggingFaceClient:
    def __init__(self):
        self.api_token = os.environ.get("HF_TOKEN", "")
        self.headers = {"Authorization": f"Bearer {self.api_token}"} if self.api_token else {}
        
        # List of FinBERT models to try in order (smaller/efficient first)
        self.model_urls = [
            "https://api-inference.huggingface.co/models/likith123/SSAF-FinBert",  # Smaller, efficient
            "https://api-inference.huggingface.co/models/ProsusAI/finbert",         # Original FinBERT
            "https://api-inference.huggingface.co/models/yiyanghkust/finbert-tone", # Tone-specific
            "https://api-inference.huggingface.co/models/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",  # Financial news model
        ]
        self.current_model_index = 0
        self.api_failed = False
        
        if not self.api_token:
            print("WARNING: HF_TOKEN not set. Using rule-based fallback for sentiment analysis.")
            self.api_failed = True
    
    def analyze_sentiment_with_evidence(self, text: str, max_sentences: int = 150) -> Dict:
        # Split into sentences - threshold 15 chars (Alex's requirement)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15][:max_sentences]
        
        if not sentences:
            return {
                'sentiment_label': 'neutral',
                'sentiment_score': 0.5,
                'confidence': 0.0,
                'evidence': [],
                'sentence_count': 0,
                'error': 'No valid sentences found in text (minimum 15 characters)'
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
        
        response = {
            'sentiment_label': label,
            'sentiment_score': round(overall_score, 3),
            'confidence': round(abs(overall_score - 0.5) * 2, 3),
            'evidence': sentence_results[:50],
            'sentence_count': len(sentence_results)
        }
        
        if self.api_failed:
            response['warning'] = 'Using rule-based fallback. Set HF_TOKEN for API access.'
        
        return response
    
    def _analyze_sentence(self, sentence: str) -> Dict:
        # If API already failed or no token, use fallback
        if self.api_failed or not self.api_token:
            return self._fallback_sentiment(sentence)
        
        # Try models from current index onward
        for i in range(self.current_model_index, len(self.model_urls)):
            url = self.model_urls[i]
            result = self._call_hf_api(url, sentence)
            if result:
                self.current_model_index = i
                return result
        
        # All models failed, switch to fallback
        print("All HF API models failed. Switching to rule-based fallback.")
        self.api_failed = True
        return self._fallback_sentiment(sentence)
    
    def _call_hf_api(self, api_url: str, sentence: str) -> Dict:
        truncated_sentence = sentence[:500]
        
        try:
            response = requests.post(api_url, headers=self.headers, json={"inputs": truncated_sentence}, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if result and len(result) > 0:
                    # Handle different response formats
                    if isinstance(result, list) and len(result) > 0:
                        item = result[0]
                        label = item.get('label', '').lower()
                        score = item.get('score', 0.5)
                    elif isinstance(result, dict):
                        label = result.get('label', '').lower()
                        score = result.get('score', 0.5)
                    else:
                        return None
                    
                    # Standardize label
                    if label in ['positive', 'POSITIVE']:
                        sentiment_score = score
                        label = 'positive'
                    elif label in ['negative', 'NEGATIVE']:
                        sentiment_score = 1 - score
                        label = 'negative'
                    else:
                        sentiment_score = 0.5
                        label = 'neutral'
                    
                    print(f"HF API ({api_url.split('/')[-1]}) returned {label} with score {score}")
                    return {
                        'sentence': sentence[:300],
                        'sentiment_label': label,
                        'sentiment_score': round(sentiment_score, 3),
                        'confidence': round(score, 3),
                        'source': f'finbert-api'
                    }
            
            elif response.status_code == 401:
                print(f"HF API Error 401: Invalid token. Check HF_TOKEN environment variable.")
                self.api_failed = True
                return None
            elif response.status_code == 503:
                print(f"HF API Error 503: Model loading at {api_url}. Trying next model.")
                return None
            else:
                print(f"HF API Error {response.status_code} at {api_url}: {response.text[:100]}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"Timeout at {api_url}")
            return None
        except Exception as e:
            print(f"HF API exception at {api_url}: {str(e)}")
            return None
    
    def _fallback_sentiment(self, sentence: str) -> Dict:
        """Rule-based fallback for when HF API is unavailable."""
        sentence_lower = sentence.lower()
        
        # Positive financial keywords
        positive_words = [
            'growth', 'grew', 'increase', 'increased', 'rising', 'up', 'higher',
            'record', 'strong', 'confidence', 'confident', 'optimistic', 'beat',
            'exceed', 'outperform', 'raise', 'raised', 'guidance', 'momentum',
            'accelerate', 'accelerated', 'expansion', 'expand', 'profit', 'profitable',
            'outlook', 'improve', 'improved', 'improvement', 'better', 'best'
        ]
        
        # Negative financial keywords
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
        
        # Boost confidence for clear signals
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
        
        # Add warning if fallback was used
        if current.get('warning'):
            result['warning'] = current['warning']
        
        return result
