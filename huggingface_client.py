import requests
import os
import re
from typing import Dict, List

class HuggingFaceClient:
    def __init__(self):
        self.api_token = os.environ.get("HF_TOKEN", "")
        self.headers = {"Authorization": f"Bearer {self.api_token}" if self.api_token else "", "Content-Type": "application/json"}
        self.hf_api_failed = False
        self.fallback_reason = None
        
        # List of FinBERT models to try in order
        self.finbert_urls = [
            "https://api-inference.huggingface.co/models/ProsusAI/finbert",
            "https://api-inference.huggingface.co/models/finbert/finbert",
            "https://api-inference.huggingface.co/models/yiyanghkust/finbert-tone",
        ]
        self.current_url_index = 0
        
        if not self.api_token:
            print("WARNING: HF_TOKEN not set. Using rule-based fallback for sentiment analysis.")
            self.hf_api_failed = True
            self.fallback_reason = "HF_TOKEN not set in environment variables"
    
    def analyze_sentiment_with_evidence(self, text: str, max_sentences: int = 150) -> Dict:
        # Split into sentences - threshold lowered to 15 chars
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
        batch_size = 20
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            for sentence in batch:
                result = self._analyze_sentence(sentence)
                sentence_results.append(result)
        
        # Calculate overall sentiment from all analyzed sentences
        scores = [r['sentiment_score'] for r in sentence_results if r.get('sentiment_score') is not None]
        if scores:
            overall_score = sum(scores) / len(scores)
        else:
            overall_score = 0.5
            
        label = 'positive' if overall_score > 0.55 else ('negative' if overall_score < 0.45 else 'neutral')
        
        # Build response with error signal if HF API failed
        response = {
            'sentiment_label': label,
            'sentiment_score': round(overall_score, 3),
            'confidence': round(abs(overall_score - 0.5) * 2, 3),
            'evidence': sentence_results[:50],
            'sentence_count': len(sentence_results)
        }
        
        # Add error signal if fallback was used
        if self.hf_api_failed:
            response['warning'] = f"Sentiment analysis using rule-based fallback. Reason: {self.fallback_reason}. For full FinBERT accuracy, set HF_TOKEN environment variable."
        
        return response
    
    def _analyze_sentence(self, sentence: str) -> Dict:
        # If HF API already failed or token missing, use fallback immediately
        if self.hf_api_failed:
            return self._fallback_sentiment(sentence)
        
        # Try current FinBERT URL, if fails try next ones
        for i in range(self.current_url_index, len(self.finbert_urls)):
            url = self.finbert_urls[i]
            result = self._call_finbert_api(url, sentence)
            if result:
                self.current_url_index = i
                return result
        
        # All URLs failed, switch to fallback
        print(f"All FinBERT URLs failed. Falling back to rule-based sentiment.")
        self.hf_api_failed = True
        self.fallback_reason = "All FinBERT API endpoints returned errors"
        return self._fallback_sentiment(sentence)
    
    def _call_finbert_api(self, api_url: str, sentence: str) -> Dict:
        truncated_sentence = sentence[:500]
        
        try:
            response = requests.post(api_url, headers=self.headers, json={"inputs": truncated_sentence}, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if result and len(result) > 0:
                    label = result[0]['label'].lower()
                    score = result[0]['score']
                    
                    if label == 'positive':
                        sentiment_score = score
                    elif label == 'negative':
                        sentiment_score = 1 - score
                    else:
                        sentiment_score = 0.5
                    
                    print(f"FinBERT API ({api_url}) returned {label} with score {score}")
                    return {
                        'sentence': sentence[:300],
                        'sentiment_label': label,
                        'sentiment_score': round(sentiment_score, 3),
                        'confidence': round(score, 3),
                        'source': 'finbert-api'
                    }
            elif response.status_code == 401:
                print(f"HF API Error 401: Invalid or missing token. Trying next URL if available.")
                return None
            elif response.status_code == 404:
                print(f"HF API Error 404: Model not found at {api_url}. Trying next URL if available.")
                return None
            elif response.status_code == 503:
                print(f"HF API Error 503: Model loading or unavailable at {api_url}. Trying next URL if available.")
                return None
            else:
                print(f"HF API Error {response.status_code} at {api_url}: {response.text[:100]}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"Timeout at {api_url}")
            return None
        except Exception as e:
            print(f"Sentiment API error at {api_url}: {str(e)}")
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
        
        # Add warning if either analysis used fallback
        warnings = []
        if current.get('warning'):
            warnings.append(f"Current quarter: {current['warning']}")
        if previous.get('warning'):
            warnings.append(f"Previous quarter: {previous['warning']}")
        if warnings:
            result['warnings'] = warnings
        
        return result
