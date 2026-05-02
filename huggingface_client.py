import requests
import os
import re
import json
from typing import Dict, List

class HuggingFaceClient:
    def __init__(self):
        # API keys and endpoints
        self.gradio_url = os.environ.get("GRADIO_SPACE_URL", "")
        self.replicate_token = os.environ.get("REPLICATE_API_TOKEN", "")
        self.hf_token = os.environ.get("HF_TOKEN", "")
        
        # Track which source is currently working
        self.current_source = self._determine_initial_source()
        
        print(f"Initialized HuggingFaceClient with source: {self.current_source}")
    
    def _determine_initial_source(self):
        """Determine which source to try first"""
        if self.gradio_url:
            return "gradio"
        elif self.replicate_token:
            return "replicate"
        elif self.hf_token:
            return "huggingface"
        else:
            return "fallback"
    
    def analyze_sentiment_with_evidence(self, text: str, max_sentences: int = 150) -> Dict:
        # Split into sentences - threshold 15 chars
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15][:max_sentences]
        
        if not sentences:
            return {
                'sentiment_label': 'neutral',
                'sentiment_score': 0.5,
                'confidence': 0.0,
                'evidence': [],
                'sentence_count': 0,
                'error': 'No valid sentences found (minimum 15 characters)'
            }
        
        # Process sentences
        sentence_results = []
        for sentence in sentences:
            result = self._analyze_sentence(sentence)
            sentence_results.append(result)
        
        # Calculate overall sentiment
        scores = [r['sentiment_score'] for r in sentence_results if r.get('sentiment_score') is not None]
        overall_score = sum(scores) / len(scores) if scores else 0.5
        label = 'positive' if overall_score > 0.55 else ('negative' if overall_score < 0.45 else 'neutral')
        
        response = {
            'sentiment_label': label,
            'sentiment_score': round(overall_score, 3),
            'confidence': round(abs(overall_score - 0.5) * 2, 3),
            'evidence': sentence_results[:50],
            'sentence_count': len(sentence_results),
            'source': self.current_source
        }
        
        if self.current_source == "fallback":
            response['warning'] = 'Using rule-based fallback. Set GRADIO_SPACE_URL, REPLICATE_API_TOKEN, or HF_TOKEN for better accuracy.'
        
        return response
    
    def _analyze_sentence(self, sentence: str) -> Dict:
        # Try sources in priority order
        if self.current_source == "gradio":
            result = self._call_gradio(sentence)
            if result:
                return result
            print("Gradio failed, switching to Replicate")
            self.current_source = "replicate"
        
        if self.current_source == "replicate":
            result = self._call_replicate(sentence)
            if result:
                return result
            print("Replicate failed, switching to Hugging Face")
            self.current_source = "huggingface"
        
        if self.current_source == "huggingface":
            result = self._call_huggingface(sentence)
            if result:
                return result
            print("Hugging Face failed, switching to fallback")
            self.current_source = "fallback"
        
        # Final fallback
        return self._fallback_sentiment(sentence)
    
    def _call_gradio(self, sentence: str) -> Dict:
    """Call Gradio Space's API endpoint directly"""
    if not self.gradio_url:
        return None
    
    try:
        # Use the dedicated API endpoint we created
        endpoint = f"{self.gradio_url}/api/sentiment"
        
        response = requests.post(
            endpoint,
            json={"text": sentence[:500]},
            timeout=15,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Gradio API - Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Parse the response format: {"data": ["{\"label\":\"positive\",...}"]}
            if result and 'data' in result and len(result['data']) > 0:
                result_data = result['data'][0]
                
                # The result is a JSON string, parse it
                if isinstance(result_data, str):
                    sentiment_data = json.loads(result_data)
                else:
                    sentiment_data = result_data
                
                # Extract label and score
                label = sentiment_data.get('label', 'neutral').lower()
                score = sentiment_data.get('score', 0.5)
                sentiment_score = sentiment_data.get('sentiment_score', 0.5)
                
                return {
                    'sentence': sentence[:300],
                    'sentiment_label': label,
                    'sentiment_score': round(sentiment_score, 3),
                    'confidence': round(score, 3),
                    'source': 'gradio-space',
                    'model_used': sentiment_data.get('model_used', 'unknown')
                }
                
    except requests.exceptions.Timeout:
        print(f"Gradio API timeout for {self.gradio_url}")
    except Exception as e:
        print(f"Gradio API error: {str(e)}")
    
    return None
    
    def _call_replicate(self, sentence: str) -> Dict:
        if not self.replicate_token:
            return None
        
        try:
            import replicate
            
            # Set the token
            replicate.client.api_token = self.replicate_token
            
            output = replicate.run(
                "nateraw/prosusai-finbert:latest",
                input={"text": sentence[:500]}
            )
            
            if output:
                label = output.get('label', 'neutral').lower()
                score = output.get('score', 0.5)
                sentiment_score = score if label == 'positive' else (1 - score if label == 'negative' else 0.5)
                
                return {
                    'sentence': sentence[:300],
                    'sentiment_label': label,
                    'sentiment_score': round(sentiment_score, 3),
                    'confidence': round(score, 3),
                    'source': 'replicate-api'
                }
        except ImportError:
            print("Replicate library not installed. Run: pip install replicate")
        except Exception as e:
            print(f"Replicate error: {str(e)}")
        
        return None
    
    def _call_huggingface(self, sentence: str) -> Dict:
        if not self.hf_token:
            return None
        
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        
        # List of models to try
        models = [
            "https://api-inference.huggingface.co/models/ProsusAI/finbert",
            "https://api-inference.huggingface.co/models/ahmedrachid/FinancialBERT-Sentiment-Analysis",
            "https://api-inference.huggingface.co/models/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
        ]
        
        for model_url in models:
            try:
                response = requests.post(
                    model_url,
                    headers=headers,
                    json={"inputs": sentence[:500]},
                    timeout=15
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result and len(result) > 0:
                        item = result[0] if isinstance(result, list) else result
                        label = item.get('label', 'neutral').lower()
                        score = item.get('score', 0.5)
                        sentiment_score = score if label == 'positive' else (1 - score if label == 'negative' else 0.5)
                        
                        return {
                            'sentence': sentence[:300],
                            'sentiment_label': label,
                            'sentiment_score': round(sentiment_score, 3),
                            'confidence': round(score, 3),
                            'source': 'hf-api'
                        }
            except Exception as e:
                print(f"HF model {model_url} error: {str(e)}")
                continue
        
        return None
    
    def _fallback_sentiment(self, sentence: str) -> Dict:
        """Rule-based fallback for when all APIs fail."""
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
            'deteriorate', 'deterioration', 'worse', 'worsening', 'problem', 'collapsing', 'poor'
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
        
        confidence = 0.8 if abs(pos_score - neg_score) >= 2 else (0.7 if abs(pos_score - neg_score) >= 1 else 0.5)
        
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
        
        if current.get('warning'):
            result['warning'] = current['warning']
        
        return result
