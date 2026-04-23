

import requests
import os
import re
from typing import Dict, List

class HuggingFaceClient:
    def __init__(self):
        self.api_token = os.environ.get("HF_TOKEN", "")
        self.headers = {
            "Authorization": f"Bearer {self.api_token}" if self.api_token else "",
            "Content-Type": "application/json"
        }
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of full text (legacy method)."""
        return self.analyze_sentiment_with_evidence(text)
    
    def analyze_sentiment_with_evidence(self, text: str, max_sentences: int = 10) -> Dict:
        """
        Analyze sentiment and return sentence-level evidence.
        This fulfills the transparent materiality requirement.
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 30][:max_sentences]
        
        if not sentences:
            return {
                'sentiment_label': 'neutral',
                'sentiment_score': 0.5,
                'confidence': 0.5,
                'evidence': [],
                'note': 'No substantial sentences found for analysis'
            }
        
        # Analyze each sentence
        sentence_results = []
        for sentence in sentences:
            result = self._analyze_single_sentence(sentence)
            sentence_results.append(result)
        
        # Calculate overall sentiment
        scores = [r['sentiment_score'] for r in sentence_results if r.get('sentiment_score') is not None]
        if scores:
            overall_score = sum(scores) / len(scores)
        else:
            overall_score = 0.5
        
        # Determine overall label
        if overall_score > 0.6:
            overall_label = 'positive'
        elif overall_score < 0.4:
            overall_label = 'negative'
        else:
            overall_label = 'neutral'
        
        return {
            'sentiment_label': overall_label,
            'sentiment_score': round(overall_score, 3),
            'confidence': round(abs(overall_score - 0.5) * 2, 3),  # Proxy confidence
            'evidence': sentence_results,
            'sentence_count': len(sentence_results),
            'model_used': 'distilbert-base-uncased-finetuned-sst-2-english',
            'api': 'HuggingFace Inference (free)',
            'transparency_note': 'Each sentence was analyzed individually. See evidence array for exact sentences and their scores.'
        }
    
    def _analyze_single_sentence(self, sentence: str) -> Dict:
        """Analyze a single sentence and return evidence."""
        api_url = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
        
        try:
            response = requests.post(
                api_url,
                headers=self.headers,
                json={"inputs": sentence[:500]},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    scores = {item['label']: item['score'] for item in result[0]}
                    positive_score = scores.get('POSITIVE', 0.5)
                    
                    return {
                        'sentence': sentence[:300],
                        'sentiment_label': 'positive' if positive_score > 0.6 else ('negative' if positive_score < 0.4 else 'neutral'),
                        'sentiment_score': round(positive_score, 3),
                        'confidence': round(max(positive_score, 1 - positive_score), 3)
                    }
            
            return {
                'sentence': sentence[:200],
                'sentiment_label': 'neutral',
                'sentiment_score': 0.5,
                'confidence': 0.5,
                'error': 'API returned unexpected format'
            }
            
        except Exception as e:
            return {
                'sentence': sentence[:200],
                'sentiment_label': 'neutral',
                'sentiment_score': 0.5,
                'confidence': 0.5,
                'error': str(e)[:50]
            }
    
    def compare_with_evidence(self, current_text: str, previous_text: str) -> Dict:
        """Compare two transcripts with sentence-level evidence."""
        current_analysis = self.analyze_sentiment_with_evidence(current_text)
        previous_analysis = self.analyze_sentiment_with_evidence(previous_text)
        
        current_score = current_analysis['sentiment_score']
        previous_score = previous_analysis['sentiment_score']
        delta = current_score - previous_score
        
        # Find most changed sentence
        most_changed = None
        if current_analysis.get('evidence') and previous_analysis.get('evidence'):
            # Simple heuristic: compare first sentence of each
            curr_first = current_analysis['evidence'][0] if current_analysis['evidence'] else None
            prev_first = previous_analysis['evidence'][0] if previous_analysis['evidence'] else None
            
            if curr_first and prev_first:
                most_changed = {
                    'current_sentence': curr_first.get('sentence', '')[:200],
                    'current_score': curr_first.get('sentiment_score', 0.5),
                    'previous_sentence': prev_first.get('sentence', '')[:200],
                    'previous_score': prev_first.get('sentiment_score', 0.5),
                    'delta': round(curr_first.get('sentiment_score', 0.5) - prev_first.get('sentiment_score', 0.5), 3)
                }
        
        return {
            'overall_delta': {
                'current': current_score,
                'previous': previous_score,
                'delta': round(delta, 3),
                'direction': 'more confident' if delta > 0.05 else ('less confident' if delta < -0.05 else 'unchanged'),
                'materiality': 'high' if abs(delta) > 0.15 else ('moderate' if abs(delta) > 0.08 else 'low')
            },
            'current_evidence': current_analysis.get('evidence', [])[:5],
            'previous_evidence': previous_analysis.get('evidence', [])[:5],
            'most_changed_sentence': most_changed,
            'methodology': {
                'model': 'distilbert-base-uncased-finetuned-sst-2-english',
                'api': 'HuggingFace Inference (free)',
                'transparency': 'Each sentence analyzed individually. See evidence arrays for exact sentences and scores.'
            }
        }
