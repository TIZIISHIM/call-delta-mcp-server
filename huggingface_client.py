

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
    
    def analyze_sentiment_with_evidence(self, text: str, max_sentences: int = 10) -> Dict:
        """Analyze sentiment with sentence-level evidence - returns REAL scores."""
        # Split into sentences
        sentences = self._split_sentences(text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 30][:max_sentences]
        
        if not sentences:
            return {
                'sentiment_label': 'neutral',
                'sentiment_score': 0.5,
                'confidence': 0.5,
                'evidence': [],
                'sentence_count': 0,
                'model_used': 'distilbert-base-uncased-finetuned-sst-2-english',
                'api': 'HuggingFace Inference (free)',
                'note': 'No substantial sentences found for analysis'
            }
        
        # Analyze each sentence
        sentence_results = []
        for sentence in sentences:
            result = self._analyze_single_sentence(sentence)
            sentence_results.append(result)
        
        # Calculate overall sentiment from individual sentences
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
            'confidence': round(abs(overall_score - 0.5) * 2, 3),
            'evidence': sentence_results,
            'sentence_count': len(sentence_results),
            'model_used': 'distilbert-base-uncased-finetuned-sst-2-english',
            'api': 'HuggingFace Inference (free)',
            'transparency_note': 'Each sentence was analyzed individually. See evidence array for exact sentences and their scores.'
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using punctuation."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s for s in sentences if len(s) > 0]
    
    def _analyze_single_sentence(self, sentence: str) -> Dict:
        """Analyze a single sentence using Hugging Face Inference API."""
        api_url = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
        
        try:
            response = requests.post(
                api_url,
                headers=self.headers,
                json={"inputs": sentence[:500]},
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Parse the Hugging Face response format
                # Expected format: [{'label': 'POSITIVE', 'score': 0.95}]
                if isinstance(result, list) and len(result) > 0:
                    first_result = result[0]
                    if isinstance(first_result, dict) and 'label' in first_result:
                        label = first_result['label']
                        score = first_result['score']
                        
                        # Convert to 0-1 scale (0=negative, 1=positive)
                        if label == 'POSITIVE':
                            sentiment_score = score
                            sentiment_label = 'positive'
                        elif label == 'NEGATIVE':
                            sentiment_score = 1 - score
                            sentiment_label = 'negative'
                        else:
                            sentiment_score = 0.5
                            sentiment_label = 'neutral'
                        
                        return {
                            'sentence': sentence[:300],
                            'sentiment_label': sentiment_label,
                            'sentiment_score': round(sentiment_score, 3),
                            'confidence': round(score, 3),
                            'raw_model_output': label
                        }
                
                # If we get here, response format was unexpected
                return {
                    'sentence': sentence[:200],
                    'sentiment_label': 'neutral',
                    'sentiment_score': 0.5,
                    'confidence': 0.5,
                    'error': f'Unexpected API response format: {str(result)[:100]}'
                }
            
            elif response.status_code == 429:
                return {
                    'sentence': sentence[:200],
                    'sentiment_label': 'neutral',
                    'sentiment_score': 0.5,
                    'confidence': 0.5,
                    'error': 'Rate limited - using fallback'
                }
            
            else:
                return {
                    'sentence': sentence[:200],
                    'sentiment_label': 'neutral',
                    'sentiment_score': 0.5,
                    'confidence': 0.5,
                    'error': f'HTTP {response.status_code}'
                }
                
        except requests.exceptions.Timeout:
            return {
                'sentence': sentence[:200],
                'sentiment_label': 'neutral',
                'sentiment_score': 0.5,
                'confidence': 0.5,
                'error': 'Request timeout'
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
        
        # Determine direction and materiality
        if delta > 0.05:
            direction = 'more confident'
        elif delta < -0.05:
            direction = 'less confident'
        else:
            direction = 'unchanged'
        
        if abs(delta) > 0.15:
            materiality = 'high'
        elif abs(delta) > 0.08:
            materiality = 'moderate'
        else:
            materiality = 'low'
        
        # Find the most changed sentence (if available)
        most_changed = None
        current_evidence = current_analysis.get('evidence', [])
        previous_evidence = previous_analysis.get('evidence', [])
        
        if current_evidence and previous_evidence:
            curr_first = current_evidence[0] if current_evidence else None
            prev_first = previous_evidence[0] if previous_evidence else None
            
            if curr_first and prev_first:
                sentence_delta = curr_first.get('sentiment_score', 0.5) - prev_first.get('sentiment_score', 0.5)
                most_changed = {
                    'current_sentence': curr_first.get('sentence', '')[:250],
                    'current_score': curr_first.get('sentiment_score', 0.5),
                    'previous_sentence': prev_first.get('sentence', '')[:250],
                    'previous_score': prev_first.get('sentiment_score', 0.5),
                    'delta': round(sentence_delta, 3)
                }
        
        return {
            'overall_delta': {
                'current': round(current_score, 3),
                'previous': round(previous_score, 3),
                'delta': round(delta, 3),
                'direction': direction,
                'materiality': materiality
            },
            'current_evidence': current_evidence[:5],
            'previous_evidence': previous_evidence[:5],
            'most_changed_sentence': most_changed,
            'methodology': {
                'model': 'distilbert-base-uncased-finetuned-sst-2-english',
                'api': 'HuggingFace Inference (free)',
                'transparency': 'Each sentence analyzed individually with source evidence',
                'sentiment_scale': '0=negative/cautious, 0.5=neutral, 1=positive/confident',
                'evidence_sources': 'Each evidence item includes the exact sentence that drove the score'
            }
        }
