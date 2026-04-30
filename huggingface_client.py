import requests
import os
import re
from typing import Dict, List

class HuggingFaceClient:
    def __init__(self):
        self.api_token = os.environ.get("HF_TOKEN", "")
        self.headers = {"Authorization": f"Bearer {self.api_token}" if self.api_token else "", "Content-Type": "application/json"}
    
    def analyze_sentiment_with_evidence(self, text: str, max_sentences: int = 150) -> Dict:
        # Split into sentences and filter out very short ones
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 40][:max_sentences]
        
        if not sentences:
            return {'sentiment_label': 'neutral', 'sentiment_score': 0.5, 'confidence': 0.5, 'evidence': [], 'sentence_count': 0}
        
        # Process sentences in batches to avoid overwhelming the API
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
            
        label = 'positive' if overall_score > 0.6 else ('negative' if overall_score < 0.4 else 'neutral')
        
        # Return evidence for all sentences (or top N for response size)
        return {
            'sentiment_label': label,
            'sentiment_score': round(overall_score, 3),
            'confidence': round(abs(overall_score - 0.5) * 2, 3),
            'evidence': sentence_results[:50],  # Limit evidence in response to 50 sentences
            'sentence_count': len(sentence_results)
        }
    
    def _analyze_sentence(self, sentence: str) -> Dict:
        # Using FinBERT (finance-optimized model)
        api_url = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
        
        # Truncate sentence to 500 chars to avoid API limits
        truncated_sentence = sentence[:500]
        
        try:
            response = requests.post(api_url, headers=self.headers, json={"inputs": truncated_sentence}, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                if result and len(result) > 0:
                    label = result[0]['label'].lower()
                    score = result[0]['score']
                    
                    # Convert to sentiment_score (0-1 scale where 0.5 is neutral)
                    if label == 'positive':
                        sentiment_score = score
                    elif label == 'negative':
                        sentiment_score = 1 - score
                    else:  # neutral
                        sentiment_score = 0.5
                    
                    return {
                        'sentence': sentence[:300],
                        'sentiment_label': label,
                        'sentiment_score': round(sentiment_score, 3),
                        'confidence': round(score, 3)
                    }
            
            # Fallback for API errors
            return {
                'sentence': sentence[:200],
                'sentiment_label': 'neutral',
                'sentiment_score': 0.5,
                'confidence': 0.5
            }
            
        except requests.exceptions.Timeout:
            print(f"Timeout analyzing sentence, using neutral")
            return {
                'sentence': sentence[:200],
                'sentiment_label': 'neutral',
                'sentiment_score': 0.5,
                'confidence': 0.5
            }
        except Exception as e:
            print(f"Sentiment API error: {str(e)}")
            return {
                'sentence': sentence[:200],
                'sentiment_label': 'neutral',
                'sentiment_score': 0.5,
                'confidence': 0.5
            }
    
    def compare_with_evidence(self, current_text: str, previous_text: str) -> Dict:
        current = self.analyze_sentiment_with_evidence(current_text)
        previous = self.analyze_sentiment_with_evidence(previous_text)
        
        delta = current['sentiment_score'] - previous['sentiment_score']
        direction = 'more confident' if delta > 0.05 else ('less confident' if delta < -0.05 else 'unchanged')
        materiality = 'high' if abs(delta) > 0.15 else ('moderate' if abs(delta) > 0.08 else 'low')
        
        # Extract key financial sentences for better evidence
        current_key_sentences = self._extract_key_sentences(current.get('evidence', []))
        previous_key_sentences = self._extract_key_sentences(previous.get('evidence', []))
        
        return {
            'overall_delta': {
                'current': current['sentiment_score'],
                'previous': previous['sentiment_score'],
                'delta': round(delta, 3),
                'direction': direction,
                'materiality': materiality
            },
            'current_evidence': current_key_sentences[:10],
            'previous_evidence': previous_key_sentences[:10],
            'total_sentences_analyzed': {
                'current': current['sentence_count'],
                'previous': previous['sentence_count']
            }
        }
    
    def _extract_key_sentences(self, evidence_list: List[Dict]) -> List[Dict]:
        """Extract sentences that are likely to contain financial guidance or metrics."""
        if not evidence_list:
            return []
        
        # Keywords that indicate important financial information
        key_terms = [
            'guidance', 'outlook', 'expect', 'forecast', 'revenue', 'margin', 
            'growth', 'quarter', 'year', 'demand', 'market', 'segment',
            'gross margin', 'operating margin', 'eps', 'earnings', 'sales',
            'capital', 'investment', 'ramp', 'production', 'supply', 'demand'
        ]
        
        # Score each sentence by relevance
        scored_sentences = []
        for evidence in evidence_list:
            sentence = evidence.get('sentence', '').lower()
            score = sum(1 for term in key_terms if term in sentence)
            scored_sentences.append((score, evidence))
        
        # Sort by relevance (highest first) and return
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        return [evidence for score, evidence in scored_sentences if score > 0]
