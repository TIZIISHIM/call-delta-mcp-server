import requests
import os
import re
from typing import Dict, List

class HuggingFaceClient:
    def __init__(self):
        self.api_token = os.environ.get("HF_TOKEN", "")
        self.headers = {"Authorization": f"Bearer {self.api_token}" if self.api_token else "", "Content-Type": "application/json"}
    
    def analyze_sentiment_with_evidence(self, text: str, max_sentences: int = 10) -> Dict:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 30][:max_sentences]
        if not sentences:
            return {'sentiment_label': 'neutral', 'sentiment_score': 0.5, 'confidence': 0.5, 'evidence': []}
        
        sentence_results = []
        for sentence in sentences:
            result = self._analyze_sentence(sentence)
            sentence_results.append(result)
        
        scores = [r['sentiment_score'] for r in sentence_results if r.get('sentiment_score') is not None]
        overall_score = sum(scores) / len(scores) if scores else 0.5
        label = 'positive' if overall_score > 0.6 else ('negative' if overall_score < 0.4 else 'neutral')
        
        return {'sentiment_label': label, 'sentiment_score': round(overall_score, 3), 'confidence': round(abs(overall_score - 0.5) * 2, 3), 'evidence': sentence_results[:5]}
    
    def _analyze_sentence(self, sentence: str) -> Dict:
        api_url = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
        try:
            response = requests.post(api_url, headers=self.headers, json={"inputs": sentence[:500]}, timeout=15)
            if response.status_code == 200:
                result = response.json()
                if result and len(result) > 0 and 'label' in result[0]:
                    label = result[0]['label']
                    score = result[0]['score']
                    sentiment_score = score if label == 'POSITIVE' else 1 - score
                    return {'sentence': sentence[:300], 'sentiment_label': 'positive' if label == 'POSITIVE' else 'negative', 'sentiment_score': round(sentiment_score, 3), 'confidence': round(score, 3)}
            return {'sentence': sentence[:200], 'sentiment_label': 'neutral', 'sentiment_score': 0.5, 'confidence': 0.5}
        except:
            return {'sentence': sentence[:200], 'sentiment_label': 'neutral', 'sentiment_score': 0.5, 'confidence': 0.5}
    
    def compare_with_evidence(self, current_text: str, previous_text: str) -> Dict:
        current = self.analyze_sentiment_with_evidence(current_text)
        previous = self.analyze_sentiment_with_evidence(previous_text)
        delta = current['sentiment_score'] - previous['sentiment_score']
        direction = 'more confident' if delta > 0.05 else ('less confident' if delta < -0.05 else 'unchanged')
        materiality = 'high' if abs(delta) > 0.15 else ('moderate' if abs(delta) > 0.08 else 'low')
        return {'overall_delta': {'current': current['sentiment_score'], 'previous': previous['sentiment_score'], 'delta': round(delta, 3), 'direction': direction, 'materiality': materiality}, 'current_evidence': current.get('evidence', [])[:3], 'previous_evidence': previous.get('evidence', [])[:3]}
