"""
Delta Detector for Earnings Call Transcripts
Detects added, removed, and modified passages between two transcripts.
"""

import difflib
import re
from typing import List, Dict, Tuple
from datetime import datetime

class TranscriptDeltaDetector:
    """Detect and classify changes between two earnings call transcripts."""
    
    def __init__(self):
        # Section markers to focus on
        self.section_markers = [
            'prepared remarks', 'opening remarks', 'forward-looking', 
            'financial results', 'business overview', 'outlook', 'guidance'
        ]
    
    def detect_changes(self, current_text: str, previous_text: str) -> Dict:
        """
        Detect changes between two transcript texts.
        Returns transparent diff with exact passages.
        """
        # Extract prepared remarks section from both
        current_remarks = self._extract_prepared_remarks(current_text)
        previous_remarks = self._extract_prepared_remarks(previous_text)
        
        # Split into paragraphs
        current_paragraphs = [p.strip() for p in current_remarks.split('\n\n') if len(p.strip()) > 50]
        previous_paragraphs = [p.strip() for p in previous_remarks.split('\n\n') if len(p.strip()) > 50]
        
        # Use difflib to find differences
        matcher = difflib.SequenceMatcher(None, previous_paragraphs, current_paragraphs)
        
        added = []
        removed = []
        modified = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'insert':
                for para in current_paragraphs[j1:j2]:
                    added.append({
                        'text': para[:500],  # Limit length
                        'length': len(para)
                    })
            elif tag == 'delete':
                for para in previous_paragraphs[i1:i2]:
                    removed.append({
                        'text': para[:500],
                        'length': len(para)
                    })
            elif tag == 'replace':
                for prev_para, curr_para in zip(previous_paragraphs[i1:i2], current_paragraphs[j1:j2]):
                    modified.append({
                        'previous': prev_para[:500],
                        'current': curr_para[:500],
                        'change_type': self._classify_change(prev_para, curr_para)
                    })
        
        # Classify changes by topic
        topic_changes = self._classify_by_topic(added + [m['current'] for m in modified])
        
        return {
            'summary': {
                'added_paragraphs': len(added),
                'removed_paragraphs': len(removed),
                'modified_paragraphs': len(modified),
                'has_material_changes': len(added) > 0 or len(modified) > 0
            },
            'added_content': added[:5],  # First 5 added passages
            'removed_content': removed[:5],  # First 5 removed passages
            'modified_content': modified[:5],  # First 5 modified passages
            'topic_changes': topic_changes,
            'detection_method': 'difflib.SequenceMatcher with paragraph-level comparison',
            'timestamp': datetime.now().isoformat()
        }
    
    def _extract_prepared_remarks(self, text: str) -> str:
        """Extract prepared remarks section from transcript."""
        text_lower = text.lower()
        
        # Find where prepared remarks start
        start_markers = ['prepared remarks', 'opening remarks', 'thank you operator']
        end_markers = ['question and answer', 'q&a', 'operator', 'thank you for joining']
        
        start_idx = None
        for marker in start_markers:
            pos = text_lower.find(marker)
            if pos != -1:
                start_idx = pos
                break
        
        if start_idx is None:
            # No clear marker, return first 5000 chars
            return text[:5000]
        
        end_idx = None
        for marker in end_markers:
            pos = text_lower.find(marker, start_idx)
            if pos != -1:
                end_idx = pos
                break
        
        if end_idx is None:
            end_idx = start_idx + 5000
        
        return text[start_idx:end_idx]
    
    def _classify_change(self, prev: str, curr: str) -> str:
        """Classify the type of change."""
        # Simple heuristic based on length change and keyword changes
        prev_len = len(prev)
        curr_len = len(curr)
        
        if abs(curr_len - prev_len) / max(prev_len, 1) > 0.5:
            return 'substantial_rewrite'
        elif 'increase' in curr.lower() and 'decrease' in prev.lower():
            return 'reversal'
        elif any(word in curr.lower() for word in ['uncertain', 'challenge', 'risk']) and \
             any(word in prev.lower() for word in ['confident', 'strong', 'momentum']):
            return 'tone_worsened'
        elif any(word in curr.lower() for word in ['confident', 'strong', 'momentum']) and \
             any(word in prev.lower() for word in ['uncertain', 'challenge', 'risk']):
            return 'tone_improved'
        else:
            return 'minor_wording'
    
    def _classify_by_topic(self, texts: List[str]) -> Dict:
        """Classify changes by topic using keyword matching."""
        topics = {
            'revenue': ['revenue', 'sales', 'top line', 'billings'],
            'guidance': ['guidance', 'outlook', 'expect', 'forecast', 'project'],
            'margin': ['margin', 'gross margin', 'profitability', 'profit'],
            'demand': ['demand', 'customer', 'orders', 'pipeline'],
            'competitive': ['competitive', 'competition', 'market share'],
            'cost': ['cost', 'expense', 'spending', 'inflation']
        }
        
        topic_counts = {topic: 0 for topic in topics}
        topic_examples = {topic: [] for topic in topics}
        
        for text in texts:
            text_lower = text.lower()
            for topic, keywords in topics.items():
                if any(keyword in text_lower for keyword in keywords):
                    topic_counts[topic] += 1
                    if len(topic_examples[topic]) < 2:
                        topic_examples[topic].append(text[:200])
        
        return {
            topic: {
                'change_count': count,
                'examples': topic_examples[topic]
            }
            for topic, count in topic_counts.items() if count > 0
        }
