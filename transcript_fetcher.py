import os
import requests
from typing import Dict
from datetime import datetime

class TranscriptFetcher:
    def __init__(self):
        self.api_key = os.environ.get("FMP_API_KEY", "")
        self.base_url = "https://financialmodelingprep.com/api/v3"
    
    def fetch_transcript(self, ticker: str, year: int, quarter: int) -> Dict:
        if not self.api_key:
            return {'status': 'error', 'error_code': 'MISSING_API_KEY', 'error_message': 'FMP_API_KEY not set'}
        
        quarter_map = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'}
        quarter_str = quarter_map.get(quarter, f'Q{quarter}')
        quarter_param = f"{quarter_str} {year}"
        
        url = f"{self.base_url}/earning_call_transcript/{ticker}"
        params = {'quarter': quarter_param, 'apikey': self.api_key}
        
        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    content = data[0].get('content', '')
                    if content and len(content) > 200:
                        return {'status': 'success', 'source': 'Financial Modeling Prep', 'content': content, 'source_used': 'FMP API', 'timestamp': datetime.now().isoformat()}
            return {'status': 'error', 'error_code': 'NOT_FOUND', 'error_message': f"No transcript for {ticker} {quarter_str} {year}"}
        except Exception as e:
            return {'status': 'error', 'error_code': 'UNKNOWN', 'error_message': str(e)[:100]}
