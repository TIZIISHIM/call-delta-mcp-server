import os
import requests
from typing import Dict
from datetime import datetime

class TranscriptFetcher:
    def __init__(self):
        self.api_key = os.environ.get("FMP_API_KEY", "")
        self.base_url = "https://financialmodelingprep.com/api/v3"
    
    def fetch_transcript(self, ticker: str, year: int, quarter: int) -> Dict:
        # Debug: Check if key is present
        print(f"DEBUG: FMP_API_KEY is {'SET' if self.api_key else 'NOT SET'}")
        
        if not self.api_key:
            return {
                'status': 'error',
                'error_code': 'MISSING_API_KEY',
                'error_message': 'FMP_API_KEY not set in environment variables',
                'timestamp': datetime.now().isoformat()
            }
        
        quarter_map = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'}
        quarter_str = quarter_map.get(quarter, f'Q{quarter}')
        
        # Updated endpoint format based on FMP documentation
        url = f"{self.base_url}/earning_call_transcript"
        params = {
            'symbol': ticker,
            'quarter': quarter_str,
            'year': year,
            'apikey': self.api_key
        }
        
        print(f"DEBUG: Fetching from URL: {url}")
        print(f"DEBUG: Params: symbol={ticker}, quarter={quarter_str}, year={year}")
        
        try:
            response = requests.get(url, params=params, timeout=15)
            print(f"DEBUG: Response status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    content = data[0].get('content', '')
                    if content and len(content) > 200:
                        return {
                            'status': 'success',
                            'source': 'Financial Modeling Prep',
                            'content': content,
                            'source_used': 'FMP API',
                            'timestamp': datetime.now().isoformat()
                        }
                    else:
                        return {
                            'status': 'error',
                            'error_code': 'EMPTY_CONTENT',
                            'error_message': f'Transcript for {ticker} {quarter_str} {year} returned empty content',
                            'timestamp': datetime.now().isoformat()
                        }
                else:
                    return {
                        'status': 'error',
                        'error_code': 'NO_DATA',
                        'error_message': f'No data found for {ticker} {quarter_str} {year}',
                        'timestamp': datetime.now().isoformat()
                    }
            elif response.status_code == 401:
                return {
                    'status': 'error',
                    'error_code': 'INVALID_API_KEY',
                    'error_message': 'FMP API key is invalid or expired. Please check your key at financialmodelingprep.com',
                    'timestamp': datetime.now().isoformat()
                }
            elif response.status_code == 403:
                return {
                    'status': 'error',
                    'error_code': 'FORBIDDEN',
                    'error_message': 'FMP API key does not have access to earnings transcripts. Free tier may not include this endpoint.',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'status': 'error',
                    'error_code': 'API_ERROR',
                    'error_message': f'FMP API returned status {response.status_code}: {response.text[:100]}',
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            print(f"DEBUG: Exception: {str(e)}")
            return {
                'status': 'error',
                'error_code': 'UNKNOWN',
                'error_message': str(e)[:100],
                'timestamp': datetime.now().isoformat()
            }
