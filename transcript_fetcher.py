import os
import requests
from typing import Dict
from datetime import datetime

class TranscriptFetcher:
    def __init__(self):
        self.fmp_api_key = os.environ.get("FMP_API_KEY", "")
        self.apify_token = os.environ.get("APIFY_TOKEN", "")
        self.finnhub_key = os.environ.get("FINNHUB_KEY", "")
        self.base_url = "https://financialmodelingprep.com/api/v3"
        
        # List of Apify actors to try in order
        self.apify_actors = [
            "junipr/earnings-call-transcript-scraper",
            "lucky-chap/earnings-call-transcript-scraper",
            "curious-cyril/earnings-transcript-scraper",
        ]
    
    def fetch_transcript(self, ticker: str, year: int, quarter: int) -> Dict:
        quarter_map = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'}
        quarter_str = quarter_map.get(quarter, f'Q{quarter}')
        
        print(f"Fetching {ticker} {quarter_str} {year}")
        
        # Source 1: FMP API
        if self.fmp_api_key:
            result = self._fetch_from_fmp(ticker, year, quarter_str)
            if result and result.get('status') == 'success':
                return result
        
        # Source 2: YFinance (free, no API key)
        result = self._fetch_from_yfinance(ticker, year, quarter)
        if result and result.get('status') == 'success':
            return result
        
        # Source 3: Finnhub
        if self.finnhub_key:
            result = self._fetch_from_finnhub(ticker, year, quarter)
            if result and result.get('status') == 'success':
                return result
        
        # Source 4: Apify actors
        if self.apify_token:
            for actor_id in self.apify_actors:
                result = self._fetch_from_apify(ticker, year, quarter, actor_id)
                if result and result.get('status') == 'success':
                    return result
        
        # All sources failed
        return {
            'status': 'error',
            'error_code': 'ALL_SOURCES_FAILED',
            'error_message': f'No transcript available for {ticker} {quarter_str} {year} from any source',
            'sources_tried': ['FMP', 'YFinance', 'Finnhub', 'Apify'],
            'timestamp': datetime.now().isoformat()
        }
    
    def _fetch_from_fmp(self, ticker: str, year: int, quarter_str: str) -> Dict:
        url = f"{self.base_url}/earning_call_transcript"
        params = {
            'symbol': ticker,
            'quarter': quarter_str,
            'year': year,
            'apikey': self.fmp_api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            print(f"FMP response status: {response.status_code}")
            
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
        except Exception as e:
            print(f"FMP error: {str(e)}")
        
        return None
    
    def _fetch_from_yfinance(self, ticker: str, year: int, quarter: int) -> Dict:
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            transcripts = stock.earnings_transcript
            
            if not transcripts:
                print(f"YFinance: No transcripts found for {ticker}")
                return None
            
            for transcript in transcripts:
                if transcript.get('year') == year and transcript.get('quarter') == quarter:
                    content = transcript.get('content', '')
                    if content and len(content) > 200:
                        print(f"YFinance: Found transcript for {ticker} Q{quarter} {year}")
                        return {
                            'status': 'success',
                            'source': 'Yahoo Finance',
                            'content': content,
                            'source_used': 'yfinance',
                            'timestamp': datetime.now().isoformat()
                        }
            
            print(f"YFinance: No matching quarter for {ticker} Q{quarter} {year}")
            return None
            
        except ImportError:
            print("YFinance: Library not installed. Run: pip install yfinance")
            return None
        except Exception as e:
            print(f"YFinance error: {str(e)}")
            return None
    
    def _fetch_from_finnhub(self, ticker: str, year: int, quarter: int) -> Dict:
        try:
            # Calculate quarter months
            start_month = (quarter - 1) * 3 + 1
            end_month = quarter * 3
            
            url = "https://finnhub.io/api/v1/calendar/earnings"
            params = {
                'symbol': ticker,
                'from': f"{year}-{start_month:02d}-01",
                'to': f"{year}-{end_month:02d}-28",
                'token': self.finnhub_key
            }
            
            response = requests.get(url, params=params, timeout=15)
            print(f"Finnhub response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                earnings_calls = data.get('earningsCalendar', [])
                
                for call in earnings_calls:
                    transcript_url = call.get('transcriptUrl')
                    if transcript_url:
                        transcript_response = requests.get(transcript_url, timeout=15)
                        if transcript_response.status_code == 200:
                            content = transcript_response.text
                            if len(content) > 500:
                                return {
                                    'status': 'success',
                                    'source': 'Finnhub',
                                    'content': content,
                                    'source_used': 'Finnhub API',
                                    'timestamp': datetime.now().isoformat()
                                }
            
            print(f"Finnhub: No transcript found for {ticker}")
            return None
            
        except Exception as e:
            print(f"Finnhub error: {str(e)}")
            return None
    
    def _fetch_from_apify(self, ticker: str, year: int, quarter: int, actor_id: str) -> Dict:
        try:
            from apify_client import ApifyClient
            
            quarter_map = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'}
            quarter_str = quarter_map.get(quarter, f'Q{quarter}')
            
            client = ApifyClient(self.apify_token)
            
            run_input = {
                "tickers": [ticker],
                "year": year,
                "quarter": quarter_str,
                "maxTranscriptsPerCompany": 1
            }
            
            print(f"Apify: Trying actor {actor_id} for {ticker} {quarter_str} {year}")
            
            run = client.actor(actor_id).call(run_input=run_input)
            
            if run and run.get('status') == 'SUCCEEDED':
                dataset_client = client.dataset(run.get('defaultDatasetId'))
                items = list(dataset_client.iterate_items())
                
                if items and len(items) > 0:
                    item = items[0]
                    content = item.get('transcript', '')
                    if not content:
                        content = item.get('content', '')
                    if not content:
                        content = item.get('text', '')
                    
                    if content and len(content) > 200:
                        print(f"Apify: Found transcript using {actor_id}")
                        return {
                            'status': 'success',
                            'source': 'Apify',
                            'content': content,
                            'source_used': f'Apify ({actor_id})',
                            'timestamp': datetime.now().isoformat()
                        }
                    else:
                        print(f"Apify: Content too short or empty from {actor_id}")
                else:
                    print(f"Apify: No items in dataset from {actor_id}")
            else:
                print(f"Apify: Run failed for {actor_id}")
                
        except ImportError:
            print("Apify: Client not installed. Run: pip install apify-client")
        except Exception as e:
            print(f"Apify error for {actor_id}: {str(e)}")
        
        return None
