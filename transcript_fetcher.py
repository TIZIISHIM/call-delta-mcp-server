

import requests
import re
from bs4 import BeautifulSoup
from typing import Dict, Optional, List
from datetime import datetime
import time
import json

class TranscriptFetcher:
    """Fetch earnings call transcripts with defensive fallback chain."""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        # Known company IR page patterns
        self.ir_patterns = {
            'NVDA': 'https://investor.nvidia.com/events/event-details/default.aspx',
            'TSLA': 'https://ir.tesla.com/events/',
            'AAPL': 'https://investor.apple.com/events/',
            'MSFT': 'https://www.microsoft.com/en-us/Investor/events/',
            'GOOGL': 'https://abc.xyz/investor/',
            'AMZN': 'https://ir.aboutamazon.com/events/',
            'META': 'https://investor.fb.com/events/',
            'PLTR': 'https://investors.palantir.com/events/'
        }
    
    def fetch_transcript(self, ticker: str, year: int, quarter: int) -> Dict:
        """
        Fetch transcript using fallback chain.
        
        Returns:
            Dict with 'status', 'content', 'source', and 'error' (if any)
        """
        # Step 1: Try Seeking Alpha
        result = self._fetch_from_seeking_alpha(ticker, year, quarter)
        if result['status'] == 'success':
            return result
        
        # Step 2: Try Fool.com
        result = self._fetch_from_fool(ticker, year, quarter)
        if result['status'] == 'success':
            return result
        
        # Step 3: Try IR page (limited support)
        result = self._fetch_from_ir_page(ticker, year, quarter)
        if result['status'] == 'success':
            return result
        
        # Step 4: All sources failed - return clean error
        return {
            'status': 'error',
            'error_code': 'SOURCE_UNAVAILABLE',
            'error_message': f"Transcript for {ticker} Q{quarter} {year} not available. Primary source (Seeking Alpha) is blocked or rate-limited. No fallback source contains this transcript.",
            'ticker': ticker,
            'year': year,
            'quarter': quarter,
            'sources_tried': ['Seeking Alpha', 'Fool.com', 'IR Page'],
            'timestamp': datetime.now().isoformat()
        }
    
    def _fetch_from_seeking_alpha(self, ticker: str, year: int, quarter: int) -> Dict:
        """Fetch transcript from Seeking Alpha."""
        try:
            # Build search URL
            search_url = f"https://seekingalpha.com/search?q={ticker}%20Q{quarter}%20{year}%20earnings%20call%20transcript"
            
            response = requests.get(search_url, headers=self.headers, timeout=15)
            
            if response.status_code == 429:
                return {
                    'status': 'error',
                    'source': 'Seeking Alpha',
                    'error_code': 'RATE_LIMITED',
                    'error_message': 'Seeking Alpha rate limit reached. This source is temporarily unavailable.'
                }
            
            if response.status_code != 200:
                return {
                    'status': 'error',
                    'source': 'Seeking Alpha',
                    'error_code': f'HTTP_{response.status_code}',
                    'error_message': f'Seeking Alpha returned status {response.status_code}'
                }
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find transcript link
            # Pattern: /article/XXXXXX-company-name-qX-year-earnings-call-transcript
            pattern = re.compile(r'/article/.*-q{}-{}-.*-transcript'.format(quarter, year), re.IGNORECASE)
            link = soup.find('a', href=pattern)
            
            if not link:
                return {
                    'status': 'error',
                    'source': 'Seeking Alpha',
                    'error_code': 'NOT_FOUND',
                    'error_message': f'No transcript found for {ticker} Q{quarter} {year} on Seeking Alpha'
                }
            
            # Fetch transcript page
            transcript_url = link['href']
            if not transcript_url.startswith('http'):
                transcript_url = 'https://seekingalpha.com' + transcript_url
            
            response = requests.get(transcript_url, headers=self.headers, timeout=15)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract transcript text
            transcript_text = self._extract_transcript_text_seeking_alpha(soup)
            
            if not transcript_text or len(transcript_text) < 500:
                return {
                    'status': 'error',
                    'source': 'Seeking Alpha',
                    'error_code': 'EMPTY_CONTENT',
                    'error_message': 'Transcript found but content could not be extracted'
                }
            
            return {
                'status': 'success',
                'source': 'Seeking Alpha',
                'content': transcript_text,
                'url': transcript_url,
                'ticker': ticker,
                'quarter': quarter,
                'year': year,
                'timestamp': datetime.now().isoformat()
            }
            
        except requests.exceptions.Timeout:
            return {
                'status': 'error',
                'source': 'Seeking Alpha',
                'error_code': 'TIMEOUT',
                'error_message': 'Seeking Alpha connection timed out'
            }
        except Exception as e:
            return {
                'status': 'error',
                'source': 'Seeking Alpha',
                'error_code': 'UNKNOWN',
                'error_message': f'Seeking Alpha error: {str(e)}'
            }
    
    def _extract_transcript_text_seeking_alpha(self, soup: BeautifulSoup) -> str:
        """Extract transcript text from Seeking Alpha page."""
        # Try to find transcript content
        transcript_div = soup.find('div', {'data-test-id': 'transcript-content'})
        if transcript_div:
            return transcript_div.get_text(separator='\n', strip=True)
        
        # Fallback: look for article content
        article = soup.find('article')
        if article:
            return article.get_text(separator='\n', strip=True)
        
        # Fallback: look for div with transcript in class
        for div in soup.find_all('div', class_=re.compile(r'transcript', re.I)):
            text = div.get_text(separator='\n', strip=True)
            if len(text) > 1000:
                return text
        
        return ''
    
    def _fetch_from_fool(self, ticker: str, year: int, quarter: int) -> Dict:
        """Fetch transcript from Fool.com (The Motley Fool)."""
        try:
            # Fool.com URL pattern
            url = f"https://www.fool.com/earnings-call-transcript/{year}/{quarter}/{ticker.lower()}/"
            
            response = requests.get(url, headers=self.headers, timeout=15)
            
            if response.status_code != 200:
                return {
                    'status': 'error',
                    'source': 'Fool.com',
                    'error_code': f'HTTP_{response.status_code}',
                    'error_message': f'Fool.com returned status {response.status_code}'
                }
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract transcript text
            article = soup.find('article')
            if article:
                text = article.get_text(separator='\n', strip=True)
                if len(text) > 1000:
                    return {
                        'status': 'success',
                        'source': 'Fool.com',
                        'content': text,
                        'url': url,
                        'ticker': ticker,
                        'quarter': quarter,
                        'year': year,
                        'timestamp': datetime.now().isoformat()
                    }
            
            return {
                'status': 'error',
                'source': 'Fool.com',
                'error_code': 'NO_CONTENT',
                'error_message': 'Transcript found but content extraction failed'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'source': 'Fool.com',
                'error_code': 'UNKNOWN',
                'error_message': f'Fool.com error: {str(e)}'
            }
    
    def _fetch_from_ir_page(self, ticker: str, year: int, quarter: int) -> Dict:
        """Fetch transcript from company IR page (limited support)."""
        if ticker.upper() not in self.ir_patterns:
            return {
                'status': 'error',
                'source': 'IR Page',
                'error_code': 'NO_PATTERN',
                'error_message': f'No IR page pattern configured for {ticker}'
            }
        
        try:
            url = self.ir_patterns[ticker.upper()]
            response = requests.get(url, headers=self.headers, timeout=15)
            
            if response.status_code != 200:
                return {
                    'status': 'error',
                    'source': 'IR Page',
                    'error_code': f'HTTP_{response.status_code}',
                    'error_message': f'IR page returned status {response.status_code}'
                }
            
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator='\n', strip=True)
            
            # Check if transcript exists (very basic check)
            if 'earnings' in text.lower() and 'transcript' in text.lower():
                # Extract relevant section (simplified)
                lines = text.split('\n')
                transcript_lines = []
                capture = False
                
                for line in lines:
                    if 'prepared remarks' in line.lower():
                        capture = True
                    if capture:
                        transcript_lines.append(line)
                    if 'question and answer' in line.lower() and capture:
                        break
                
                transcript_text = '\n'.join(transcript_lines)
                if len(transcript_text) > 500:
                    return {
                        'status': 'success',
                        'source': 'IR Page',
                        'content': transcript_text,
                        'url': url,
                        'ticker': ticker,
                        'quarter': quarter,
                        'year': year,
                        'timestamp': datetime.now().isoformat()
                    }
            
            return {
                'status': 'error',
                'source': 'IR Page',
                'error_code': 'NO_TRANSCRIPT',
                'error_message': f'No transcript found on IR page for {ticker}'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'source': 'IR Page',
                'error_code': 'UNKNOWN',
                'error_message': f'IR page error: {str(e)}'
            }
