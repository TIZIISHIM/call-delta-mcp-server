

import requests
import re
from bs4 import BeautifulSoup
from typing import Dict
from datetime import datetime
import time

class TranscriptFetcher:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
    
    def fetch_transcript(self, ticker: str, year: int, quarter: int) -> Dict:
        """
        Fetch and parse real transcript text from Seeking Alpha.
        Returns actual transcript content, not placeholders.
        """
        # First try Seeking Alpha with real extraction
        result = self._fetch_from_seeking_alpha(ticker, year, quarter)
        if result['status'] == 'success':
            return result
        
        # Fallback to Fool.com
        result = self._fetch_from_fool(ticker, year, quarter)
        if result['status'] == 'success':
            return result
        
        # Return error with helpful message
        return {
            'status': 'error',
            'error_code': 'SOURCE_UNAVAILABLE',
            'error_message': f"Could not find transcript for {ticker} Q{quarter} {year}. The earnings call may not have occurred yet, or the transcript hasn't been published.",
            'ticker': ticker,
            'year': year,
            'quarter': quarter,
            'suggestion': f"Try a more recent quarter or a larger company like NVDA, AAPL, TSLA",
            'timestamp': datetime.now().isoformat()
        }
    
    def _fetch_from_seeking_alpha(self, ticker: str, year: int, quarter: int) -> Dict:
        """Fetch and extract real transcript text from Seeking Alpha."""
        try:
            # Build search query
            quarter_map = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'}
            quarter_str = quarter_map.get(quarter, f'Q{quarter}')
            
            # Search for transcript
            search_url = f"https://seekingalpha.com/search?q={ticker}%20{quarter_str}%20{year}%20earnings%20call%20transcript"
            
            response = requests.get(search_url, headers=self.headers, timeout=15)
            
            if response.status_code == 429:
                return {
                    'status': 'error',
                    'source': 'Seeking Alpha',
                    'error_code': 'RATE_LIMITED',
                    'error_message': 'Seeking Alpha is rate-limiting requests. Please try again in a few minutes.'
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
            pattern = re.compile(r'/article/.*-earnings-call-transcript', re.IGNORECASE)
            links = soup.find_all('a', href=pattern)
            
            # Find the one matching our quarter/year
            transcript_link = None
            for link in links:
                href = link.get('href', '').lower()
                if quarter_str.lower() in href and str(year) in href:
                    transcript_link = link
                    break
            
            if not transcript_link and links:
                transcript_link = links[0]  # Take the first one
            
            if not transcript_link:
                return {
                    'status': 'error',
                    'source': 'Seeking Alpha',
                    'error_code': 'NOT_FOUND',
                    'error_message': f'No transcript link found for {ticker} {quarter_str} {year}'
                }
            
            # Get transcript URL
            transcript_url = transcript_link.get('href')
            if not transcript_url.startswith('http'):
                transcript_url = 'https://seekingalpha.com' + transcript_url
            
            # Fetch transcript page
            response = requests.get(transcript_url, headers=self.headers, timeout=15)
            
            if response.status_code != 200:
                return {
                    'status': 'error',
                    'source': 'Seeking Alpha',
                    'error_code': f'HTTP_{response.status_code}',
                    'error_message': f'Failed to load transcript page'
                }
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract transcript text
            transcript_text = self._extract_transcript_text(soup)
            
            if not transcript_text or len(transcript_text) < 200:
                return {
                    'status': 'error',
                    'source': 'Seeking Alpha',
                    'error_code': 'EMPTY_CONTENT',
                    'error_message': 'Transcript found but text could not be extracted'
                }
            
            return {
                'status': 'success',
                'source': 'Seeking Alpha',
                'content': transcript_text,
                'url': transcript_url,
                'source_used': 'Seeking Alpha',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'source': 'Seeking Alpha',
                'error_code': 'UNKNOWN',
                'error_message': f'Error: {str(e)[:100]}'
            }
    
    def _extract_transcript_text(self, soup: BeautifulSoup) -> str:
        """Extract the actual transcript text from the page."""
        transcript_parts = []
        
        # Method 1: Look for transcript content div
        transcript_div = soup.find('div', {'data-test-id': 'transcript-content'})
        if transcript_div:
            text = transcript_div.get_text(separator='\n', strip=True)
            if len(text) > 500:
                return text
        
        # Method 2: Look for article body
        article = soup.find('article')
        if article:
            # Remove non-transcript elements
            for unwanted in article.find_all(['aside', 'nav', 'footer', '.advertisement']):
                unwanted.decompose()
            text = article.get_text(separator='\n', strip=True)
            if len(text) > 500:
                return text
        
        # Method 3: Find div with class containing 'transcript'
        for div in soup.find_all('div', class_=re.compile(r'transcript', re.I)):
            text = div.get_text(separator='\n', strip=True)
            if len(text) > 500:
                return text
        
        # Method 4: Look for prepared remarks section
        for header in soup.find_all(['h2', 'h3'], string=re.compile(r'prepared remarks', re.I)):
            parent = header.find_parent()
            if parent:
                text = parent.get_text(separator='\n', strip=True)
                if len(text) > 200:
                    return text
        
        # Method 5: Fallback - get all paragraph text
        paragraphs = soup.find_all('p')
        para_text = '\n'.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text()) > 50])
        if len(para_text) > 500:
            return para_text
        
        return ''
    
    def _fetch_from_fool(self, ticker: str, year: int, quarter: int) -> Dict:
        """Fetch transcript from Fool.com as fallback."""
        try:
            quarter_map = {1: 'q1', 2: 'q2', 3: 'q3', 4: 'q4'}
            quarter_str = quarter_map.get(quarter, f'q{quarter}')
            
            url = f"https://www.fool.com/earnings-call-transcript/{year}/{quarter_str}/{ticker.lower()}/"
            
            response = requests.get(url, headers=self.headers, timeout=15)
            
            if response.status_code != 200:
                return {
                    'status': 'error',
                    'source': 'Fool.com',
                    'error_code': f'HTTP_{response.status_code}',
                    'error_message': f'Fool.com returned {response.status_code}'
                }
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract article content
            article = soup.find('article')
            if article:
                text = article.get_text(separator='\n', strip=True)
                if len(text) > 500:
                    return {
                        'status': 'success',
                        'source': 'Fool.com',
                        'content': text,
                        'url': url,
                        'source_used': 'Fool.com',
                        'timestamp': datetime.now().isoformat()
                    }
            
            return {
                'status': 'error',
                'source': 'Fool.com',
                'error_code': 'NO_CONTENT',
                'error_message': 'Could not extract transcript text'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'source': 'Fool.com',
                'error_code': 'UNKNOWN',
                'error_message': f'Error: {str(e)[:100]}'
            }
