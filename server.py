"""
CallDelta MCP Server - Earnings Call Transcript Delta Intelligence
Deployed on Railway with HTTP transport.
"""

import json
import os
from datetime import datetime
from typing import Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

from transcript_fetcher import TranscriptFetcher
from sentiment_analyzer import TransparentSentimentAnalyzer
from delta_detector import TranscriptDeltaDetector

# Initialize components
fetcher = TranscriptFetcher()
sentiment_analyzer = TransparentSentimentAnalyzer()
delta_detector = TranscriptDeltaDetector()

# Create FastAPI app
app = FastAPI(title="CallDelta MCP Server", description="Earnings Call Transcript Delta Intelligence")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "CallDelta MCP Server",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/call")
async def handle_tool_call(request: Request):
    """Handle MCP tool calls."""
    body = await request.json()
    
    tool_name = body.get("tool")
    arguments = body.get("arguments", {})
    
    if tool_name == "compare_earnings_calls":
        result = await compare_earnings_calls(arguments)
        return JSONResponse(content=result)
    
    elif tool_name == "analyze_sentiment":
        result = await analyze_sentiment(arguments)
        return JSONResponse(content=result)
    
    else:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown tool: {tool_name}"}
        )


async def compare_earnings_calls(arguments: Dict) -> Dict:
    """Compare two earnings call transcripts."""
    ticker = arguments.get("ticker", "").upper()
    current_year = arguments.get("current_year")
    current_quarter = arguments.get("current_quarter")
    previous_year = arguments.get("previous_year")
    previous_quarter = arguments.get("previous_quarter")
    
    if not ticker:
        return {"error": "Ticker is required"}
    
    # Fetch current transcript
    current_result = fetcher.fetch_transcript(ticker, current_year, current_quarter)
    
    if current_result['status'] == 'error':
        return {
            "error": "Failed to fetch current transcript",
            "details": current_result
        }
    
    # Fetch previous transcript
    previous_result = fetcher.fetch_transcript(ticker, previous_year, previous_quarter)
    
    if previous_result['status'] == 'error':
        return {
            "error": "Failed to fetch previous transcript",
            "details": previous_result
        }
    
    # Compare transcripts
    sentiment_comparison = sentiment_analyzer.compare_transcripts(
        current_result['content'],
        previous_result['content']
    )
    
    text_changes = delta_detector.detect_changes(
        current_result['content'],
        previous_result['content']
    )
    
    return {
        'tool': 'compare_earnings_calls',
        'ticker': ticker,
        'current_quarter': f"Q{current_quarter} {current_year}",
        'previous_quarter': f"Q{previous_quarter} {previous_year}",
        'sources': {
            'current': {
                'source': current_result['source'],
                'url': current_result.get('url'),
                'fetched_at': current_result.get('timestamp')
            },
            'previous': {
                'source': previous_result['source'],
                'url': previous_result.get('url'),
                'fetched_at': previous_result.get('timestamp')
            }
        },
        'sentiment_analysis': sentiment_comparison,
        'text_changes': text_changes,
        'query_timestamp': datetime.now().isoformat()
    }


async def analyze_sentiment(arguments: Dict) -> Dict:
    """Analyze sentiment of a transcript."""
    text = arguments.get("text", "")
    
    if not text or len(text) < 50:
        return {"error": "Text is required and must be at least 50 characters"}
    
    analysis = sentiment_analyzer.analyze_transcript(text)
    
    return {
        'tool': 'analyze_sentiment',
        'analysis': analysis,
        'query_timestamp': datetime.now().isoformat()
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting CallDelta MCP Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
