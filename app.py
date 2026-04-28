

import os
import json
from datetime import datetime
from typing import Dict, Any
from mcp.server.fastmcp import FastMCP
import requests

from transcript_fetcher import TranscriptFetcher
from huggingface_client import HuggingFaceClient

# Initialize fetchers
fetcher = TranscriptFetcher()
sentiment_client = HuggingFaceClient()

# Initialize MCP server
port = int(os.environ.get("PORT", 8080))
mcp = FastMCP(
    name="CallDelta MCP Server",
    host="0.0.0.0",
    port=port
)

# Define outputSchema for compare_earnings_calls
COMPARE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "ticker": {"type": "string"},
        "current_quarter": {"type": "string"},
        "previous_quarter": {"type": "string"},
        "sources": {
            "type": "object",
            "properties": {
                "current": {"type": "object"},
                "previous": {"type": "object"}
            }
        },
        "sentiment_analysis": {
            "type": "object",
            "properties": {
                "overall_delta": {"type": "object"},
                "current_evidence": {"type": "array"},
                "previous_evidence": {"type": "array"},
                "methodology": {"type": "object"}
            }
        },
        "transparency_note": {"type": "string"},
        "timestamp": {"type": "string"}
    }
}

# Define outputSchema for analyze_sentiment
ANALYZE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {
            "type": "object",
            "properties": {
                "sentiment_label": {"type": "string"},
                "sentiment_score": {"type": "number"},
                "confidence": {"type": "number"},
                "evidence": {"type": "array"},
                "sentence_count": {"type": "integer"}
            }
        },
        "text_preview": {"type": "string"},
        "transparency_note": {"type": "string"},
        "timestamp": {"type": "string"}
    }
}


@mcp.tool(
    name="compare_earnings_calls",
    description="Compare two earnings call transcripts and return sentiment delta with sentence-level evidence. Use for NVDA, TSLA, AAPL, MSFT, META, AMD earnings sentiment comparison.",
    outputSchema=COMPARE_OUTPUT_SCHEMA,
    _meta={
        "surface": "query",
        "queryEligible": True,
        "contextRequirements": [],
        "rateLimit": {
            "maxRequestsPerMinute": 30,
            "cooldownMs": 2000
        }
    }
)
def compare_earnings_calls(
    ticker: str,
    current_year: int,
    current_quarter: int,
    previous_year: int,
    previous_quarter: int
) -> Dict[str, Any]:
    """Compare two earnings calls and return sentiment delta."""
    ticker = ticker.upper()
    
    current = fetcher.fetch_transcript(ticker, current_year, current_quarter)
    if current.get('status') == 'error':
        return current
    
    previous = fetcher.fetch_transcript(ticker, previous_year, previous_quarter)
    if previous.get('status') == 'error':
        return previous
    
    comparison = sentiment_client.compare_with_evidence(
        current.get('content', ''),
        previous.get('content', '')
    )
    
    return {
        "ticker": ticker,
        "current_quarter": f"Q{current_quarter} {current_year}",
        "previous_quarter": f"Q{previous_quarter} {previous_year}",
        "sources": {
            "current": {"source": current.get('source_used', 'Unknown'), "url": current.get('url', '')},
            "previous": {"source": previous.get('source_used', 'Unknown'), "url": previous.get('url', '')}
        },
        "sentiment_analysis": comparison,
        "transparency_note": "All sentiment claims are backed by exact sentence-level evidence.",
        "timestamp": datetime.now().isoformat()
    }


@mcp.tool(
    name="analyze_sentiment",
    description="Analyze sentiment of earnings call text, financial text, or any qualitative passage. Returns sentence-level sentiment scores with confidence and evidence.",
    outputSchema=ANALYZE_OUTPUT_SCHEMA,
    _meta={
        "surface": "query",
        "queryEligible": True,
        "contextRequirements": [],
        "rateLimit": {
            "maxRequestsPerMinute": 60,
            "cooldownMs": 1000
        }
    }
)
def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Analyze sentiment of a single text passage."""
    if not text or len(text) < 20:
        return {
            "error": "Text must be at least 20 characters",
            "timestamp": datetime.now().isoformat()
        }
    
    result = sentiment_client.analyze_sentiment_with_evidence(text)
    
    return {
        "analysis": result,
        "text_preview": text[:300] + "..." if len(text) > 300 else text,
        "transparency_note": "Each sentence analyzed individually with evidence.",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    print(f"Starting CallDelta MCP Server on port {port}")
    print(f"Features: FMP API, real sentiment, outputSchema, _meta, auth ready")
    mcp.run(transport="sse")
