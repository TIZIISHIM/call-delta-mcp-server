

import os
from datetime import datetime
from fastmcp import FastMCP
from ctxprotocol import verify_context_request, is_protected_mcp_method
from fastapi import Request
import uvicorn

from transcript_fetcher import TranscriptFetcher
from huggingface_client import HuggingFaceClient

# Initialize fetchers
fetcher = TranscriptFetcher()
sentiment_client = HuggingFaceClient()

# Create FastMCP server (includes built-in FastAPI)
mcp = FastMCP("CallDelta MCP Server")

# Define output schemas
COMPARE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "ticker": {"type": "string"},
        "current_quarter": {"type": "string"},
        "previous_quarter": {"type": "string"},
        "sources": {"type": "object"},
        "sentiment_analysis": {"type": "object"},
        "transparency_note": {"type": "string"},
        "error": {"type": "string"},
        "timestamp": {"type": "string"}
    }
}

ANALYZE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {"type": "object"},
        "transparency_note": {"type": "string"},
        "error": {"type": "string"},
        "timestamp": {"type": "string"}
    }
}


@mcp.tool(
    name="compare_earnings_calls",
    description="Compare two earnings call transcripts and return sentiment delta with sentence-level evidence.",
    output_schema=COMPARE_OUTPUT_SCHEMA
)
def compare_earnings_calls(
    ticker: str,
    current_year: int,
    current_quarter: int,
    previous_year: int,
    previous_quarter: int
) -> dict:
    """Compare two earnings calls."""
    ticker = ticker.upper()
    
    # Fetch current transcript
    current = fetcher.fetch_transcript(ticker, current_year, current_quarter)
    if current.get('status') == 'error':
        return {
            "error": f"Failed to fetch transcript for {ticker} Q{current_quarter} {current_year}",
            "details": current,
            "timestamp": datetime.now().isoformat()
        }
    
    # Fetch previous transcript
    previous = fetcher.fetch_transcript(ticker, previous_year, previous_quarter)
    if previous.get('status') == 'error':
        return {
            "error": f"Failed to fetch transcript for {ticker} Q{previous_quarter} {previous_year}",
            "details": previous,
            "timestamp": datetime.now().isoformat()
        }
    
    # Compare sentiment
    comparison = sentiment_client.compare_with_evidence(
        current.get('content', ''),
        previous.get('content', '')
    )
    
    return {
        "ticker": ticker,
        "current_quarter": f"Q{current_quarter} {current_year}",
        "previous_quarter": f"Q{previous_quarter} {previous_year}",
        "sources": {
            "current": {"source": current.get('source_used', 'Unknown')},
            "previous": {"source": previous.get('source_used', 'Unknown')}
        },
        "sentiment_analysis": comparison,
        "transparency_note": "All claims backed by sentence-level evidence.",
        "timestamp": datetime.now().isoformat()
    }


@mcp.tool(
    name="analyze_sentiment",
    description="Analyze sentiment of earnings call text with sentence-level evidence.",
    output_schema=ANALYZE_OUTPUT_SCHEMA
)
def analyze_sentiment(text: str) -> dict:
    """Analyze sentiment of a single text."""
    if len(text) < 20:
        return {
            "error": "Text must be at least 20 characters",
            "timestamp": datetime.now().isoformat()
        }
    
    result = sentiment_client.analyze_sentiment_with_evidence(text)
    return {
        "analysis": result,
        "transparency_note": "Sentence-level evidence provided.",
        "timestamp": datetime.now().isoformat()
    }


# Add health check route
@mcp.get("/health")
async def health():
    return {"status": "alive", "timestamp": datetime.now().isoformat()}


# Add root route
@mcp.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "CallDelta MCP Server",
        "version": "11.0.0",
        "features": ["sse_transport", "output_schema", "fastmcp"],
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting CallDelta MCP Server on port {port}")
    print(f"SSE endpoint: http://0.0.0.0:{port}/sse")
    print(f"Health check: http://0.0.0.0:{port}/health")
    
    # Run with SSE transport (fastmcp handles this)
    # fastmcp automatically creates /sse endpoint
    mcp.run(transport="sse", host="0.0.0.0", port=port)
