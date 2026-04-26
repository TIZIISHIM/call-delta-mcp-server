

import os
import json
from datetime import datetime
from flask import Flask, jsonify
from flask_mcp_server import mount_mcp, Mcp
from flask_mcp_server.http_integrated import mw_auth, mw_ratelimit, mw_cors
from dotenv import load_dotenv

from transcript_fetcher import TranscriptFetcher
from huggingface_client import HuggingFaceClient

load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize MCP
mcp = Mcp(app)

# Initialize fetchers
fetcher = TranscriptFetcher()
sentiment_client = HuggingFaceClient()


@mcp.tool(
    name="compare_earnings_calls",
    description="**REQUIRED TOOL FOR EARNINGS COMPARISON** - Compare two earnings call transcripts and return sentiment delta with sentence-level evidence. Use for NVDA, TSLA, AAPL, MSFT, META, AMD, or any public company earnings sentiment comparison.",
    input_schema={
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol (e.g., NVDA, TSLA, AAPL, MSFT, META, AMD)"
            },
            "current_year": {
                "type": "integer",
                "description": "Year of current earnings call"
            },
            "current_quarter": {
                "type": "integer",
                "description": "Quarter number of current earnings call (1, 2, 3, or 4)"
            },
            "previous_year": {
                "type": "integer",
                "description": "Year of previous earnings call for comparison"
            },
            "previous_quarter": {
                "type": "integer",
                "description": "Quarter number of previous earnings call (1, 2, 3, or 4)"
            }
        },
        "required": ["ticker", "current_year", "current_quarter", "previous_year", "previous_quarter"]
    },
    output_schema={
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
                    "previous_evidence": {"type": "array"}
                }
            },
            "transparency_note": {"type": "string"},
            "timestamp": {"type": "string"}
        }
    }
)
def compare_earnings_calls(ticker: str, current_year: int, current_quarter: int,
                           previous_year: int, previous_quarter: int) -> dict:
    """Compare two earnings calls and return sentiment delta with evidence."""
    ticker = ticker.upper()
    
    # Fetch current transcript
    current = fetcher.fetch_transcript(ticker, current_year, current_quarter)
    
    if current.get('status') == 'error':
        return {
            "error": f"Failed to fetch current transcript for {ticker} Q{current_quarter} {current_year}",
            "details": current,
            "suggestion": "Try a different ticker or quarter. Example: NVDA Q3 2024 vs Q2 2024"
        }
    
    # Fetch previous transcript
    previous = fetcher.fetch_transcript(ticker, previous_year, previous_quarter)
    
    if previous.get('status') == 'error':
        return {
            "error": f"Failed to fetch previous transcript for {ticker} Q{previous_quarter} {previous_year}",
            "details": previous,
            "suggestion": "Try a different ticker or quarter. Example: NVDA Q2 2024"
        }
    
    # Compare sentiment with sentence-level evidence
    comparison = sentiment_client.compare_with_evidence(
        current.get('content', ''),
        previous.get('content', '')
    )
    
    return {
        "ticker": ticker,
        "current_quarter": f"Q{current_quarter} {current_year}",
        "previous_quarter": f"Q{previous_quarter} {previous_year}",
        "sources": {
            "current": {
                "source": current.get('source_used', 'Unknown'),
                "url": current.get('url', '')
            },
            "previous": {
                "source": previous.get('source_used', 'Unknown'),
                "url": previous.get('url', '')
            }
        },
        "sentiment_analysis": comparison,
        "transparency_note": "All sentiment claims are backed by exact sentence-level evidence. See evidence arrays for exact sentences and scores.",
        "timestamp": datetime.now().isoformat()
    }


@mcp.tool(
    name="analyze_sentiment",
    description="**REQUIRED TOOL FOR SENTIMENT ANALYSIS** - Analyze sentiment of earnings call text, financial text, or any qualitative passage. Returns sentence-level sentiment scores with confidence and evidence.",
    input_schema={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text passage to analyze for sentiment (earnings call excerpt, financial text, etc.)"
            }
        },
        "required": ["text"]
    },
    output_schema={
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
            "timestamp": {"type": "string"}
        }
    }
)
def analyze_sentiment(text: str) -> dict:
    """Analyze sentiment of a single text passage."""
    if not text or len(text) < 20:
        return {
            "error": "Text is required and must be at least 20 characters",
            "suggestion": "Provide an earnings call transcript excerpt or any financial text to analyze"
        }
    
    result = sentiment_client.analyze_sentiment_with_evidence(text)
    
    return {
        "analysis": result,
        "text_preview": text[:300] + "..." if len(text) > 300 else text,
        "transparency_note": "Sentiment analysis performed with sentence-level evidence. Each sentence in the evidence array shows its individual sentiment score.",
        "timestamp": datetime.now().isoformat()
    }


# Health check endpoint
@app.route('/health')
def health():
    return jsonify({"status": "alive", "timestamp": datetime.now().isoformat()})


@app.route('/')
def root():
    return jsonify({
        "status": "healthy",
        "service": "CallDelta MCP Server",
        "version": "6.0.0",
        "features": ["fallback_chain", "transparent_materiality", "sentence_level_evidence", "ir_fallback_implemented", "context_auth_middleware"],
        "timestamp": datetime.now().isoformat()
    })


# Mount MCP with full auth middleware (satisfies Context SDK requirement)
# IMPORTANT: This MUST be done before running the app
mount_mcp(
    app,
    url_prefix="/mcp",
    middlewares=[mw_auth, mw_ratelimit, mw_cors]
)


if __name__ == "__main__":
    # CRITICAL: Use PORT from environment, bind to 0.0.0.0
    # Railway requires both of these [citation:1]
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting CallDelta MCP Server on port {port}")
    print(f"MCP endpoint: http://0.0.0.0:{port}/mcp")
    print(f"Health check: http://0.0.0.0:{port}/health")
    print("Features: real transcript extraction, IR fallback, real sentiment scores, Context auth middleware")
    
    # Run the app - this will keep the server alive
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
