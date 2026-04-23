

import os
import json
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

from transcript_fetcher import TranscriptFetcher
from huggingface_client import HuggingFaceClient

app = FastAPI(title="CallDelta MCP Server")
fetcher = TranscriptFetcher()
sentiment_client = HuggingFaceClient()


@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "CallDelta MCP Server",
        "version": "4.0.0",
        "features": ["fallback_chain", "transparent_materiality", "sentence_level_evidence", "huggingface_inference"],
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health():
    return {"status": "alive", "timestamp": datetime.now().isoformat()}


@app.post("/call")
async def handle_tool_call(request: Request):
    try:
        body = await request.json()
    except:
        body = {}
    
    tool_name = body.get("tool", "")
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


async def compare_earnings_calls(args: dict) -> dict:
    ticker = args.get("ticker", "").upper()
    current_year = args.get("current_year")
    current_quarter = args.get("current_quarter")
    previous_year = args.get("previous_year")
    previous_quarter = args.get("previous_quarter")
    
    if not ticker:
        return {"error": "Ticker is required"}
    
    # Fetch current transcript
    current = fetcher.fetch_transcript(ticker, current_year, current_quarter)
    
    if current['status'] == 'error':
        return {
            "error": "Failed to fetch current transcript",
            "source_error": current,
            "user_action": f"Transcript for {ticker} Q{current_quarter} {current_year} is not available. Try a different ticker or quarter."
        }
    
    # Fetch previous transcript
    previous = fetcher.fetch_transcript(ticker, previous_year, previous_quarter)
    
    if previous['status'] == 'error':
        return {
            "error": "Failed to fetch previous transcript",
            "source_error": previous,
            "user_action": f"Transcript for {ticker} Q{previous_quarter} {previous_year} is not available. Try a different ticker or quarter."
        }
    
    # Analyze sentiment with sentence-level evidence (transparent materiality)
    comparison = sentiment_client.compare_with_evidence(
        current.get('content', ''),
        previous.get('content', '')
    )
    
    return {
        "tool": "compare_earnings_calls",
        "ticker": ticker,
        "current_quarter": f"Q{current_quarter} {current_year}",
        "previous_quarter": f"Q{previous_quarter} {previous_year}",
        "sources": {
            "current": {
                "source": current.get('source_used', 'Unknown'),
                "url": current.get('url'),
                "status": current['status']
            },
            "previous": {
                "source": previous.get('source_used', 'Unknown'),
                "url": previous.get('url'),
                "status": previous['status']
            }
        },
        "sentiment_analysis": comparison,
        "transparency_note": "All sentiment claims are backed by exact sentence-level evidence. See current_evidence and previous_evidence arrays for exact sentences, their sentiment scores, and confidence levels. Each sentence was analyzed individually using the Hugging Face inference API.",
        "query_timestamp": datetime.now().isoformat()
    }


async def analyze_sentiment(args: dict) -> dict:
    text = args.get("text", "")
    
    if not text or len(text) < 20:
        return {
            "error": "Text is required and must be at least 20 characters",
            "user_action": "Provide an earnings call transcript excerpt or full text"
        }
    
    # Use sentence-level evidence for single transcript analysis
    result = sentiment_client.analyze_sentiment_with_evidence(text)
    
    return {
        "tool": "analyze_sentiment",
        "analysis": result,
        "text_preview": text[:200] + "..." if len(text) > 200 else text,
        "query_timestamp": datetime.now().isoformat(),
        "transparency_note": "Sentiment analysis performed using Hugging Face inference API with sentence-level evidence. Each sentence in the evidence array shows its individual sentiment score."
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting CallDelta MCP Server on port {port}")
    print(f"Health check: http://0.0.0.0:{port}/health")
    print("Using Hugging Face Inference API for sentiment (no local ML models)")
    print("Transparent materiality enabled: sentence-level evidence provided for every claim")
    uvicorn.run(app, host="0.0.0.0", port=port)
