"""
CallDelta MCP Server - Correct Implementation
Using FastAPI + MCP SDK + Context Auth Middleware
"""

import os
import json
from datetime import datetime
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import JSONResponse, Response
from sse_starlette.sse import EventSourceResponse
from ctxprotocol import create_context_middleware, is_protected_mcp_method
import uvicorn

from transcript_fetcher import TranscriptFetcher
from huggingface_client import HuggingFaceClient

# Initialize FastAPI app
app = FastAPI(title="CallDelta MCP Server")

# Create Context auth middleware
verify_context = create_context_middleware(
    audience="https://calldelta-mcp-server-production.up.railway.app"
)

# Initialize fetchers
fetcher = TranscriptFetcher()
sentiment_client = HuggingFaceClient()

# Define tools with outputSchema and _meta
AVAILABLE_TOOLS = [
    {
        "name": "compare_earnings_calls",
        "description": "Compare two earnings call transcripts and return sentiment delta with sentence-level evidence.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol (e.g., NVDA, TSLA, AAPL, MSFT)"},
                "current_year": {"type": "integer", "description": "Year of current earnings call"},
                "current_quarter": {"type": "integer", "description": "Quarter number (1-4)"},
                "previous_year": {"type": "integer", "description": "Year of previous earnings call"},
                "previous_quarter": {"type": "integer", "description": "Quarter number (1-4)"}
            },
            "required": ["ticker", "current_year", "current_quarter", "previous_year", "previous_quarter"]
        },
        "outputSchema": {
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
        },
        "_meta": {"surface": "query", "queryEligible": True}
    },
    {
        "name": "analyze_sentiment",
        "description": "Analyze sentiment of earnings call text with sentence-level evidence.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to analyze for sentiment"}
            },
            "required": ["text"]
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "analysis": {"type": "object"},
                "transparency_note": {"type": "string"},
                "error": {"type": "string"},
                "timestamp": {"type": "string"}
            }
        },
        "_meta": {"surface": "query", "queryEligible": True}
    }
]


@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "CallDelta MCP Server",
        "version": "12.0.0",
        "features": ["http_endpoint", "outputSchema", "_meta", "context_auth_middleware"],
        "tools": [t["name"] for t in AVAILABLE_TOOLS],
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health():
    return {"status": "alive", "timestamp": datetime.now().isoformat()}


@app.post("/mcp")
async def mcp_endpoint(request: Request, context: dict = Depends(verify_context)):
    """
    MCP endpoint with Context auth middleware.
    - tools/list and initialize: no auth required (context will be None)
    - tools/call: requires valid JWT (context contains verified payload)
    """
    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"jsonrpc": "2.0", "error": {"code": -32700, "message": f"Parse error: {str(e)}"}}
        )
    
    method = body.get("method", "")
    msg_id = body.get("id")
    
    print(f"Received: {method} (id: {msg_id})")
    print(f"Auth context present: {context is not None}")
    
    # Initialize handshake
    if method == "initialize":
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "calldelta-mcp-server", "version": "12.0.0"}
            }
        })
    
    # Initialized notification
    elif method == "notifications/initialized":
        return JSONResponse(content={}, status_code=202)
    
    # List tools
    elif method == "tools/list":
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"tools": AVAILABLE_TOOLS}
        })
    
    # Call tool (requires auth - middleware ensures this)
    elif method == "tools/call":
        tool_name = body.get("params", {}).get("name", "")
        arguments = body.get("params", {}).get("arguments", {})
        
        print(f"Executing tool: {tool_name} (authenticated: {context is not None})")
        
        if tool_name == "compare_earnings_calls":
            result = await compare_earnings_calls(arguments)
        elif tool_name == "analyze_sentiment":
            result = await analyze_sentiment(arguments)
        else:
            return JSONResponse(
                status_code=400,
                content={"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"Tool not found: {tool_name}"}}
            )
        
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
        })
    
    # Unknown method
    else:
        return JSONResponse(
            status_code=400,
            content={"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}
        )


async def compare_earnings_calls(args: dict) -> dict:
    """Compare two earnings calls."""
    ticker = args.get("ticker", "").upper()
    current_year = args.get("current_year")
    current_quarter = args.get("current_quarter")
    previous_year = args.get("previous_year")
    previous_quarter = args.get("previous_quarter")
    
    if not ticker:
        return {"error": "Ticker is required", "timestamp": datetime.now().isoformat()}
    
    # Fetch transcripts
    current = fetcher.fetch_transcript(ticker, current_year, current_quarter)
    if current.get('status') == 'error':
        return {"error": f"Failed to fetch transcript for {ticker} Q{current_quarter} {current_year}", "details": current, "timestamp": datetime.now().isoformat()}
    
    previous = fetcher.fetch_transcript(ticker, previous_year, previous_quarter)
    if previous.get('status') == 'error':
        return {"error": f"Failed to fetch transcript for {ticker} Q{previous_quarter} {previous_year}", "details": previous, "timestamp": datetime.now().isoformat()}
    
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


async def analyze_sentiment(args: dict) -> dict:
    """Analyze sentiment of a single text."""
    text = args.get("text", "")
    if len(text) < 20:
        return {"error": "Text must be at least 20 characters", "timestamp": datetime.now().isoformat()}
    
    result = sentiment_client.analyze_sentiment_with_evidence(text)
    return {
        "analysis": result,
        "transparency_note": "Sentence-level evidence provided.",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting CallDelta MCP Server on port {port}")
    print(f"MCP endpoint: http://0.0.0.0:{port}/mcp")
    print(f"Health check: http://0.0.0.0:{port}/health")
    print("Features: HTTP endpoint, outputSchema, _meta, Context auth middleware")
    uvicorn.run(app, host="0.0.0.0", port=port)
