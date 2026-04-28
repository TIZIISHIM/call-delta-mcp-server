

import os
import json
from datetime import datetime
from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse, Response
from ctxprotocol import create_context_middleware
import uvicorn

from transcript_fetcher import TranscriptFetcher
from huggingface_client import HuggingFaceClient

# Initialize
app = FastAPI(title="CallDelta MCP Server")
fetcher = TranscriptFetcher()
sentiment_client = HuggingFaceClient()

# Create auth middleware
verify_context = create_context_middleware(
    audience="https://calldelta-mcp-server-production.up.railway.app"
)

# Define tools
TOOLS = [
    {
        "name": "compare_earnings_calls",
        "description": "Compare two earnings call transcripts and return sentiment delta with sentence-level evidence.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "current_year": {"type": "integer"},
                "current_quarter": {"type": "integer"},
                "previous_year": {"type": "integer"},
                "previous_quarter": {"type": "integer"}
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
                "text": {"type": "string"}
            },
            "required": ["text"]
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "analysis": {"type": "object"},
                "transparency_note": {"type": "string"},
                "timestamp": {"type": "string"}
            }
        },
        "_meta": {"surface": "query", "queryEligible": True}
    }
]


@app.get("/health")
async def health():
    return {"status": "alive", "timestamp": datetime.now().isoformat()}


@app.api_route("/", methods=["GET", "POST"])
async def root(request: Request, context: dict = Depends(verify_context)):
    """Handle both GET (SSE connection) and POST (JSON-RPC) at the root endpoint."""
    
    if request.method == "GET":
        # SSE connection - keep alive
        return Response(
            content="event: endpoint\ndata: /\n\n",
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*"
            }
        )
    
    elif request.method == "POST":
        try:
            body = await request.json()
        except:
            return JSONResponse(status_code=400, content={"error": "Invalid JSON"})
        
        method = body.get("method")
        msg_id = body.get("id")
        params = body.get("params", {})
        
        print(f"Method: {method}, ID: {msg_id}")
        
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "calldelta", "version": "1.0.0"}
                }
            }
        
        if method == "notifications/initialized":
            return Response(status_code=202)
        
        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": TOOLS}
            }
        
        if method == "tools/call":
            tool_name = params.get("name")
            args = params.get("arguments", {})
            
            if tool_name == "compare_earnings_calls":
                result = await compare_earnings_calls(args)
            elif tool_name == "analyze_sentiment":
                result = await analyze_sentiment(args)
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Tool not found: {tool_name}"}
                }
            
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2)}]
                }
            }
        
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"}
        }


async def compare_earnings_calls(args):
    ticker = args.get("ticker", "").upper()
    current_year = args.get("current_year")
    current_quarter = args.get("current_quarter")
    previous_year = args.get("previous_year")
    previous_quarter = args.get("previous_quarter")
    
    if not ticker:
        return {"error": "Ticker is required"}
    
    current = fetcher.fetch_transcript(ticker, current_year, current_quarter)
    if current.get('status') == 'error':
        return {"error": f"Failed to fetch transcript for {ticker} Q{current_quarter} {current_year}"}
    
    previous = fetcher.fetch_transcript(ticker, previous_year, previous_quarter)
    if previous.get('status') == 'error':
        return {"error": f"Failed to fetch transcript for {ticker} Q{previous_quarter} {previous_year}"}
    
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


async def analyze_sentiment(args):
    text = args.get("text", "")
    if len(text) < 20:
        return {"error": "Text must be at least 20 characters"}
    
    result = sentiment_client.analyze_sentiment_with_evidence(text)
    return {
        "analysis": result,
        "transparency_note": "Sentence-level evidence provided.",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting CallDelta MCP Server on port {port}")
    print(f"Root endpoint: http://0.0.0.0:{port}/")
    uvicorn.run(app, host="0.0.0.0", port=port)
