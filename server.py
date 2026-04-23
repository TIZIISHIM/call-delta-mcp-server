

import os
import json
import asyncio
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from transcript_fetcher import TranscriptFetcher
from huggingface_client import HuggingFaceClient

app = FastAPI(title="CallDelta MCP Server")
fetcher = TranscriptFetcher()
sentiment_client = HuggingFaceClient()

PROTOCOL_VERSION = "2024-11-05"


@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "CallDelta MCP Server",
        "version": "4.0.0",
        "features": ["fallback_chain", "transparent_materiality", "sentence_level_evidence"],
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health():
    return {"status": "alive", "timestamp": datetime.now().isoformat()}


@app.post("/mcp")
async def mcp_endpoint(request: Request):
    """
    HTTP Streaming endpoint for MCP protocol.
    Context Protocol will POST JSON-RPC messages here.
    """
    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": "2.0",
                "error": {
                    "code": -32700,
                    "message": f"Parse error: {str(e)}"
                }
            }
        )
    
    method = body.get("method", "")
    params = body.get("params", {})
    msg_id = body.get("id")
    
    print(f"Received: {method} (id: {msg_id})")
    
    # Initialize handshake
    if method == "initialize":
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "calldelta-mcp-server",
                    "version": "4.0.0"
                }
            }
        })
    
    # Initialized notification
    elif method == "initialized":
        return Response(status_code=202)
    
    # List tools
    elif method == "tools/list":
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "tools": [
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
                        }
                    },
                    {
                        "name": "analyze_sentiment",
                        "description": "Analyze sentiment of text with sentence-level evidence.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"}
                            },
                            "required": ["text"]
                        }
                    }
                ]
            }
        })
    
    # Call tool
    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        
        if tool_name == "compare_earnings_calls":
            result = await compare_earnings_calls(arguments)
        elif tool_name == "analyze_sentiment":
            result = await analyze_sentiment(arguments)
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Tool not found: {tool_name}"}
                }
            )
        
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "content": [
                    {"type": "text", "text": json.dumps(result, indent=2)}
                ]
            }
        })
    
    # Unknown method
    else:
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"}
            }
        )


async def compare_earnings_calls(args: dict) -> dict:
    ticker = args.get("ticker", "").upper()
    current_year = args.get("current_year")
    current_quarter = args.get("current_quarter")
    previous_year = args.get("previous_year")
    previous_quarter = args.get("previous_quarter")
    
    if not ticker:
        return {"error": "Ticker is required"}
    
    current = fetcher.fetch_transcript(ticker, current_year, current_quarter)
    previous = fetcher.fetch_transcript(ticker, previous_year, previous_quarter)
    
    if current.get('status') == 'error':
        return {"error": "Failed to fetch current transcript", "details": current}
    
    if previous.get('status') == 'error':
        return {"error": "Failed to fetch previous transcript", "details": previous}
    
    comparison = sentiment_client.compare_with_evidence(
        current.get('content', ''),
        previous.get('content', '')
    )
    
    return {
        "ticker": ticker,
        "current_quarter": f"Q{current_quarter} {current_year}",
        "previous_quarter": f"Q{previous_quarter} {previous_year}",
        "sources": {
            "current": {"source": current.get('source_used'), "url": current.get('url')},
            "previous": {"source": previous.get('source_used'), "url": previous.get('url')}
        },
        "sentiment_analysis": comparison,
        "transparency_note": "All claims backed by sentence-level evidence.",
        "timestamp": datetime.now().isoformat()
    }


async def analyze_sentiment(args: dict) -> dict:
    text = args.get("text", "")
    if len(text) < 20:
        return {"error": "Text must be at least 20 characters"}
    
    result = sentiment_client.analyze_sentiment_with_evidence(text)
    return {
        "analysis": result,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting CallDelta MCP Server on port {port}")
    print(f"MCP endpoint: http://0.0.0.0:{port}/mcp")
    uvicorn.run(app, host="0.0.0.0", port=port)
