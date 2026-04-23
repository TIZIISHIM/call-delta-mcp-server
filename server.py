

import os
import json
import asyncio
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from sse_starlette.sse import EventSourceResponse
import uvicorn

from transcript_fetcher import TranscriptFetcher
from huggingface_client import HuggingFaceClient

app = FastAPI(title="CallDelta MCP Server")
fetcher = TranscriptFetcher()
sentiment_client = HuggingFaceClient()

# Store active connections (simplified)
active_connections = {}


@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "CallDelta MCP Server",
        "version": "4.0.0",
        "features": ["fallback_chain", "transparent_materiality", "sentence_level_evidence", "mcp_sse"],
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health():
    return {"status": "alive", "timestamp": datetime.now().isoformat()}


@app.get("/sse")
async def sse_endpoint(request: Request):
    """SSE endpoint for MCP protocol."""
    
    async def event_generator():
        session_id = datetime.now().timestamp()
        
        # Send initial endpoint event
        yield {
            "event": "endpoint",
            "data": f"/messages?sessionId={session_id}"
        }
        
        # Keep connection alive
        while await request.is_disconnected() == False:
            await asyncio.sleep(1)
            yield {
                "event": "ping",
                "data": ""
            }
    
    return EventSourceResponse(event_generator())


@app.post("/messages")
async def messages_endpoint(request: Request):
    """Handle MCP messages."""
    try:
        body = await request.json()
    except:
        body = {}
    
    method = body.get("method", "")
    params = body.get("params", {})
    msg_id = body.get("id")
    
    if method == "listTools":
        # Return list of available tools
        result = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "tools": [
                    {
                        "name": "compare_earnings_calls",
                        "description": "Compare two earnings call transcripts (current quarter vs previous quarter) and return structured delta showing changes in management tone, guidance language, and topic-specific sentiment shifts. Every claim includes exact source sentences, sentiment scores, and confidence.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "ticker": {
                                    "type": "string",
                                    "description": "Stock ticker symbol (e.g., 'NVDA', 'TSLA', 'AAPL')"
                                },
                                "current_year": {
                                    "type": "integer",
                                    "description": "Year of the current earnings call"
                                },
                                "current_quarter": {
                                    "type": "integer",
                                    "description": "Quarter number (1, 2, 3, or 4)"
                                },
                                "previous_year": {
                                    "type": "integer",
                                    "description": "Year of the previous earnings call"
                                },
                                "previous_quarter": {
                                    "type": "integer",
                                    "description": "Quarter number (1, 2, 3, or 4)"
                                }
                            },
                            "required": ["ticker", "current_year", "current_quarter", "previous_year", "previous_quarter"]
                        }
                    },
                    {
                        "name": "analyze_sentiment",
                        "description": "Analyze sentiment of an earnings call transcript or text passage. Returns sentence-level evidence with scores and confidence.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "Text passage to analyze"
                                }
                            },
                            "required": ["text"]
                        }
                    }
                ]
            }
        }
        return JSONResponse(content=result)
    
    elif method == "callTool":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        
        if tool_name == "compare_earnings_calls":
            result = await compare_earnings_calls(arguments)
            return JSONResponse(content={
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2)
                        }
                    ]
                }
            })
        
        elif tool_name == "analyze_sentiment":
            result = await analyze_sentiment(arguments)
            return JSONResponse(content={
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2)
                        }
                    ]
                }
            })
        
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32601,
                        "message": f"Tool not found: {tool_name}"
                    }
                }
            )
    
    else:
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }
        )


@app.post("/call")
async def call_endpoint(request: Request):
    """Legacy endpoint - kept for backward compatibility."""
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
    
    current = fetcher.fetch_transcript(ticker, current_year, current_quarter)
    
    if current['status'] == 'error':
        return {
            "error": "Failed to fetch current transcript",
            "source_error": current,
            "user_action": f"Transcript for {ticker} Q{current_quarter} {current_year} is not available."
        }
    
    previous = fetcher.fetch_transcript(ticker, previous_year, previous_quarter)
    
    if previous['status'] == 'error':
        return {
            "error": "Failed to fetch previous transcript",
            "source_error": previous,
            "user_action": f"Transcript for {ticker} Q{previous_quarter} {previous_year} is not available."
        }
    
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
        "transparency_note": "All sentiment claims are backed by exact sentence-level evidence.",
        "query_timestamp": datetime.now().isoformat()
    }


async def analyze_sentiment(args: dict) -> dict:
    text = args.get("text", "")
    
    if not text or len(text) < 20:
        return {
            "error": "Text is required and must be at least 20 characters",
            "user_action": "Provide an earnings call transcript excerpt or full text"
        }
    
    result = sentiment_client.analyze_sentiment_with_evidence(text)
    
    return {
        "tool": "analyze_sentiment",
        "analysis": result,
        "text_preview": text[:200] + "..." if len(text) > 200 else text,
        "query_timestamp": datetime.now().isoformat(),
        "transparency_note": "Sentiment analysis performed with sentence-level evidence."
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting CallDelta MCP Server on port {port}")
    print(f"SSE endpoint: http://0.0.0.0:{port}/sse")
    print(f"Messages endpoint: http://0.0.0.0:{port}/messages")
    uvicorn.run(app, host="0.0.0.0", port=port)
