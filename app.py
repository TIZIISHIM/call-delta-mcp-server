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
from ctxprotocol import verify_context_request

app = FastAPI(title="CallDelta MCP Server")

fetcher = TranscriptFetcher()
sentiment_client = HuggingFaceClient()

sessions = {}

AUDIENCE_URL = os.environ.get("AUDIENCE_URL", "https://calldelta-mcp-server-production.up.railway.app")

AVAILABLE_TOOLS = [
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
        "_meta": {
            "surface": "both",
            "queryEligible": True,
            "executeEligible": True
        }
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
        "_meta": {
            "surface": "both",
            "queryEligible": True,
            "executeEligible": True
        }
    }
]


@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "CallDelta MCP Server",
        "version": "17.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health():
    return {"status": "alive", "timestamp": datetime.now().isoformat()}


@app.get("/sse")
async def sse_endpoint(request: Request):
    session_id = os.urandom(16).hex()
    sessions[session_id] = {"messages": []}
    
    async def event_generator():
        yield {
            "event": "endpoint",
            "data": f"/messages?session_id={session_id}"
        }
        
        while True:
            await asyncio.sleep(30)
            yield {"event": "ping", "data": ""}
            if await request.is_disconnected():
                break
    
    return EventSourceResponse(event_generator())


@app.post("/messages")
async def messages_endpoint(request: Request):
    session_id = request.query_params.get("session_id")
    
    try:
        body = await request.json()
        print(f"Messages endpoint - Method: {body.get('method')}, ID: {body.get('id')}, Session: {session_id}")
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"jsonrpc": "2.0", "error": {"code": -32700, "message": f"Parse error: {str(e)}"}}
        )
    
    method = body.get("method", "")
    msg_id = body.get("id")
    params = body.get("params", {})
    
    if method == "initialize":
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "calldelta-mcp-server", "version": "17.0.0"}
            }
        })
    
    elif method == "notifications/initialized":
        return Response(status_code=202)
    
    elif method == "tools/list":
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"tools": AVAILABLE_TOOLS}
        })
    
    elif method == "tools/call":
        auth_header = request.headers.get("authorization", "")
        try:
            payload = await verify_context_request(
                authorization_header=auth_header,
                audience=AUDIENCE_URL
            )
            print(f"Auth successful for tool call: {payload.get('sub', 'unknown')}")
        except Exception as auth_error:
            print(f"Auth failed: {str(auth_error)}")
            return JSONResponse(
                status_code=401,
                content={
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32000, "message": f"Unauthorized: {str(auth_error)}"}
                }
            )
        
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        
        print(f"Executing tool: {tool_name}")
        print(f"Arguments: {json.dumps(arguments, indent=2)}")
        
        if tool_name == "compare_earnings_calls":
            result = await compare_earnings_calls(arguments)
        elif tool_name == "analyze_sentiment":
            result = await analyze_sentiment(arguments)
        else:
            return JSONResponse(
                status_code=400,
                content={"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"Tool not found: {tool_name}"}}
            )
        
        print("Tool execution complete, sending response")
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
        })
    
    else:
        print(f"Unknown method: {method}")
        return JSONResponse(
            status_code=400,
            content={"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}
        )


@app.post("/mcp")
async def mcp_fallback(request: Request):
    try:
        body = await request.json()
        print(f"MCP fallback - Method: {body.get('method')}, ID: {body.get('id')}")
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"jsonrpc": "2.0", "error": {"code": -32700, "message": f"Parse error: {str(e)}"}}
        )
    
    method = body.get("method", "")
    msg_id = body.get("id")
    params = body.get("params", {})
    
    if method == "initialize":
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "calldelta-mcp-server", "version": "17.0.0"}
            }
        })
    
    elif method == "notifications/initialized":
        return Response(status_code=202)
    
    elif method == "tools/list":
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"tools": AVAILABLE_TOOLS}
        })
    
    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        
        print(f"MCP fallback - Executing tool: {tool_name}")
        print(f"Arguments: {json.dumps(arguments, indent=2)}")
        
        if tool_name == "compare_earnings_calls":
            result = await compare_earnings_calls(arguments)
        elif tool_name == "analyze_sentiment":
            result = await analyze_sentiment(arguments)
        else:
            return JSONResponse(
                status_code=400,
                content={"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"Tool not found: {tool_name}"}}
            )
        
        print("MCP fallback - Tool execution complete")
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
        })
    
    else:
        print(f"MCP fallback - Unknown method: {method}")
        return JSONResponse(
            status_code=400,
            content={"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}
        )


@app.get("/mcp")
async def mcp_get_fallback():
    return JSONResponse(
        status_code=405,
        content={"error": "Method Not Allowed", "message": "Use POST for MCP requests"}
    )


async def compare_earnings_calls(args: dict) -> dict:
    ticker = args.get("ticker", "").upper()
    current_year = args.get("current_year")
    current_quarter = args.get("current_quarter")
    previous_year = args.get("previous_year")
    previous_quarter = args.get("previous_quarter")
    
    if not ticker:
        return {"error": "Ticker is required", "timestamp": datetime.now().isoformat()}
    
    print(f"Fetching {ticker} Q{current_quarter} {current_year} and Q{previous_quarter} {previous_year}")
    
    current = fetcher.fetch_transcript(ticker, current_year, current_quarter)
    if current.get('status') == 'error':
        return {"error": f"Failed to fetch transcript for {ticker} Q{current_quarter} {current_year}", "details": current, "timestamp": datetime.now().isoformat()}
    
    previous = fetcher.fetch_transcript(ticker, previous_year, previous_quarter)
    if previous.get('status') == 'error':
        return {"error": f"Failed to fetch transcript for {ticker} Q{previous_quarter} {previous_year}", "details": previous, "timestamp": datetime.now().isoformat()}
    
    print("Comparing transcripts...")
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
        "transparency_note": "All claims backed by sentence-level evidence from actual transcript excerpts using FinBERT (finance-optimized sentiment model).",
        "timestamp": datetime.now().isoformat()
    }


async def analyze_sentiment(args: dict) -> dict:
    text = args.get("text", "")
    if len(text) < 20:
        return {"error": "Text must be at least 20 characters", "timestamp": datetime.now().isoformat()}
    
    result = sentiment_client.analyze_sentiment_with_evidence(text)
    return {
        "analysis": result,
        "transparency_note": "Sentence-level evidence provided using FinBERT (finance-optimized sentiment model).",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting CallDelta MCP Server on port {port}")
    print(f"SSE endpoint: http://0.0.0.0:{port}/sse")
    print(f"Messages endpoint: http://0.0.0.0:{port}/messages")
    print(f"Health check: http://0.0.0.0:{port}/health")
    uvicorn.run(app, host="0.0.0.0", port=port)
