import os
import json
import asyncio
import requests
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from sse_starlette.sse import EventSourceResponse
import uvicorn

from transcript_fetcher import TranscriptFetcher
from huggingface_client import HuggingFaceClient
from ctxprotocol import verify_context_request

# Initialize FastAPI app
app = FastAPI(title="CallDelta MCP Server")

# Initialize fetchers
fetcher = TranscriptFetcher()
sentiment_client = HuggingFaceClient()

# Store active sessions
sessions = {}

# Get audience URL from environment variable
AUDIENCE_URL = os.environ.get("AUDIENCE_URL", "https://calldelta-mcp-server-production.up.railway.app")

# Define tools with both modes enabled
AVAILABLE_TOOLS = [
    {
        "name": "compare_earnings_calls",
        "description": "Use this tool when the user asks to compare earnings call transcripts between two specific quarters (e.g., Q3 vs Q2), determine changes in management tone on topics like revenue, margins, guidance, or competitive risk, or evaluate if management sounds more or less confident. Do not use for single text sentiment analysis.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol (e.g., NVDA, TSLA, AAPL)"},
                "current_year": {"type": "integer", "description": "Year of the current/most recent quarter"},
                "current_quarter": {"type": "integer", "description": "Current quarter (1=Q1, 2=Q2, 3=Q3, 4=Q4)"},
                "previous_year": {"type": "integer", "description": "Year of the previous quarter to compare against"},
                "previous_quarter": {"type": "integer", "description": "Previous quarter (1=Q1, 2=Q2, 3=Q3, 4=Q4)"}
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
        "description": "Use this tool to analyze sentiment of a single earnings call excerpt or text snippet. Returns sentence-level evidence with FinBERT (finance-optimized sentiment model). Do not use for quarter-over-quarter comparisons.",
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


@app.get("/debug/env")
async def debug_env():
    """Debug endpoint to verify environment variables are being read."""
    fmp_key = os.environ.get("FMP_API_KEY", "")
    apify_token = os.environ.get("APIFY_TOKEN", "")
    return {
        "fmp_key_set": bool(fmp_key),
        "fmp_key_preview": fmp_key[:5] + "..." if fmp_key else "not set",
        "apify_token_set": bool(apify_token),
        "apify_token_preview": apify_token[:10] + "..." if apify_token else "not set",
        "hf_token_set": bool(os.environ.get("HF_TOKEN", "")),
        "audience_url": os.environ.get("AUDIENCE_URL", "not set"),
    }


@app.get("/debug/yfinance")
async def debug_yfinance():
    """Debug endpoint to test YFinance functionality."""
    try:
        import yfinance as yf
        ticker = yf.Ticker("NVDA")
        transcripts = ticker.earnings_transcript
        return {
            "yfinance_installed": True,
            "transcripts_found": len(transcripts) if transcripts else 0,
            "first_transcript": transcripts[0] if transcripts else None,
            "transcript_keys": list(transcripts[0].keys()) if transcripts and len(transcripts) > 0 else []
        }
    except ImportError:
        return {"yfinance_installed": False, "error": "yfinance not installed. Run: pip install yfinance"}
    except Exception as e:
        return {"yfinance_installed": True, "error": str(e)}


@app.get("/debug/sentiment/test")
async def debug_sentiment_test():
    """Debug endpoint to test sentiment analysis with Alex's test inputs."""
    positive_input = "Our growth accelerated this quarter with record revenue and expanding gross margins across all segments. We are extremely confident in our outlook."
    negative_input = "Demand collapsed this quarter. Revenue fell 40 percent. Margins are under severe pressure and we are losing key customers to competitors."
    
    positive_result = sentiment_client.analyze_sentiment_with_evidence(positive_input)
    negative_result = sentiment_client.analyze_sentiment_with_evidence(negative_input)
    
    return {
        "positive_input": positive_input,
        "positive_result": {
            "sentiment_label": positive_result.get('sentiment_label'),
            "sentiment_score": positive_result.get('sentiment_score'),
            "confidence": positive_result.get('confidence'),
            "warning": positive_result.get('warning'),
            "source": positive_result.get('evidence', [{}])[0].get('source', 'unknown') if positive_result.get('evidence') else 'none'
        },
        "negative_input": negative_input,
        "negative_result": {
            "sentiment_label": negative_result.get('sentiment_label'),
            "sentiment_score": negative_result.get('sentiment_score'),
            "confidence": negative_result.get('confidence'),
            "warning": negative_result.get('warning'),
            "source": negative_result.get('evidence', [{}])[0].get('source', 'unknown') if negative_result.get('evidence') else 'none'
        },
        "test_passed": (
            positive_result.get('sentiment_label') == 'positive' and
            negative_result.get('sentiment_label') == 'negative' and
            positive_result.get('sentiment_score', 0) > 0.55 and
            negative_result.get('sentiment_score', 0) < 0.45
        )
    }

@app.get("/debug/hf_token")
async def debug_hf_token():
    token = os.environ.get("HF_TOKEN", "")
    return {
        "token_set": bool(token),
        "token_preview": token[:10] + "..." if token else "not set",
        "token_length": len(token) if token else 0
    }

@app.get("/debug/gradio_status")
async def debug_gradio_status():
    """Debug endpoint to check Gradio Space status."""
    gradio_url = os.environ.get("GRADIO_SPACE_URL", "")
    if not gradio_url:
        return {"error": "GRADIO_SPACE_URL not set in environment variables"}
    
    try:
        # Test if the Gradio Space is reachable
        response = requests.get(f"{gradio_url}/gradio_api/call/sentiment_analysis", timeout=10)
        return {
            "gradio_url": gradio_url,
            "status_code": response.status_code,
            "reachable": response.status_code < 500
        }
    except Exception as e:
        return {
            "gradio_url": gradio_url,
            "reachable": False,
            "error": str(e)
        }

@app.get("/debug/replicate_status")
async def debug_replicate_status():
    replicate_token = os.environ.get("REPLICATE_API_TOKEN", "")
    return {
        "replicate_token_set": bool(replicate_token),
        "replicate_token_preview": replicate_token[:10] + "..." if replicate_token else "not set",
        "hf_token_set": bool(os.environ.get("HF_TOKEN", "")),
        "gradio_url_set": bool(os.environ.get("GRADIO_SPACE_URL", ""))
    }

@app.get("/debug/gradio_test")
async def debug_gradio_test():
    gradio_url = os.environ.get("GRADIO_SPACE_URL", "")
    if not gradio_url:
        return {"error": "GRADIO_SPACE_URL not set"}
    
    test_text = "Our revenue grew 20% this quarter."
    
    # Try different API paths
    results = {}
    
    for path in ["/api/predict/sentiment_analysis", "/gradio_api/predict/sentiment_analysis", "/predict/sentiment_analysis"]:
        try:
            response = requests.post(
                f"{gradio_url}{path}",
                json={"data": [test_text]},
                timeout=10,
                headers={"Content-Type": "application/json"}
            )
            results[path] = {
                "status": response.status_code,
                "response": response.text[:200] if response.status_code == 200 else None
            }
        except Exception as e:
            results[path] = {"error": str(e)}
    
    return {
        "gradio_url": gradio_url,
        "results": results
    }

@app.get("/debug/gradio_raw_test")
async def debug_gradio_raw_test():
    """Direct test of Gradio Space API - shows raw response."""
    import requests
    
    gradio_url = os.environ.get("GRADIO_SPACE_URL", "")
    if not gradio_url:
        return {"error": "GRADIO_SPACE_URL not set"}
    
    test_text = "Our revenue grew 20% this quarter with strong margins."
    
    results = {}
    
    # Try the /api/sentiment endpoint
    try:
        response = requests.post(
            f"{gradio_url}/api/sentiment",
            json={"text": test_text},
            timeout=10,
            headers={"Content-Type": "application/json"}
        )
        results["/api/sentiment"] = {
            "status_code": response.status_code,
            "response_text": response.text[:500],
            "response_json": response.json() if response.headers.get('content-type', '').startswith('application/json') else None
        }
    except Exception as e:
        results["/api/sentiment"] = {"error": str(e)}
    
    # Try the root endpoint to see if Space is alive
    try:
        response = requests.get(gradio_url, timeout=10)
        results["root_get"] = {
            "status_code": response.status_code,
            "html_preview": response.text[:300]
        }
    except Exception as e:
        results["root_get"] = {"error": str(e)}
    
    return results
    
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
    
    elif method == "notifications/cancelled":
        print(f"Received cancellation for request ID: {params.get('requestId')}")
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
    
    elif method == "notifications/cancelled":
        print(f"MCP fallback - Received cancellation for request ID: {params.get('requestId')}")
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
