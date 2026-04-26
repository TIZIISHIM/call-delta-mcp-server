

import os
import json
from datetime import datetime
from fastapi import FastAPI, Request, Response, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv

from transcript_fetcher import TranscriptFetcher
from huggingface_client import HuggingFaceClient

load_dotenv()

app = FastAPI(title="CallDelta MCP Server")
fetcher = TranscriptFetcher()
sentiment_client = HuggingFaceClient()

PROTOCOL_VERSION = "2024-11-05"

# Auth setup for Context Protocol (optional for free tier)
security = HTTPBearer(auto_error=False)


def verify_context_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verify Context Protocol authentication.
    For free tools ($0), this is optional and always passes.
    For paid tools, implement proper JWT verification.
    """
    # Free tier - accept all requests
    # When ready to charge, implement proper verification using Context's public key
    return {"authenticated": True, "tier": "free"}


@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "CallDelta MCP Server",
        "version": "5.0.0",
        "features": ["fallback_chain", "transparent_materiality", "sentence_level_evidence", "ir_fallback_implemented", "context_auth_ready"],
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health():
    return {"status": "alive", "timestamp": datetime.now().isoformat()}


@app.post("/mcp")
async def mcp_endpoint(request: Request, auth: dict = Depends(verify_context_auth)):
    """Main MCP endpoint with Context auth integration."""
    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": "2.0",
                "error": {"code": -32700, "message": f"Parse error: {str(e)}"}
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
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "calldelta-mcp-server",
                    "version": "5.0.0"
                }
            }
        })
    
    # Initialized notification
    elif method == "notifications/initialized":
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
                                "ticker": {"type": "string", "description": "Stock ticker symbol (e.g., NVDA, TSLA, AAPL)"},
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
                                "sources": {
                                    "type": "object",
                                    "properties": {
                                        "current": {"type": "object", "properties": {"source": {"type": "string"}, "url": {"type": "string"}}},
                                        "previous": {"type": "object", "properties": {"source": {"type": "string"}, "url": {"type": "string"}}}
                                    }
                                },
                                "sentiment_analysis": {
                                    "type": "object",
                                    "properties": {
                                        "overall_delta": {
                                            "type": "object",
                                            "properties": {
                                                "current": {"type": "number"},
                                                "previous": {"type": "number"},
                                                "delta": {"type": "number"},
                                                "direction": {"type": "string"},
                                                "materiality": {"type": "string"}
                                            }
                                        },
                                        "current_evidence": {"type": "array"},
                                        "previous_evidence": {"type": "array"},
                                        "methodology": {"type": "object"}
                                    }
                                },
                                "transparency_note": {"type": "string"},
                                "timestamp": {"type": "string"}
                            }
                        }
                    },
                    {
                        "name": "analyze_sentiment",
                        "description": "Analyze sentiment of text with sentence-level evidence.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string", "description": "Text to analyze"}
                            },
                            "required": ["text"]
                        },
                        "outputSchema": {
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
                ],
                "structuredContent": result
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
    """Compare two earnings calls and return sentiment delta with evidence."""
    ticker = args.get("ticker", "").upper()
    current_year = args.get("current_year")
    current_quarter = args.get("current_quarter")
    previous_year = args.get("previous_year")
    previous_quarter = args.get("previous_quarter")
    
    if not ticker:
        return {"error": "Ticker is required"}
    
    if not all([current_year, current_quarter, previous_year, previous_quarter]):
        return {"error": "Year and quarter fields are required"}
    
    # Fetch current transcript
    current = fetcher.fetch_transcript(ticker, current_year, current_quarter)
    
    if current.get('status') == 'error':
        return {
            "error": f"Failed to fetch current transcript for {ticker} Q{current_quarter} {current_year}",
            "details": current,
            "suggestion": "Try a different ticker or quarter. For example: NVDA Q3 2024 vs Q2 2024, AAPL Q4 2024, TSLA Q2 2024"
        }
    
    # Fetch previous transcript
    previous = fetcher.fetch_transcript(ticker, previous_year, previous_quarter)
    
    if previous.get('status') == 'error':
        return {
            "error": f"Failed to fetch previous transcript for {ticker} Q{previous_quarter} {previous_year}",
            "details": previous,
            "suggestion": "Try a different ticker or quarter. For example: NVDA Q2 2024, AAPL Q3 2024, TSLA Q1 2024"
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
        "transparency_note": "All sentiment claims are backed by exact sentence-level evidence. See current_evidence and previous_evidence arrays for exact sentences and their scores.",
        "timestamp": datetime.now().isoformat()
    }


async def analyze_sentiment(args: dict) -> dict:
    """Analyze sentiment of a single text passage."""
    text = args.get("text", "")
    
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting CallDelta MCP Server on port {port}")
    print(f"MCP endpoint: http://0.0.0.0:{port}/mcp")
    print(f"Health check: http://0.0.0.0:{port}/health")
    print("Features: real transcript extraction, IR fallback, real sentiment scores, Context auth ready")
    uvicorn.run(app, host="0.0.0.0", port=port)
