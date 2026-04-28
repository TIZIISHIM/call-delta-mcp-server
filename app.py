
import os
import asyncio
import json
from datetime import datetime
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response
from starlette.routing import Route

from transcript_fetcher import TranscriptFetcher
from huggingface_client import HuggingFaceClient

# Initialize
fetcher = TranscriptFetcher()
sentiment_client = HuggingFaceClient()

# Create MCP server
server = Server("calldelta-mcp-server")

# Create FastAPI app
app = FastAPI(title="CallDelta MCP Server")

# Define tools with FULL outputSchema and _meta
TOOLS = [
    Tool(
        name="compare_earnings_calls",
        description="Compare two earnings call transcripts and return sentiment delta with sentence-level evidence.",
        inputSchema={
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"},
                "current_year": {"type": "integer", "description": "Year of current earnings call"},
                "current_quarter": {"type": "integer", "description": "Quarter number (1-4)"},
                "previous_year": {"type": "integer", "description": "Year of previous earnings call"},
                "previous_quarter": {"type": "integer", "description": "Quarter number (1-4)"}
            },
            "required": ["ticker", "current_year", "current_quarter", "previous_year", "previous_quarter"]
        },
        outputSchema={
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
        _meta={
            "surface": "query",
            "queryEligible": True
        }
    ),
    Tool(
        name="analyze_sentiment",
        description="Analyze sentiment of earnings call text with sentence-level evidence.",
        inputSchema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to analyze"}
            },
            "required": ["text"]
        },
        outputSchema={
            "type": "object",
            "properties": {
                "analysis": {"type": "object"},
                "transparency_note": {"type": "string"},
                "timestamp": {"type": "string"}
            }
        },
        _meta={
            "surface": "query",
            "queryEligible": True
        }
    )
]


@server.list_tools()
async def handle_list_tools():
    """Return the list of tools."""
    return TOOLS


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict):
    """Handle tool execution."""
    
    if name == "compare_earnings_calls":
        ticker = arguments.get("ticker", "").upper()
        current_year = arguments.get("current_year")
        current_quarter = arguments.get("current_quarter")
        previous_year = arguments.get("previous_year")
        previous_quarter = arguments.get("previous_quarter")
        
        current = fetcher.fetch_transcript(ticker, current_year, current_quarter)
        if current.get('status') == 'error':
            return [TextContent(type="text", text=json.dumps({"error": "Failed to fetch current transcript", "details": current}))]
        
        previous = fetcher.fetch_transcript(ticker, previous_year, previous_quarter)
        if previous.get('status') == 'error':
            return [TextContent(type="text", text=json.dumps({"error": "Failed to fetch previous transcript", "details": previous}))]
        
        comparison = sentiment_client.compare_with_evidence(
            current.get('content', ''),
            previous.get('content', '')
        )
        
        result = {
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
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    elif name == "analyze_sentiment":
        text = arguments.get("text", "")
        if len(text) < 20:
            return [TextContent(type="text", text=json.dumps({"error": "Text must be at least 20 characters"}))]
        
        result = sentiment_client.analyze_sentiment_with_evidence(text)
        output = {
            "analysis": result,
            "transparency_note": "Sentence-level evidence provided.",
            "timestamp": datetime.now().isoformat()
        }
        return [TextContent(type="text", text=json.dumps(output, indent=2))]
    
    else:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


# Create SSE transport
sse = SseServerTransport("/messages")


@app.get("/sse")
async def handle_sse(request: Request):
    """SSE endpoint for MCP."""
    async with sse.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await server.run(
            streams[0], streams[1], server.create_initialization_options()
        )
    return Response()


@app.post("/messages")
async def handle_messages(request: Request):
    """Messages endpoint for MCP."""
    await sse.handle_post_message(request)
    return Response()


@app.get("/health")
async def health():
    return {"status": "alive", "timestamp": datetime.now().isoformat()}


@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "CallDelta MCP Server",
        "version": "7.0.0",
        "features": ["sse_transport", "outputSchema", "_meta", "fmp_api"],
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting CallDelta MCP Server on port {port}")
    print(f"SSE endpoint: http://0.0.0.0:{port}/sse")
    print(f"Health check: http://0.0.0.0:{port}/health")
    uvicorn.run(app, host="0.0.0.0", port=port)
