"""
CallDelta MCP Server - Earnings Call Transcript Delta Intelligence

"""

import json
import os
from datetime import datetime
from typing import Dict

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.streamable_http import StreamableHTTPServerTransport
import mcp.types as types

from transcript_fetcher import TranscriptFetcher
from sentiment_analyzer import TransparentSentimentAnalyzer
from delta_detector import TranscriptDeltaDetector

# Initialize server
server = Server("calldelta-mcp-server")

# Initialize components
fetcher = TranscriptFetcher()
sentiment_analyzer = TransparentSentimentAnalyzer()
delta_detector = TranscriptDeltaDetector()


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools."""
    return [
        types.Tool(
            name="compare_earnings_calls",
            description="Compare two earnings call transcripts (current quarter vs previous quarter) and return structured delta showing changes in management tone, guidance language, and topic-specific sentiment shifts. Every claim includes exact source sentences, FinBERT output, and confidence scores. No black-box verdicts.",
            inputSchema={
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
                        "description": "Quarter number of the current earnings call (1, 2, 3, or 4)"
                    },
                    "previous_year": {
                        "type": "integer",
                        "description": "Year of the previous earnings call"
                    },
                    "previous_quarter": {
                        "type": "integer",
                        "description": "Quarter number of the previous earnings call (1, 2, 3, or 4)"
                    }
                },
                "required": ["ticker", "current_year", "current_quarter", "previous_year", "previous_quarter"]
            }
        ),
        types.Tool(
            name="analyze_sentiment",
            description="Analyze sentiment of an earnings call transcript or text passage. Returns transparent results with sentence-level evidence, FinBERT output, and confidence scores.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text passage to analyze"
                    }
                },
                "required": ["text"]
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    """Handle tool execution."""
    
    if name == "compare_earnings_calls":
        ticker = arguments.get("ticker", "").upper()
        current_year = arguments.get("current_year")
        current_quarter = arguments.get("current_quarter")
        previous_year = arguments.get("previous_year")
        previous_quarter = arguments.get("previous_quarter")
        
        if not ticker:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": "Ticker is required"}, indent=2)
            )]
        
        current_result = fetcher.fetch_transcript(ticker, current_year, current_quarter)
        
        if current_result['status'] == 'error':
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "error": "Failed to fetch transcript",
                    "details": current_result
                }, indent=2)
            )]
        
        previous_result = fetcher.fetch_transcript(ticker, previous_year, previous_quarter)
        
        if previous_result['status'] == 'error':
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "error": "Failed to fetch previous transcript",
                    "details": previous_result
                }, indent=2)
            )]
        
        sentiment_comparison = sentiment_analyzer.compare_transcripts(
            current_result['content'],
            previous_result['content']
        )
        
        text_changes = delta_detector.detect_changes(
            current_result['content'],
            previous_result['content']
        )
        
        response = {
            'tool': 'compare_earnings_calls',
            'ticker': ticker,
            'current_quarter': f"Q{current_quarter} {current_year}",
            'previous_quarter': f"Q{previous_quarter} {previous_year}",
            'sources': {
                'current': {
                    'source': current_result['source'],
                    'url': current_result.get('url'),
                    'fetched_at': current_result.get('timestamp')
                },
                'previous': {
                    'source': previous_result['source'],
                    'url': previous_result.get('url'),
                    'fetched_at': previous_result.get('timestamp')
                }
            },
            'sentiment_analysis': sentiment_comparison,
            'text_changes': text_changes,
            'query_timestamp': datetime.now().isoformat()
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(response, indent=2)
        )]
    
    elif name == "analyze_sentiment":
        text = arguments.get("text", "")
        
        if not text or len(text) < 50:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "error": "Text is required and must be at least 50 characters"
                }, indent=2)
            )]
        
        analysis = sentiment_analyzer.analyze_transcript(text)
        
        response = {
            'tool': 'analyze_sentiment',
            'analysis': analysis,
            'query_timestamp': datetime.now().isoformat()
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(response, indent=2)
        )]
    
    else:
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": f"Unknown tool: {name}"}, indent=2)
        )]


async def main():
    """Run the MCP server with Streamable HTTP transport."""
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get("PORT", 10000))
    
    # Create HTTP transport
    transport = StreamableHTTPServerTransport(server, port=port)
    
    print(f"Starting CallDelta MCP Server on port {port}")
    
    # Start the server
    await transport.start()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
