"""
Siddha RAG API Server
Flask-based REST API for the Siddha Medicine RAG System

Usage:
    python api_server.py                     # Default: Port 5001
    python api_server.py --port 8000         # Custom port
    python api_server.py --password <pw>     # Neo4j password for hybrid mode
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add src directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

# Import RAG components
from src.rag.rag_system import SiddhaRAG

# ============================================
# Flask App Configuration
# ============================================
app = Flask(__name__)
CORS(app)

# Global RAG instance
rag_system: Optional[SiddhaRAG] = None

# Chat storage (in-memory for demo, use DB for production)
chats: Dict[str, Dict] = {}


def initialize_rag(use_graph: bool = False, neo4j_password: str = None):
    """Initialize the RAG system with specified configuration."""
    global rag_system
    
    print("🔄 Initializing Siddha RAG System...")
    
    try:
        rag_system = SiddhaRAG(
            db_path="data/vectordb",
            collection_name="siddha_knowledge",
            use_graph=use_graph,
            neo4j_password=neo4j_password
        )
        mode = "Hybrid (Vector + Graph)" if use_graph else "Vector Only"
        print(f"✅ RAG System initialized in {mode} mode")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize RAG: {e}")
        return False


# ============================================
# API Routes
# ============================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "rag_initialized": rag_system is not None,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/query', methods=['POST'])
def query():
    """
    Query the RAG system.
    
    Request body:
    {
        "question": "What herbs treat fever?",
        "strategy": "hybrid",  // "vector", "graph", or "hybrid"
        "top_k": 5
    }
    """
    if rag_system is None:
        return jsonify({"error": "RAG system not initialized"}), 503
    
    try:
        data = request.get_json()
        question = data.get('question', '')
        strategy = data.get('strategy', 'hybrid')
        top_k = data.get('top_k', 5)
        
        if not question:
            return jsonify({"error": "Question is required"}), 400
        
        # Query the RAG system
        response = rag_system.query(
            question=question,
            strategy=strategy,
            top_k=top_k
        )
        
        return jsonify({
            "success": True,
            "question": question,
            "answer": response.get('answer', ''),
            "sources": response.get('sources', []),
            "entities_found": response.get('entities_found', []),
            "graph_facts": response.get('graph_facts', []),
            "strategy_used": response.get('strategy_used', strategy)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/query/stream', methods=['POST'])
def query_stream():
    """
    Stream response from RAG system using Server-Sent Events.
    
    Request body:
    {
        "question": "What herbs treat fever?",
        "strategy": "hybrid"
    }
    """
    if rag_system is None:
        return jsonify({"error": "RAG system not initialized"}), 503
    
    data = request.get_json()
    question = data.get('question', '')
    strategy = data.get('strategy', 'hybrid')
    
    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    def generate():
        try:
            response = rag_system.query(
                question=question,
                strategy=strategy
            )
            
            # Send the response in chunks (simulating streaming)
            answer = response.get('answer', '')
            words = answer.split()
            
            for i, word in enumerate(words):
                chunk = word + (' ' if i < len(words) - 1 else '')
                yield f"data: {json.dumps({'token': chunk})}\n\n"
            
            # Send completion with metadata
            yield f"data: {json.dumps({'done': True, 'sources': response.get('sources', [])})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/chats', methods=['GET'])
def list_chats():
    """List all chat sessions."""
    chat_list = [
        {
            "id": chat_id,
            "title": chat_data.get('title', 'Untitled Chat'),
            "created_at": chat_data.get('created_at'),
            "message_count": len(chat_data.get('messages', []))
        }
        for chat_id, chat_data in chats.items()
    ]
    return jsonify({"chats": chat_list})


@app.route('/api/chats', methods=['POST'])
def create_chat():
    """Create a new chat session."""
    import uuid
    chat_id = str(uuid.uuid4())
    
    data = request.get_json() or {}
    chats[chat_id] = {
        "title": data.get('title', 'New Chat'),
        "created_at": datetime.now().isoformat(),
        "messages": []
    }
    
    return jsonify({"chat_id": chat_id, "success": True}), 201


@app.route('/api/chats/<chat_id>', methods=['GET'])
def get_chat(chat_id: str):
    """Get a specific chat session with messages."""
    if chat_id not in chats:
        return jsonify({"error": "Chat not found"}), 404
    
    return jsonify(chats[chat_id])


@app.route('/api/chats/<chat_id>/messages', methods=['POST'])
def add_message(chat_id: str):
    """Add a message to a chat and get RAG response."""
    if chat_id not in chats:
        return jsonify({"error": "Chat not found"}), 404
    
    if rag_system is None:
        return jsonify({"error": "RAG system not initialized"}), 503
    
    data = request.get_json()
    question = data.get('content', '')
    
    if not question:
        return jsonify({"error": "Message content required"}), 400
    
    # Add user message
    user_message = {
        "role": "user",
        "content": question,
        "timestamp": datetime.now().isoformat()
    }
    chats[chat_id]['messages'].append(user_message)
    
    # Get RAG response
    try:
        response = rag_system.query(question=question, strategy='hybrid')
        
        assistant_message = {
            "role": "assistant",
            "content": response.get('answer', ''),
            "sources": response.get('sources', []),
            "timestamp": datetime.now().isoformat()
        }
        chats[chat_id]['messages'].append(assistant_message)
        
        # Update chat title if it's the first message
        if len(chats[chat_id]['messages']) == 2:
            chats[chat_id]['title'] = question[:50] + ('...' if len(question) > 50 else '')
        
        return jsonify({
            "user_message": user_message,
            "assistant_message": assistant_message
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/suggest', methods=['POST'])
def suggest_questions():
    """Get suggested follow-up questions based on context."""
    data = request.get_json() or {}
    context = data.get('context', '')
    
    # Simple suggestions based on Siddha medicine topics
    suggestions = [
        "What are the common herbs used in Siddha medicine?",
        "How does Siddha treat digestive disorders?",
        "What is the Siddha approach to fever treatment?",
        "Tell me about Vatha, Pitha, and Kapha in Siddha",
        "What are traditional Siddha remedies for respiratory issues?"
    ]
    
    return jsonify({"suggestions": suggestions[:3]})


# ============================================
# Main Entry Point
# ============================================

def main():
    """Main entry point for the API server."""
    parser = argparse.ArgumentParser(
        description="Siddha RAG API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=5001,
        help='Port to run the server on (default: 5001)'
    )
    parser.add_argument(
        '--password',
        help='Neo4j password for hybrid mode'
    )
    parser.add_argument(
        '--no-graph',
        action='store_true',
        help='Disable graph retrieval (vector-only mode)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🩺 Siddha Medicine RAG - API Server")
    print("=" * 60)
    
    # Get Neo4j password from args or environment
    neo4j_password = args.password or os.getenv('NEO4J_PASSWORD')
    use_graph = not args.no_graph and neo4j_password is not None
    
    # Initialize RAG system
    if not initialize_rag(use_graph=use_graph, neo4j_password=neo4j_password):
        print("⚠️ Starting server with RAG disabled. Some endpoints will return 503.")
    
    print(f"\n🚀 Starting API server on port {args.port}")
    print(f"📡 API available at: http://localhost:{args.port}")
    print(f"📋 Health check: http://localhost:{args.port}/api/health")
    print("\nPress Ctrl+C to stop the server.\n")
    
    app.run(
        host='0.0.0.0',
        port=args.port,
        debug=args.debug
    )


if __name__ == '__main__':
    main()
