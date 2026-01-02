"""
Siddha RAG System - Unified Launcher with Mode Selection
Supports three modes: Vector, Graph, and Hybrid

Usage:
    python start_rag.py                  # Interactive mode selection
    python start_rag.py --mode vector    # Vector-only mode
    python start_rag.py --mode graph     # Graph-only mode  
    python start_rag.py --mode hybrid    # Hybrid mode (recommended)
"""

import sys
import os
import argparse
from pathlib import Path

# Add src directory to Python path
SCRIPT_DIR = Path(__file__).resolve().parent  # This is src/rag
SRC_DIR = SCRIPT_DIR.parent  # This is src
PROJECT_ROOT = SRC_DIR.parent  # This is project root
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# Import configurations
try:
    from config import RAG_MODE_CONFIG, GRAPHDB_CONFIG
except ImportError:
    RAG_MODE_CONFIG = {'default_mode': 'vector', 'available_modes': ['vector', 'graph', 'hybrid']}
    GRAPHDB_CONFIG = {'neo4j_password': None}

from rag.rag_system import SiddhaRAG


def display_mode_menu():
    """Display interactive mode selection menu."""
    print("\n" + "="*60)
    print("🩺 Siddha RAG System - Mode Selection")
    print("="*60)
    print("\nAvailable modes:\n")
    print("  [1] 📊 VECTOR MODE")
    print("      Fast semantic search using ChromaDB")
    print("      ✅ No additional setup required")
    print()
    print("  [2] 🔗 GRAPH MODE") 
    print("      Relationship-based retrieval using Neo4j")
    print("      ⚠️ Requires Neo4j database running")
    print()
    print("  [3] 🔀 HYBRID MODE (Recommended)")
    print("      Combines vector + graph for best results")
    print("      ⚠️ Requires Neo4j database running")
    print()


def get_mode_from_user() -> str:
    """Get mode selection from user input."""
    display_mode_menu()
    
    while True:
        choice = input("Select mode [1/2/3] or 'q' to quit: ").strip().lower()
        
        if choice == 'q':
            print("\n👋 Goodbye!")
            sys.exit(0)
        elif choice == '1':
            return 'vector'
        elif choice == '2':
            return 'graph'
        elif choice == '3':
            return 'hybrid'
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")


def get_neo4j_password() -> str:
    """Get Neo4j password from environment or user input."""
    # Try environment variable first
    password = os.getenv('NEO4J_PASSWORD')
    if password:
        return password
    
    # Try config
    if GRAPHDB_CONFIG.get('neo4j_password'):
        return GRAPHDB_CONFIG['neo4j_password']
    
    # Ask user
    print("\n⚠️ Neo4j password required for graph/hybrid mode.")
    password = input("Enter Neo4j password (or set NEO4J_PASSWORD env var): ").strip()
    
    if not password:
        print("❌ Password required for graph mode.")
        sys.exit(1)
    
    return password


def run_vector_mode():
    """Run RAG in vector-only mode."""
    print("\n" + "="*60)
    print("📊 Starting VECTOR MODE")
    print("="*60)
    print("Using: ChromaDB for semantic search")
    print()
    
    rag = SiddhaRAG(use_graph=False)
    rag.interactive_mode()


def run_graph_mode(password: str):
    """Run RAG in graph-only mode."""
    print("\n" + "="*60)
    print("🔗 Starting GRAPH MODE")
    print("="*60)
    print("Using: Neo4j knowledge graph for relationship queries")
    print()
    
    rag = SiddhaRAG(
        use_graph=True,
        neo4j_password=password
    )
    
    # Custom graph-only query loop
    print("\n" + "="*60)
    print("🔗 Graph Mode - Interactive Q&A")
    print("="*60)
    print("\nType your questions. Commands: 'quit' to exit\n")
    
    while True:
        try:
            question = input("❓ Your question: ").strip()
            
            if not question:
                continue
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Use graph strategy
            response = rag.query(question, strategy='graph')
            
            print("\n" + "-"*40)
            print("📝 Answer:")
            print(response['answer'])
            print("-"*40 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break


def run_hybrid_mode(password: str):
    """Run RAG in hybrid mode."""
    print("\n" + "="*60)
    print("🔀 Starting HYBRID MODE")
    print("="*60)
    print("Using: ChromaDB + Neo4j for best results")
    print()
    
    rag = SiddhaRAG(
        use_graph=True,
        neo4j_password=password
    )
    
    # Custom hybrid query loop
    print("\n" + "="*60)
    print("🔀 Hybrid Mode - Interactive Q&A")
    print("="*60)
    print("\nType your questions. Commands: 'quit' to exit\n")
    
    while True:
        try:
            question = input("❓ Your question: ").strip()
            
            if not question:
                continue
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Use hybrid strategy
            response = rag.query(question, strategy='hybrid')
            
            print("\n" + "-"*40)
            print("📝 Answer:")
            print(response['answer'])
            
            # Show additional info
            if response.get('entities_found'):
                print(f"\n🔗 Entities: {', '.join(response['entities_found'][:5])}")
            if response.get('graph_facts'):
                print(f"📊 Graph facts used: {len(response['graph_facts'])}")
            
            print("-"*40 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break


def main():
    """Main launcher with mode selection."""
    parser = argparse.ArgumentParser(
        description="Siddha RAG System - Unified Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  vector   Fast semantic search (ChromaDB only)
  graph    Relationship-based search (Neo4j only)
  hybrid   Combined approach (recommended)

Examples:
  python start_rag.py                   # Interactive selection
  python start_rag.py --mode vector     # Vector mode
  python start_rag.py --mode hybrid     # Hybrid mode
        """
    )
    parser.add_argument(
        '--mode', '-m',
        choices=['vector', 'graph', 'hybrid'],
        help='RAG mode to use'
    )
    parser.add_argument(
        '--password', '-p',
        help='Neo4j password (or set NEO4J_PASSWORD env var)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("🩺 Siddha Medicine RAG System")
    print("="*60)
    
    # Determine mode
    if args.mode:
        mode = args.mode
        print(f"\n📌 Mode: {mode.upper()}")
    else:
        mode = get_mode_from_user()
    
    # Run appropriate mode
    try:
        if mode == 'vector':
            run_vector_mode()
            
        elif mode == 'graph':
            password = args.password or get_neo4j_password()
            run_graph_mode(password)
            
        elif mode == 'hybrid':
            password = args.password or get_neo4j_password()
            run_hybrid_mode(password)
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if mode in ['graph', 'hybrid']:
            print("\n💡 Troubleshooting:")
            print("  1. Is Neo4j running? (Check Docker or Neo4j Desktop)")
            print("  2. Is the password correct?")
            print("  3. Try: python scripts/setup_neo4j.py --password <password>")
        else:
            print("\n💡 Make sure you've run the pipeline first:")
            print("  python src/pipeline/main_pipeline.py")


if __name__ == "__main__":
    main()
