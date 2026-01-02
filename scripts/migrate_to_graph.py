"""
Migration Script: Documents to Knowledge Graph
Reads documents and populates the Neo4j graph database.
"""

import os
import sys
import glob
from pathlib import Path
from typing import List, Dict
import time

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.graph.entity_extractor import KnowledgeExtractor
from src.graph.graph_builder import GraphBuilder
from src.graph.config import EXTRACTION_BATCH_SIZE


def load_documents(data_dir: str) -> List[Dict]:
    """
    Load documents from the data directory.
    Supports .txt and .md files.
    """
    documents = []
    path = Path(data_dir)
    
    if not path.exists():
        print(f"⚠️ Data directory not found: {data_dir}")
        return []
    
    # Find all text files
    files = list(path.glob("**/*.txt")) + list(path.glob("**/*.md"))
    
    print(f"📂 Found {len(files)} documents in {data_dir}")
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                if len(text.strip()) > 50:  # Skip empty/tiny files
                    documents.append({
                        'filename': file_path.name,
                        'path': str(file_path),
                        'text': text
                    })
        except Exception as e:
            print(f"⚠️ Error reading {file_path.name}: {e}")
            
    return documents


def migrate_documents(
    docs_dir: str = "data/processed",
    batch_size: int = 5,
    neo4j_password: str = None
):
    """
    Main migration function.
    """
    print("="*70)
    print("🚀 Siddha RAG - Knowledge Graph Migration")
    print("="*70)
    
    # 1. Load documents
    print(f"\nStep 1: Loading documents from {docs_dir}...")
    documents = load_documents(docs_dir)
    
    if not documents:
        print("❌ No documents found to migrate.")
        print("   Please ensure your documents are in the data directory.")
        return
    
    # 2. Initialize components
    print("\nStep 2: Initializing graph components...")
    
    try:
        extractor = KnowledgeExtractor(model="llama3.2:3b")
        builder = GraphBuilder(password=neo4j_password)
        
        # Check connection
        stats = builder.get_graph_stats()
        print(f"✅ Connected to Neo4j (Current nodes: {stats['total_nodes']})")
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return

    # 3. Process documents
    print(f"\nStep 3: Processing {len(documents)} documents...")
    print("   This may take a while (approx 1-2 mins per document)...")
    
    total_triples = 0
    start_time = time.time()
    
    for i, doc in enumerate(documents, 1):
        print(f"\n📄 Processing {i}/{len(documents)}: {doc['filename']}")
        
        # Chunk the text to avoid context limits and improve speed
        text = doc['text']
        chunk_size = 2000  # Characters per chunk
        chunks = [text[j:j+chunk_size] for j in range(0, len(text), chunk_size)]
        
        print(f"   • Split into {len(chunks)} chunks")
        
        doc_triples = []
        for j, chunk in enumerate(chunks, 1):
            # Extract triples from chunk
            print(f"     - Processing chunk {j}/{len(chunks)}...", end="\r")
            chunk_triples = extractor.extract_triples(chunk)
            doc_triples.extend(chunk_triples)
            
        print(f"     - Processing chunk {len(chunks)}/{len(chunks)}... Done")
        
        if not doc_triples:
            print("   ⚠️ No knowledge extracted")
            continue
            
        print(f"   • Found {len(doc_triples)} facts")
        
        # Add source metadata
        for triple in doc_triples:
            triple['source'] = doc['filename']
        
        # Add to graph
        print("   • Writing to Neo4j...")
        stats = builder.batch_add_triples(doc_triples, show_progress=False)
        
        total_triples += stats['successful']
        print(f"   ✅ Added {stats['successful']} facts")
        
    duration = time.time() - start_time
    
    print("\n" + "="*70)
    print("✅ Migration Complete!")
    print("="*70)
    print(f"Total Documents: {len(documents)}")
    print(f"Total Facts Added: {total_triples}")
    print(f"Total Time: {duration:.1f}s")
    print("\nNext steps:")
    print("1. Explore your graph in Neo4j Browser")
    print("2. Run the hybrid RAG demo: python examples/hybrid_rag_demo.py")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate documents to Neo4j")
    parser.add_argument("--docs", default="data/processed", help="Directory containing documents")
    parser.add_argument("--password", required=True, help="Neo4j password")
    
    args = parser.parse_args()
    
    migrate_documents(
        docs_dir=args.docs,
        neo4j_password=args.password
    )
