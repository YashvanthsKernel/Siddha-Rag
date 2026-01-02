"""
Import Graph Data to Local Neo4j
Imports the JSON files generated from Colab into your local Neo4j instance.

Usage:
    python scripts/import_graph_data.py --nodes nodes_XXXXX.json --triples triples_XXXXX.json
"""

import json
import argparse
from pathlib import Path
from neo4j import GraphDatabase
from tqdm import tqdm
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()


def import_to_neo4j(
    nodes_file: str,
    triples_file: str,
    neo4j_uri: str = None,
    neo4j_user: str = None,
    neo4j_password: str = None
):
    """
    Import nodes and triples from JSON files to Neo4j.
    """
    # Use environment variables or defaults
    uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
    password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password")
    
    print("=" * 60)
    print("📥 Import Graph Data to Neo4j")
    print("=" * 60)
    
    # Load JSON files
    print(f"\n📂 Loading data files...")
    
    with open(nodes_file, 'r', encoding='utf-8') as f:
        nodes = json.load(f)
    print(f"   ✓ Loaded {len(nodes)} nodes from {Path(nodes_file).name}")
    
    with open(triples_file, 'r', encoding='utf-8') as f:
        triples = json.load(f)
    print(f"   ✓ Loaded {len(triples)} triples from {Path(triples_file).name}")
    
    # Connect to Neo4j
    print(f"\n🔌 Connecting to Neo4j at {uri}...")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        print("   ✓ Connected successfully!")
    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
        print("\n💡 Make sure:")
        print("   1. Neo4j is running")
        print("   2. Check your .env file for correct credentials")
        return
    
    # Import nodes
    print(f"\n📊 Importing {len(nodes)} nodes...")
    
    with driver.session() as session:
        for node in tqdm(nodes, desc="Nodes"):
            try:
                # Create node with properties
                node_type = node.get('type', 'Entity')
                node_name = node.get('name', '')
                
                if not node_name:
                    continue
                
                query = f"""
                    MERGE (n:{node_type} {{name: $name}})
                    SET n.sources = $sources
                    SET n.updated_at = datetime()
                """
                
                session.run(
                    query,
                    name=node_name,
                    sources=node.get('sources', [])
                )
                
                # Add embedding if present
                if 'embedding' in node:
                    session.run(f"""
                        MATCH (n:{node_type} {{name: $name}})
                        SET n.embedding = $embedding
                    """, name=node_name, embedding=node['embedding'])
                    
            except Exception as e:
                print(f"\n   ⚠️ Error with node '{node.get('name', 'unknown')}': {e}")
    
    # Import relationships
    print(f"\n🔗 Importing {len(triples)} relationships...")
    
    with driver.session() as session:
        for triple in tqdm(triples, desc="Relationships"):
            try:
                subject = triple.get('subject', '')
                predicate = triple.get('predicate', 'RELATED_TO')
                obj = triple.get('object', '')
                subject_type = triple.get('subject_type', 'Entity')
                object_type = triple.get('object_type', 'Entity')
                
                if not subject or not obj:
                    continue
                
                # Ensure predicate is valid Cypher identifier
                predicate = predicate.upper().replace(" ", "_").replace("-", "_")
                
                query = f"""
                    MERGE (s:{subject_type} {{name: $subject}})
                    MERGE (o:{object_type} {{name: $object}})
                    MERGE (s)-[r:{predicate}]->(o)
                    SET r.source = $source
                    SET r.confidence = $confidence
                    SET r.updated_at = datetime()
                """
                
                session.run(
                    query,
                    subject=subject,
                    object=obj,
                    source=triple.get('source', ''),
                    confidence=triple.get('confidence', 0.8)
                )
                
            except Exception as e:
                print(f"\n   ⚠️ Error with triple: {e}")
    
    # Get final stats
    print("\n📊 Verifying import...")
    
    with driver.session() as session:
        node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
    
    driver.close()
    
    print("\n" + "=" * 60)
    print("✅ IMPORT COMPLETE!")
    print("=" * 60)
    print(f"\n   Total Nodes: {node_count}")
    print(f"   Total Relationships: {rel_count}")
    print(f"\n💡 Open Neo4j Browser: http://localhost:7474")
    print("   Run: MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import graph data to Neo4j")
    parser.add_argument("--nodes", required=True, help="Path to nodes JSON file")
    parser.add_argument("--triples", required=True, help="Path to triples JSON file")
    parser.add_argument("--uri", default=None, help="Neo4j URI (default: from .env)")
    parser.add_argument("--user", default=None, help="Neo4j user (default: from .env)")
    parser.add_argument("--password", default=None, help="Neo4j password (default: from .env)")
    
    args = parser.parse_args()
    
    import_to_neo4j(
        nodes_file=args.nodes,
        triples_file=args.triples,
        neo4j_uri=args.uri,
        neo4j_user=args.user,
        neo4j_password=args.password
    )
