"""
Graph Data Export Utility
Exports Neo4j graph data to JSON files for backup and portability.
Saves to data/graphdb directory.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from graph.config import GRAPH_DATA_PATH, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from neo4j import GraphDatabase


def export_graph_to_json(
    output_dir: str = None,
    neo4j_password: str = None
) -> Dict:
    """
    Export all nodes and relationships from Neo4j to JSON files.
    
    Args:
        output_dir: Directory to save exports (default: data/graphdb)
        neo4j_password: Neo4j password
        
    Returns:
        Export statistics
    """
    output_path = Path(output_dir or GRAPH_DATA_PATH)
    output_path.mkdir(parents=True, exist_ok=True)
    
    password = neo4j_password or NEO4J_PASSWORD
    
    print("="*60)
    print("📤 Exporting Neo4j Graph to JSON")
    print("="*60)
    
    try:
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, password)
        )
        driver.verify_connectivity()
        print(f"✅ Connected to Neo4j")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return {'error': str(e)}
    
    stats = {
        'nodes': 0,
        'relationships': 0,
        'timestamp': datetime.now().isoformat()
    }
    
    with driver.session() as session:
        # Export all nodes
        print("\n📊 Exporting nodes...")
        nodes_query = """
        MATCH (n)
        RETURN 
            id(n) as id,
            labels(n) as labels,
            properties(n) as properties
        """
        result = session.run(nodes_query)
        nodes = []
        for record in result:
            node = {
                'id': record['id'],
                'labels': record['labels'],
                'properties': dict(record['properties'])
            }
            # Remove embeddings (too large for JSON)
            if 'embedding' in node['properties']:
                del node['properties']['embedding']
            nodes.append(node)
        
        stats['nodes'] = len(nodes)
        print(f"   Found {len(nodes)} nodes")
        
        # Export all relationships
        print("🔗 Exporting relationships...")
        rels_query = """
        MATCH (a)-[r]->(b)
        RETURN 
            id(r) as id,
            type(r) as type,
            id(a) as source,
            id(b) as target,
            properties(r) as properties
        """
        result = session.run(rels_query)
        relationships = []
        for record in result:
            rel = {
                'id': record['id'],
                'type': record['type'],
                'source': record['source'],
                'target': record['target'],
                'properties': dict(record['properties'])
            }
            relationships.append(rel)
        
        stats['relationships'] = len(relationships)
        print(f"   Found {len(relationships)} relationships")
    
    driver.close()
    
    # Save to files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    nodes_file = output_path / f"nodes_{timestamp}.json"
    with open(nodes_file, 'w', encoding='utf-8') as f:
        json.dump(nodes, f, indent=2, default=str)
    print(f"\n💾 Saved nodes to: {nodes_file}")
    
    rels_file = output_path / f"relationships_{timestamp}.json"
    with open(rels_file, 'w', encoding='utf-8') as f:
        json.dump(relationships, f, indent=2, default=str)
    print(f"💾 Saved relationships to: {rels_file}")
    
    # Save combined export
    combined = {
        'metadata': stats,
        'nodes': nodes,
        'relationships': relationships
    }
    combined_file = output_path / f"graph_export_{timestamp}.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"💾 Saved combined export to: {combined_file}")
    
    # Save latest reference
    latest_file = output_path / "latest_export.json"
    with open(latest_file, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("✅ Export Complete!")
    print("="*60)
    print(f"Nodes: {stats['nodes']}")
    print(f"Relationships: {stats['relationships']}")
    print(f"Location: {output_path}")
    
    return stats


def import_graph_from_json(
    input_file: str,
    neo4j_password: str = None,
    clear_existing: bool = False
) -> Dict:
    """
    Import graph data from JSON file to Neo4j.
    
    Args:
        input_file: Path to JSON file
        neo4j_password: Neo4j password
        clear_existing: Clear existing data before import
        
    Returns:
        Import statistics
    """
    print("="*60)
    print("📥 Importing JSON to Neo4j Graph")
    print("="*60)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    password = neo4j_password or NEO4J_PASSWORD
    
    try:
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, password)
        )
        driver.verify_connectivity()
        print(f"✅ Connected to Neo4j")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return {'error': str(e)}
    
    stats = {'nodes_created': 0, 'relationships_created': 0}
    
    with driver.session() as session:
        if clear_existing:
            print("⚠️ Clearing existing data...")
            session.run("MATCH (n) DETACH DELETE n")
        
        # Import nodes
        nodes = data.get('nodes', [])
        print(f"\n📊 Importing {len(nodes)} nodes...")
        
        for node in nodes:
            labels = ':'.join(node['labels'])
            props = node['properties']
            
            query = f"CREATE (n:{labels}) SET n = $props"
            session.run(query, props=props)
            stats['nodes_created'] += 1
        
        print(f"   Created {stats['nodes_created']} nodes")
        
        # Note: Relationships require node matching which is complex
        # This is a simplified version
        print("⚠️ Relationships import requires node ID matching (not implemented)")
    
    driver.close()
    
    print("\n" + "="*60)
    print("✅ Import Complete!")
    print("="*60)
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export/Import Neo4j graph data")
    parser.add_argument('action', choices=['export', 'import'])
    parser.add_argument('--password', required=True, help='Neo4j password')
    parser.add_argument('--file', help='Input file for import')
    parser.add_argument('--output', default='data/graphdb', help='Output directory for export')
    
    args = parser.parse_args()
    
    if args.action == 'export':
        export_graph_to_json(args.output, args.password)
    elif args.action == 'import':
        if not args.file:
            print("❌ --file required for import")
        else:
            import_graph_from_json(args.file, args.password)
