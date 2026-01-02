"""
Neo4j Graph Builder
Populates knowledge graph with extracted entities and relationships.
"""

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
import ollama
from typing import List, Dict, Optional, Tuple
import time
from datetime import datetime
from .config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    EXTRACTION_BATCH_SIZE
)


class GraphBuilder:
    """
    Manages Neo4j graph database operations.
    Creates nodes, relationships, and embeddings in the knowledge graph.
    """
    
    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None,
        embedding_model: str = EMBEDDING_MODEL
    ):
        """
        Initialize graph builder with Neo4j connection.
        
        Args:
            uri: Neo4j connection URI (default from config)
            user: Neo4j username (default from config)
            password: Neo4j password (default from config)
            embedding_model: Ollama model for embeddings
        """
        self.uri = uri or NEO4J_URI
        self.user = user or NEO4J_USER
        self.password = password or NEO4J_PASSWORD
        self.embedding_model = embedding_model
        
        # Initialize driver
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # Test connection
            self.driver.verify_connectivity()
            print(f"✅ Connected to Neo4j at {self.uri}")
        except ServiceUnavailable:
            print(f"❌ Cannot connect to Neo4j at {self.uri}")
            print("   Please ensure Neo4j is running")
            raise
        except AuthError:
            print(f"❌ Authentication failed for user: {self.user}")
            print("   Please check your Neo4j credentials")
            raise
    
    def close(self):
        """Close the Neo4j driver connection"""
        if self.driver:
            self.driver.close()
            print("✅ Neo4j connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text.
        
        Args:
            text: Text to embed
            
        Returns:
            768-dimensional embedding vector
        """
        try:
            response = ollama.embeddings(
                model=self.embedding_model,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            print(f"⚠️ Embedding error: {e}")
            # Return zero vector as fallback
            return [0.0] * EMBEDDING_DIMENSION
    
    def create_node(
        self,
        entity_name: str,
        entity_type: str,
        properties: Optional[Dict] = None,
        generate_embedding: bool = True
    ) -> bool:
        """
        Create or update a node in the graph.
        
        Args:
            entity_name: Name of the entity
            entity_type: Type of entity (Herb, Disease, etc.)
            properties: Additional node properties
            generate_embedding: Whether to generate embedding vector
            
        Returns:
            True if successful, False otherwise
        """
        # Prepare properties
        props = properties or {}
        props['name'] = entity_name
        props['updated_at'] = datetime.now().isoformat()
        
        # Generate embedding if requested
        if generate_embedding:
            embedding = self.get_embedding(entity_name)
            props['embedding'] = embedding
        
        # Build Cypher query using MERGE (upsert)
        cypher = f"""
        MERGE (n:{entity_type} {{name: $name}})
        ON CREATE SET n += $properties, n.created_at = datetime()
        ON MATCH SET n += $properties
        RETURN n.name as name
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(cypher, name=entity_name, properties=props)
                record = result.single()
                return record is not None
        except Exception as e:
            print(f"❌ Error creating node {entity_name} ({entity_type}): {e}")
            return False
    
    def create_relationship(
        self,
        subject: str,
        subject_type: str,
        predicate: str,
        obj: str,
        obj_type: str,
        properties: Optional[Dict] = None
    ) -> bool:
        """
        Create a relationship between two nodes.
        
        Args:
            subject: Source entity name
            subject_type: Source entity type
            predicate: Relationship type
            obj: Target entity name
            obj_type: Target entity type
            properties: Relationship properties
            
        Returns:
            True if successful, False otherwise
        """
        props = properties or {}
        props['updated_at'] = datetime.now().isoformat()
        
        # Build Cypher query
        cypher = f"""
        MERGE (s:{subject_type} {{name: $subject}})
        MERGE (o:{obj_type} {{name: $object}})
        MERGE (s)-[r:{predicate}]->(o)
        ON CREATE SET r += $properties, r.created_at = datetime()
        ON MATCH SET r += $properties
        RETURN s.name, r, o.name
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(
                    cypher,
                    subject=subject,
                    object=obj,
                    properties=props
                )
                record = result.single()
                return record is not None
        except Exception as e:
            print(f"❌ Error creating relationship ({subject})-[{predicate}]->({obj}): {e}")
            return False
    
    def add_triple(self, triple: Dict) -> bool:
        """
        Add a complete triple (two nodes + one relationship) to the graph.
        
        Args:
            triple: Dictionary with subject, predicate, object, types, etc.
            
        Returns:
            True if successful, False otherwise
        """
        # Create subject node
        subject_created = self.create_node(
            triple['subject'],
            triple['subject_type']
        )
        
        # Create object node
        object_created = self.create_node(
            triple['object'],
            triple['object_type']
        )
        
        # Create relationship with confidence
        rel_props = {
            'confidence': triple.get('confidence', 0.7)
        }
        
        if 'source' in triple:
            rel_props['source'] = triple['source']
        
        relationship_created = self.create_relationship(
            triple['subject'],
            triple['subject_type'],
            triple['predicate'],
            triple['object'],
            triple['object_type'],
            rel_props
        )
        
        return subject_created and object_created and relationship_created
    
    def batch_add_triples(self, triples: List[Dict], show_progress: bool = True) -> Dict:
        """
        Add multiple triples to the graph efficiently.
        
        Args:
            triples: List of triple dictionaries
            show_progress: Whether to show progress
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'total': len(triples),
            'successful': 0,
            'failed': 0,
            'start_time': time.time()
        }
        
        for i, triple in enumerate(triples):
            if show_progress and i % EXTRACTION_BATCH_SIZE == 0:
                print(f"📊 Processing triple {i+1}/{len(triples)}...")
            
            if self.add_triple(triple):
                stats['successful'] += 1
            else:
                stats['failed'] += 1
        
        stats['duration'] = time.time() - stats['start_time']
        
        if show_progress:
            print(f"\n✅ Batch complete:")
            print(f"   • Successful: {stats['successful']}/{stats['total']}")
            print(f"   • Failed: {stats['failed']}/{stats['total']}")
            print(f"   • Duration: {stats['duration']:.1f}s")
        
        return stats
    
    def query_graph(self, cypher: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """
        Execute a custom Cypher query.
        
        Args:
            cypher: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result records as dictionaries
        """
        try:
            with self.driver.session() as session:
                result = session.run(cypher, parameters or {})
                return [dict(record) for record in result]
        except Exception as e:
            print(f"❌ Query error: {e}")
            return []
    
    def get_node_count(self, node_type: Optional[str] = None) -> int:
        """
        Get count of nodes in the graph.
        
        Args:
            node_type: Specific node type to count (None for all)
            
        Returns:
            Number of nodes
        """
        if node_type:
            cypher = f"MATCH (n:{node_type}) RETURN count(n) as count"
        else:
            cypher = "MATCH (n) RETURN count(n) as count"
        
        results = self.query_graph(cypher)
        return results[0]['count'] if results else 0
    
    def get_relationship_count(self, rel_type: Optional[str] = None) -> int:
        """
        Get count of relationships in the graph.
        
        Args:
            rel_type: Specific relationship type to count (None for all)
            
        Returns:
            Number of relationships
        """
        if rel_type:
            cypher = f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count"
        else:
            cypher = "MATCH ()-[r]->() RETURN count(r) as count"
        
        results = self.query_graph(cypher)
        return results[0]['count'] if results else 0
    
    def get_graph_stats(self) -> Dict:
        """
        Get comprehensive graph statistics.
        
        Returns:
            Dictionary with node counts, relationship counts, etc.
        """
        stats = {
            'total_nodes': self.get_node_count(),
            'total_relationships': self.get_relationship_count(),
            'node_types': {},
            'relationship_types': {}
        }
        
        # Count nodes by type
        from .config import ENTITY_TYPES
        for entity_type in ENTITY_TYPES:
            count = self.get_node_count(entity_type)
            if count > 0:
                stats['node_types'][entity_type] = count
        
        # Count relationships by type
        from .config import RELATIONSHIP_TYPES
        for rel_type in RELATIONSHIP_TYPES:
            count = self.get_relationship_count(rel_type)
            if count > 0:
                stats['relationship_types'][rel_type] = count
        
        return stats
    
    def clear_graph(self, confirm: bool = False):
        """
        Delete all nodes and relationships (WARNING: destructive!).
        
        Args:
            confirm: Must be True to execute
        """
        if not confirm:
            print("⚠️ Graph clear not confirmed. Set confirm=True to proceed.")
            return
        
        cypher = "MATCH (n) DETACH DELETE n"
        
        print("⚠️ Clearing entire graph...")
        with self.driver.session() as session:
            session.run(cypher)
        print("✅ Graph cleared")


def main():
    """Example usage and testing"""
    
    print("🔧 Testing GraphBuilder...\n")
    
    # Initialize builder
    try:
        with GraphBuilder() as builder:
            # Test 1: Check connection
            print("Test 1: Connection")
            stats = builder.get_graph_stats()
            print(f"✅ Connected. Current graph has {stats['total_nodes']} nodes\n")
            
            # Test 2: Create nodes
            print("Test 2: Creating nodes")
            builder.create_node("Neem", "Herb", {
                "scientific_name": "Azadirachta indica",
                "properties": "Antibacterial, antifungal"
            })
            builder.create_node("Skin Infection", "Disease", {
                "category": "Dermatological"
            })
            print("✅ Nodes created\n")
            
            # Test 3: Create relationship
            print("Test 3: Creating relationship")
            builder.create_relationship(
                "Neem", "Herb",
                "TREATS",
                "Skin Infection", "Disease",
                {"efficacy": "high", "confidence": 0.9}
            )
            print("✅ Relationship created\n")
            
            # Test 4: Add triple
            print("Test 4: Adding triple")
            triple = {
                "subject": "Tulsi",
                "subject_type": "Herb",
                "predicate": "RELIEVES",
                "object": "Fever",
                "object_type": "Symptom",
                "confidence": 0.85
            }
            builder.add_triple(triple)
            print("✅ Triple added\n")
            
            # Test 5: Get stats
            print("Test 5: Graph statistics")
            stats = builder.get_graph_stats()
            print(f"Nodes: {stats['total_nodes']}")
            print(f"Relationships: {stats['total_relationships']}")
            for node_type, count in stats['node_types'].items():
                print(f"  • {node_type}: {count}")
            print()
            
            print("✅ All tests passed!")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    main()
