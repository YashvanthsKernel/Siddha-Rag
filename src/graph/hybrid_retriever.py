"""
Hybrid Retriever: Combines Vector DB and Graph DB
Provides flexible retrieval strategies for RAG system.
"""

from neo4j import GraphDatabase
import chromadb
import ollama
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Any, Tuple
import json
from .config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    DEFAULT_TOP_K,
    GRAPH_TRAVERSAL_DEPTH,
    HYBRID_WEIGHT_VECTOR,
    HYBRID_WEIGHT_GRAPH,
    EMBEDDING_MODEL
)
from .siddha_entities import extract_siddha_entities, ALL_SIDDHA_ENTITIES, get_entity_type

# Similarity threshold - filter out low quality matches
MIN_SIMILARITY_THRESHOLD = 0.5  # ChromaDB uses L2 distance, lower is better


class HybridRetriever:
    """
    Unified retrieval engine combining vector and graph databases.
    Supports three modes: vector-only, graph-only, and hybrid.
    """
    
    def __init__(
        self,
        chroma_path: str = "data/vectordb",
        chroma_collection: str = "siddha_knowledge",
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        embedding_model: str = "all-MiniLM-L6-v2"  # Use SentenceTransformers model (384d)
    ):
        """
        Initialize hybrid retriever with both databases.
        
        Args:
            chroma_path: Path to ChromaDB storage
            chroma_collection: ChromaDB collection name
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            embedding_model: SentenceTransformers model for embeddings
        """
        # Initialize SentenceTransformer for embeddings (384d - matches ChromaDB)
        print(f"🔄 Loading embedding model: {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"✅ Model loaded (dim={self.embedding_model.get_sentence_embedding_dimension()})")
        
        # Initialize ChromaDB (vector database)
        try:
            self.chroma_client = chromadb.PersistentClient(path=chroma_path)
            self.collection = self.chroma_client.get_collection(chroma_collection)
            print(f"✅ Connected to ChromaDB: {chroma_collection}")
        except Exception as e:
            print(f"⚠️ ChromaDB connection failed: {e}")
            self.chroma_client = None
            self.collection = None
        
        # Initialize Neo4j (graph database)
        try:
            self.neo4j_driver = GraphDatabase.driver(
                neo4j_uri or NEO4J_URI,
                auth=(neo4j_user or NEO4J_USER, neo4j_password or NEO4J_PASSWORD)
            )
            self.neo4j_driver.verify_connectivity()
            print(f"✅ Connected to Neo4j at {neo4j_uri or NEO4J_URI}")
        except Exception as e:
            print(f"⚠️ Neo4j connection failed: {e}")
            self.neo4j_driver = None
    
    def close(self):
        """Close database connections"""
        if self.neo4j_driver:
            self.neo4j_driver.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def _generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a query string using SentenceTransformers"""
        try:
            embedding = self.embedding_model.encode(query, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            print(f"❌ Embedding generation error: {e}")
            return []
    
    def _extract_entities_from_query(self, query: str) -> List[str]:
        """
        Extract entity names from user query.
        Uses Siddha dictionary first, then LLM, then keyword fallback.
        
        Args:
            query: User question
            
        Returns:
            List of entity names
        """
        entities_found = []
        
        # Step 1: Use Siddha-specific dictionary (fast and reliable)
        siddha_entities = extract_siddha_entities(query)
        if siddha_entities:
            entities_found.extend(siddha_entities)
            print(f"   📚 Siddha dictionary matched: {siddha_entities}")
        
        # Step 2: Keyword extraction for terms not in dictionary
        stopwords = {'what', 'which', 'where', 'when', 'does', 'have', 'this', 'that', 'with', 
                     'from', 'about', 'help', 'treat', 'used', 'medicine', 'medicene', 'medical',
                     'siddha', 'ayurveda', 'traditional', 'indian', 'system', 'tell', 'know',
                     'good', 'best', 'cure', 'treatment', 'remedy', 'natural'}
        
        import re
        words = re.findall(r'\b[a-zA-Z]{4,}\b', query.lower())
        keywords = [w for w in words if w not in stopwords and w not in entities_found]
        entities_found.extend(keywords)
        
        # Step 3: Try LLM for complex queries (if few entities found)
        if len(entities_found) < 2:
            try:
                prompt = f"""Extract key medical entities from this text. Return ONLY a JSON array.
Text: {query}
Examples: "fever treatment" → ["fever"] | "neem for skin" → ["neem", "skin"]"""
                
                response = ollama.generate(
                    model="llama3.2:3b",
                    prompt=prompt,
                    format="json",
                    options={"temperature": 0.1, "num_predict": 50}
                )
                
                llm_entities = json.loads(response['response'])
                if isinstance(llm_entities, list):
                    for e in llm_entities:
                        if e and str(e).lower() not in [x.lower() for x in entities_found]:
                            entities_found.append(str(e).lower())
            except:
                pass  # Silently fail, we have dictionary results
        
        # Return unique entities, at least one
        result = list(set(entities_found))
        return result if result else [query.split()[-1]]
    
    def retrieve_vector(self, query: str, top_k: int = DEFAULT_TOP_K) -> Dict:
        """
        Vector-only retrieval using ChromaDB.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            Dictionary with documents, metadata, distances
        """
        if not self.collection:
            return {'method': 'vector', 'documents': [], 'metadatas': [], 'distances': []}
        
        print(f"🔍 Vector search: '{query}'")
        
        # Generate query embedding
        query_embedding = self._generate_query_embedding(query)
        if not query_embedding:
            return {'method': 'vector', 'documents': [], 'metadatas': [], 'distances': []}
        
        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Filter by similarity threshold
        docs = results['documents'][0] if results['documents'] else []
        metas = results['metadatas'][0] if results['metadatas'] else []
        dists = results['distances'][0] if results['distances'] else []
        
        # Keep only results below threshold (lower distance = more similar)
        filtered_docs, filtered_metas, filtered_dists = [], [], []
        for doc, meta, dist in zip(docs, metas, dists):
            if dist <= MIN_SIMILARITY_THRESHOLD:
                filtered_docs.append(doc)
                filtered_metas.append(meta)
                filtered_dists.append(dist)
        
        # If threshold filters too much, keep top 3
        if len(filtered_docs) == 0 and len(docs) > 0:
            filtered_docs = docs[:3]
            filtered_metas = metas[:3]
            filtered_dists = dists[:3]
        
        formatted = {
            'method': 'vector',
            'query': query,
            'documents': filtered_docs,
            'metadatas': filtered_metas,
            'distances': filtered_dists
        }
        
        print(f"✅ Found {len(filtered_docs)} vector results (filtered from {len(docs)})")
        
        return formatted
    
    def retrieve_graph(self, query: str, top_k: int = DEFAULT_TOP_K) -> Dict:
        """
        Graph-only retrieval using Neo4j.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            Dictionary with graph facts and entities
        """
        if not self.neo4j_driver:
            return {'method': 'graph', 'entities': [], 'facts': []}
        
        print(f"🔍 Graph search: '{query}'")
        
        # Extract entities from query
        query_entities = self._extract_entities_from_query(query)
        
        if not query_entities:
            print("⚠️ No entities extracted from query")
            return {'method': 'graph', 'entities': [], 'facts': []}
        
        print(f"📝 Query entities: {query_entities}")
        
        # Fuzzy match entities in graph (case-insensitive, partial match)
        cypher_find_entities = """
        MATCH (entity)
        WHERE toLower(entity.name) CONTAINS toLower($search_term)
        RETURN DISTINCT entity.name as name, labels(entity)[0] as type
        LIMIT 5
        """
        
        found_entities = []
        with self.neo4j_driver.session() as session:
            for entity_name in query_entities:
                result = session.run(cypher_find_entities, search_term=entity_name)
                found_entities.extend([dict(record) for record in result])
        
        if not found_entities:
            print("⚠️ No matching entities found in graph")
            return {'method': 'graph', 'entities': [], 'facts': []}
        
        print(f"✅ Found {len(found_entities)} matching entities")
        
        # Get relationships for found entities (up to GRAPH_TRAVERSAL_DEPTH hops)
        entity_names = [e['name'] for e in found_entities]
        
        cypher_get_facts = f"""
        MATCH (entity)
        WHERE entity.name IN $entity_names
        
        // Get direct relationships
        OPTIONAL MATCH (entity)-[r1]-(related1)
        
        // Get 2-hop relationships
        OPTIONAL MATCH (entity)-[r1]-(related1)-[r2]-(related2)
        WHERE r1 <> r2
        
        RETURN 
            entity.name as entity,
            labels(entity)[0] as entity_type,
            collect(DISTINCT {{
                relationship: type(r1),
                related_entity: related1.name,
                related_type: labels(related1)[0]
            }})[0..{top_k}] as direct_facts,
            collect(DISTINCT {{
                hop1_rel: type(r1),
                hop1_entity: related1.name,
                hop2_rel: type(r2),
                hop2_entity: related2.name
            }})[0..{top_k // 2}] as multi_hop_facts
        """
        
        graph_facts = []
        with self.neo4j_driver.session() as session:
            result = session.run(cypher_get_facts, entity_names=entity_names)
            graph_facts = [dict(record) for record in result]
        
        formatted = {
            'method': 'graph',
            'query': query,
            'entities': found_entities,
            'facts': graph_facts,
            'query_entities': query_entities
        }
        
        total_facts = sum(
            len(f.get('direct_facts', [])) + len(f.get('multi_hop_facts', []))
            for f in graph_facts
        )
        print(f"✅ Found {total_facts} graph facts")
        
        return formatted
    
    def retrieve_hybrid(self, query: str, top_k: int = DEFAULT_TOP_K) -> Dict:
        """
        Hybrid retrieval: vector search + graph expansion.
        
        Strategy: Retrieve-then-traverse
        1. Vector search finds semantically relevant chunks
        2. Extract entities mentioned in those chunks
        3. Graph traversal expands relationships around entities
        4. Merge results
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            Dictionary with both vector and graph results
        """
        print(f"🔍 Hybrid search: '{query}'")
        
        # Step 1: Vector retrieval
        vector_results = self.retrieve_vector(query, top_k)
        
        # Step 2: Extract entities from BOTH query AND retrieved chunks
        entities_to_search = set()
        
        # Extract from query directly
        query_entities = self._extract_entities_from_query(query)
        entities_to_search.update(query_entities)
        
        # Extract from retrieved chunks
        for doc in vector_results['documents'][:3]:  # Top 3 chunks
            chunk_entities = self._extract_entities_from_query(doc[:300])
            entities_to_search.update(chunk_entities)
        
        # Step 3: Graph expansion with FUZZY MATCHING
        graph_results = {'entities': [], 'facts': []}
        if self.neo4j_driver and entities_to_search:
            print(f"📝 Searching graph for: {list(entities_to_search)[:5]}...")
            
            # Use fuzzy matching (CONTAINS) instead of exact match
            found_entities = []
            with self.neo4j_driver.session() as session:
                for entity_name in entities_to_search:
                    if len(entity_name) < 3:  # Skip very short terms
                        continue
                    try:
                        result = session.run("""
                            MATCH (entity)
                            WHERE toLower(entity.name) CONTAINS toLower($search_term)
                            RETURN DISTINCT entity.name as name, labels(entity)[0] as type
                            LIMIT 3
                        """, search_term=entity_name)
                        found_entities.extend([dict(r) for r in result])
                    except:
                        pass
                
                if found_entities:
                    print(f"   ✅ Found {len(found_entities)} matching nodes")
                    entity_names = list(set([e['name'] for e in found_entities]))
                    
                    # Get relationships for found entities
                    try:
                        result = session.run("""
                            MATCH (entity)
                            WHERE entity.name IN $entity_names
                            OPTIONAL MATCH (entity)-[r]-(related)
                            RETURN 
                                entity.name as entity,
                                labels(entity)[0] as entity_type,
                                collect(DISTINCT {
                                    relationship: type(r),
                                    related_entity: related.name,
                                    related_type: labels(related)[0]
                                })[0..10] as relationships
                        """, entity_names=entity_names)
                        graph_results['facts'] = [dict(record) for record in result]
                        graph_results['entities'] = found_entities
                    except Exception as e:
                        print(f"   ⚠️ Graph query error: {e}")
                else:
                    print(f"   ⚠️ No matching nodes found")
        
        # Step 4: Merge results
        formatted = {
            'method': 'hybrid',
            'query': query,
            'vector_results': vector_results,
            'graph_results': graph_results,
            'entities_found': list(entities_to_search),
            'total_sources': len(vector_results['documents']) + len(graph_results.get('facts', []))
        }
        
        print(f"✅ Hybrid: {len(vector_results['documents'])} chunks + {len(graph_results.get('facts', []))} graph facts")
        
        return formatted
    
    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        strategy: str = "hybrid"
    ) -> Dict:
        """
        Unified retrieval interface.
        
        Args:
            query: User question
            top_k: Number of results
            strategy: "vector", "graph", or "hybrid"
            
        Returns:
            Retrieved results dictionary
        """
        if strategy == "vector":
            return self.retrieve_vector(query, top_k)
        elif strategy == "graph":
            return self.retrieve_graph(query, top_k)
        elif strategy == "hybrid":
            return self.retrieve_hybrid(query, top_k)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'vector', 'graph', or 'hybrid'")
    
    def format_context_for_llm(self, results: Dict) -> str:
        """
        Convert retrieval results into formatted context string for LLM.
        
        Args:
            results: Results from retrieve()
            
        Returns:
           Formatted context string
        """
        method = results.get('method', 'unknown')
        context_parts = []
        
        if method == 'vector' or 'vector_results' in results:
            # Format vector results
            vector_data = results if method == 'vector' else results.get('vector_results', {})
            docs = vector_data.get('documents', [])
            metas = vector_data.get('metadatas', [])
            
            if docs:
                context_parts.append("## Retrieved Documents:")
                for i, (doc, meta) in enumerate(zip(docs, metas), 1):
                    source = meta.get('filename', 'unknown')
                    context_parts.append(f"\n### Source {i}: {source}")
                    context_parts.append(doc)
        
        if method == 'graph' or 'graph_results' in results:
            # Format graph results
            graph_data = results if method == 'graph' else results.get('graph_results', {})
            facts = graph_data.get('facts', [])
            
            if facts:
                context_parts.append("\n## Knowledge Graph Relationships:")
                for fact in facts:
                    entity = fact.get('entity', '')
                    relationships = fact.get('relationships', []) or fact.get('direct_facts', [])
                    
                    if relationships:
                        context_parts.append(f"\n### {entity}:")
                        for rel in relationships[:5]:  # Limit to top 5
                            rel_type = rel.get('relationship', 'RELATED_TO')
                            related = rel.get('related_entity', 'unknown')
                            context_parts.append(f"- {entity} {rel_type} {related}")
        
        return "\n".join(context_parts)


def main():
    """Example usage and testing"""
    
    print("🔧 Testing HybridRetriever...\n")
    
    try:
        with HybridRetriever() as retriever:
            test_query = "What herbs treat fever?"
            
            print(f"Query: {test_query}\n")
            print("=" * 60)
            
            # Test 1: Vector-only
            print("\n1️⃣ VECTOR-ONLY RETRIEVAL")
            print("=" * 60)
            vector_results = retriever.retrieve(test_query, top_k=3, strategy="vector")
            print(f"Found {len(vector_results.get('documents', []))} documents")
            
            # Test 2: Graph-only
            print("\n2️⃣ GRAPH-ONLY RETRIEVAL")
            print("=" * 60)
            graph_results = retriever.retrieve(test_query, top_k=5, strategy="graph")
            print(f"Found {len(graph_results.get('facts', []))} graph facts")
            
            # Test 3: Hybrid
            print("\n3️⃣ HYBRID RETRIEVAL")
            print("=" * 60)
            hybrid_results = retriever.retrieve(test_query, top_k=5, strategy="hybrid")
            print(f"Total sources: {hybrid_results.get('total_sources', 0)}")
            
            # Test 4: Format context
            print("\n4️⃣ FORMAT CONTEXT FOR LLM")
            print("=" * 60)
            context = retriever.format_context_for_llm(hybrid_results)
            print(context[:500] + "..." if len(context) > 500 else context)
            
            print("\n✅ All tests completed!")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
