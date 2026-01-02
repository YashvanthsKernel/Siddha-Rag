"""
Document Retriever Module
Performs semantic search to retrieve relevant documents from vector database.
"""

import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional


class DocumentRetriever:
    """
    Retrieves relevant documents using vector similarity search.
    """
    
    def __init__(
        self,
        db_path: str = "data/vectordb",
        collection_name: str = "siddha_knowledge",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the document retriever.
        
        Args:
            db_path: Path to ChromaDB database
            collection_name: Name of the collection to query
            embedding_model: Sentence Transformer model for query embeddings
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Initialize Sentence Transformer model (same as embeddings.py)
        print(f"🔄 Loading embedding model: {embedding_model}...")
        self.model = SentenceTransformer(embedding_model)
        print(f"✅ Model loaded (dim={self.model.get_sentence_embedding_dimension()})")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"✅ Connected to collection: {collection_name}")
            print(f"   Total documents: {self.collection.count()}")
        except Exception as e:
            print(f"❌ Error: Collection '{collection_name}' not found!")
            print(f"   Make sure you've run the embeddings pipeline first.")
            raise e
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a query string.
        
        Args:
            query: Search query
            
        Returns:
            Embedding vector
        """
        try:
            embedding = self.model.encode(query, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            print(f"❌ Error generating query embedding: {e}")
            raise
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Retrieve most relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            filter_metadata: Optional metadata filter (e.g., {'source': 'specific_file.pdf'})
            
        Returns:
            Dictionary with documents, metadatas, and distances
        """
        print(f"\n🔍 Searching for: '{query}'")
        print(f"   Retrieving top {top_k} results...")
        
        # Generate query embedding
        query_embedding = self.generate_query_embedding(query)
        
        # Prepare query parameters
        query_params = {
            'query_embeddings': [query_embedding],
            'n_results': top_k
        }
        
        # Add metadata filter if provided
        if filter_metadata:
            query_params['where'] = filter_metadata
        
        # Search in ChromaDB
        results = self.collection.query(**query_params)
        
        # Format results
        formatted_results = {
            'query': query,
            'documents': results['documents'][0] if results['documents'] else [],
            'metadatas': results['metadatas'][0] if results['metadatas'] else [],
            'distances': results['distances'][0] if results['distances'] else [],
            'ids': results['ids'][0] if results['ids'] else []
        }
        
        # Display results
        print(f"\n✅ Found {len(formatted_results['documents'])} results")
        for i, (doc, metadata, distance) in enumerate(zip(
            formatted_results['documents'],
            formatted_results['metadatas'],
            formatted_results['distances']
        )):
            print(f"\n   Result {i+1} (distance: {distance:.4f}):")
            print(f"   Source: {metadata.get('filename', 'unknown')}")
            print(f"   Preview: {doc[:150]}...")
        
        return formatted_results
    
    def retrieve_by_source(self, query: str, source_filename: str, top_k: int = 3) -> Dict:
        """
        Retrieve documents from a specific source file.
        
        Args:
            query: Search query
            source_filename: Filename to filter by
            top_k: Number of results to return
            
        Returns:
            Retrieval results
        """
        return self.retrieve(
            query=query,
            top_k=top_k,
            filter_metadata={'filename': source_filename}
        )
    
    def get_context_string(self, results: Dict, include_sources: bool = True) -> str:
        """
        Format retrieval results into a context string for LLM.
        
        Args:
            results: Results from retrieve() method
            include_sources: Whether to include source citations
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, (doc, metadata) in enumerate(zip(
            results['documents'],
            results['metadatas']
        )):
            if include_sources:
                source = metadata.get('filename', 'unknown')
                context_parts.append(f"[Source: {source}]\n{doc}")
            else:
                context_parts.append(doc)
        
        return "\n\n---\n\n".join(context_parts)
    
    def similarity_threshold_filter(
        self, 
        results: Dict, 
        threshold: float = 0.5
    ) -> Dict:
        """
        Filter results by similarity threshold.
        Lower distance = more similar.
        
        Args:
            results: Results from retrieve()
            threshold: Maximum distance threshold (lower is more similar)
            
        Returns:
            Filtered results
        """
        filtered = {
            'query': results['query'],
            'documents': [],
            'metadatas': [],
            'distances': [],
            'ids': []
        }
        
        for doc, meta, dist, doc_id in zip(
            results['documents'],
            results['metadatas'],
            results['distances'],
            results['ids']
        ):
            if dist <= threshold:
                filtered['documents'].append(doc)
                filtered['metadatas'].append(meta)
                filtered['distances'].append(dist)
                filtered['ids'].append(doc_id)
        
        print(f"🔍 Filtered to {len(filtered['documents'])} results (threshold: {threshold})")
        return filtered


def main():
    """
    Example usage and testing
    """
    # Initialize retriever
    retriever = DocumentRetriever()
    
    # Test queries
    test_queries = [
        "What are the medicinal properties of Neem?",
        "How is Tulsi used in Siddha medicine?",
        "What herbs are used for treating skin diseases?"
    ]
    
    for query in test_queries:
        print("\n" + "="*80)
        results = retriever.retrieve(query, top_k=3)
        
        # Get formatted context
        context = retriever.get_context_string(results)
        print(f"\n📄 Context for LLM:")
        print(context[:500] + "...")
        
        # Filter by threshold
        filtered = retriever.similarity_threshold_filter(results, threshold=0.5)
        print(f"\n   After filtering: {len(filtered['documents'])} highly relevant results")


if __name__ == "__main__":
    main()
