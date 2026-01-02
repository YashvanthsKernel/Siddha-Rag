"""
Complete RAG System
Combines retrieval and generation for question answering.
Supports both vector-only and hybrid (vector + graph) retrieval.
"""

from rag.retriever import DocumentRetriever
from rag.generator import ResponseGenerator
from typing import Dict, List, Optional
import json
import os


class SiddhaRAG:
    """
    Complete RAG (Retrieval-Augmented Generation) system for Siddha medicine.
    Combines document retrieval and LLM generation.
    """
    
    def __init__(
        self,
        db_path: str = "data/vectordb",
        collection_name: str = "siddha_knowledge",
        llm_model: str = "llama3.2:3b",
        embedding_model: str = "all-MiniLM-L6-v2",  # Must match ChromaDB embeddings (384d)
        temperature: float = 0.7,
        use_graph: bool = False,  # NEW: Enable hybrid retrieval
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None
    ):
        """
        Initialize the RAG system.
        
        Args:
            db_path: Path to vector database
            collection_name: Name of ChromaDB collection
            llm_model: Ollama model for generation
            embedding_model: Ollama model for embeddings
            temperature: LLM temperature
            use_graph: Enable hybrid vector+graph retrieval (requires Neo4j)
            neo4j_uri: Neo4j connection URI (default: bolt://localhost:7687)
            neo4j_user: Neo4j username (default: neo4j)
            neo4j_password: Neo4j password (required if use_graph=True)
        """
        print("🚀 Initializing Siddha RAG System...\n")
        
        self.use_graph = use_graph
        
        # Initialize retriever (hybrid or vector-only)
        if use_graph:
            try:
                # Import hybrid retriever
                from graph.hybrid_retriever import HybridRetriever
                
                # Check password is provided
                if not neo4j_password:
                    neo4j_password = os.getenv("NEO4J_PASSWORD")
                    if not neo4j_password:
                        print("⚠️ Warning: Neo4j password not provided")
                        print("   Set NEO4J_PASSWORD environment variable or pass neo4j_password parameter")
                        print("   Falling back to vector-only retrieval")
                        self.use_graph = False
                
                if self.use_graph:
                    self.retriever = HybridRetriever(
                        chroma_path=db_path,
                        chroma_collection=collection_name,
                        neo4j_uri=neo4j_uri,
                        neo4j_user=neo4j_user,
                        neo4j_password=neo4j_password,
                        embedding_model=embedding_model
                    )
                    print("✅ Hybrid retrieval enabled (Vector + Graph)")
                else:
                    # Fallback to vector-only
                    self.retriever = DocumentRetriever(
                        db_path=db_path,
                        collection_name=collection_name,
                        embedding_model=embedding_model
                    )
            except ImportError as e:
                print(f"⚠️ Graph retrieval not available: {e}")
                print("   Install dependencies: pip install -r requirements-graph.txt")
                print("   Falling back to vector-only retrieval")
                self.use_graph = False
                self.retriever = DocumentRetriever(
                    db_path=db_path,
                    collection_name=collection_name,
                    embedding_model=embedding_model
                )
            except Exception as e:
                print(f"⚠️ Failed to initialize graph retrieval: {e}")
                print("   Falling back to vector-only retrieval")
                self.use_graph = False
                self.retriever = DocumentRetriever(
                    db_path=db_path,
                    collection_name=collection_name,
                    embedding_model=embedding_model
                )
        else:
            # Vector-only mode (original behavior)
            self.retriever = DocumentRetriever(
                db_path=db_path,
                collection_name=collection_name,
                embedding_model=embedding_model
            )
            print("✅ Vector-only retrieval (legacy mode)")
        
        # Initialize generator
        self.generator = ResponseGenerator(
            model_name=llm_model,
            temperature=temperature
        )
        
        print("\n✅ RAG System ready!")
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
        include_sources: bool = True,
        strategy: str = "hybrid"  # NEW: "vector", "graph", or "hybrid"
    ) -> Dict:
        """
        Answer a question using RAG.
        
        Args:
            question: User's question
            top_k: Number of documents to retrieve
            similarity_threshold: Optional filter for similarity (lower = more similar)
            include_sources: Whether to include source citations in response
            strategy: Retrieval strategy - "vector", "graph", or "hybrid" (default)
                     Only applies if use_graph=True, otherwise uses vector
            
        Returns:
            Dictionary with question, answer, sources, and metadata
        """
        print("\n" + "="*80)
        print(f"❓ Question: {question}")
        if self.use_graph:
            print(f"🔍 Strategy: {strategy}")
        print("="*80)
        
        # Step 1: Retrieve relevant documents
        if self.use_graph:
            # Hybrid retrieval (vector + graph)
            retrieval_results = self.retriever.retrieve(
                query=question,
                top_k=top_k,
                strategy=strategy
            )
            
            # Extract documents for LLM context
            if strategy == "hybrid" and 'vector_results' in retrieval_results:
                documents = retrieval_results['vector_results'].get('documents', [])
                metadatas = retrieval_results['vector_results'].get('metadatas', [])
                distances = retrieval_results['vector_results'].get('distances', [])
                
                # Add graph facts to context
                graph_facts = retrieval_results.get('graph_results', {}).get('facts', [])
                entities_found = retrieval_results.get('entities_found', [])
                
            elif strategy == "graph":
                # Graph-only: no traditional documents, build from graph facts
                documents = []
                metadatas = []
                distances = []
                graph_facts = retrieval_results.get('facts', [])
                entities_found = retrieval_results.get('entities', [])
                
                # Convert graph facts to text chunks
                for fact in graph_facts:
                    fact_text = self._format_graph_fact(fact)
                    if fact_text:
                        documents.append(fact_text)
                        metadatas.append({'source': 'knowledge_graph', 'entity': fact.get('entity', '')})
                        distances.append(0.0)  # Graph facts have perfect relevance
                
            else:  # vector
                documents = retrieval_results.get('documents', [])
                metadatas = retrieval_results.get('metadatas', [])
                distances = retrieval_results.get('distances', [])
                graph_facts = []
                entities_found = []
        else:
            # Vector-only retrieval (original behavior)
            retrieval_results = self.retriever.retrieve(question, top_k=top_k)
            documents = retrieval_results.get('documents', [])
            metadatas = retrieval_results.get('metadatas', [])
            distances = retrieval_results.get('distances', [])
            graph_facts = []
            entities_found = []
        
        # Apply similarity threshold if specified (vector results only)
        if similarity_threshold is not None and distances:
            filtered_docs = []
            filtered_metas = []
            filtered_dists = []
            
            for doc, meta, dist in zip(documents, metadatas, distances):
                if dist <= similarity_threshold:
                    filtered_docs.append(doc)
                    filtered_metas.append(meta)
                    filtered_dists.append(dist)
            
            documents = filtered_docs
            metadatas = filtered_metas
            distances = filtered_dists
        
        # Check if we have results
        if not documents:
            return {
                'question': question,
                'answer': "I couldn't find relevant information in the knowledge base to answer this question.",
                'sources': [],
                'num_sources': 0,
                'strategy': strategy if self.use_graph else 'vector'
            }
        
        # Step 2: Generate answer with retrieved context
        answer = self.generator.generate(
            query=question,
            context_chunks=documents,
            include_sources=include_sources
        )
        
        # Prepare response
        response = {
            'question': question,
            'answer': answer,
            'sources': metadatas,
            'num_sources': len(documents),
            'distances': distances,
            'retrieved_chunks': documents if include_sources else None,
            'strategy': strategy if self.use_graph else 'vector'
        }
        
        # Add graph-specific data if available
        if self.use_graph and graph_facts:
            response['graph_facts'] = graph_facts
            response['entities_found'] = entities_found
        
        return response
    
    def _format_graph_fact(self, fact: Dict) -> str:
        """
        Format a graph fact as readable text.
        
        Args:
            fact: Graph fact dictionary
            
        Returns:
            Formatted text string
        """
        entity = fact.get('entity', '')
        entity_type = fact.get('entity_type', '')
        relationships = fact.get('relationships', []) or fact.get('direct_facts', [])
        
        if not relationships:
            return f"{entity} is a {entity_type}."
        
        lines = [f"{entity} ({entity_type}):"]
        for rel in relationships[:5]:  # Limit to 5 relationships
            rel_type = rel.get('relationship', 'RELATED_TO')
            related = rel.get('related_entity', '')
            if related:
                lines.append(f"  - {rel_type.replace('_', ' ').lower()}: {related}")
        
        return "\n".join(lines)
    
    def query_with_history(
        self,
        question: str,
        chat_history: List[Dict[str, str]],
        top_k: int = 5
    ) -> Dict:
        """
        Answer question with conversation history.
        
        Args:
            question: Current question
            chat_history: Previous messages
            top_k: Number of documents to retrieve
            
        Returns:
            Response dictionary
        """
        # Retrieve documents
        retrieval_results = self.retriever.retrieve(question, top_k=top_k)
        
        if not retrieval_results['documents']:
            return {
                'question': question,
                'answer': "No relevant information found.",
                'sources': []
            }
        
        # Generate with history
        answer = self.generator.generate_with_chat_history(
            query=question,
            context_chunks=retrieval_results['documents'],
            chat_history=chat_history
        )
        
        return {
            'question': question,
            'answer': answer,
            'sources': retrieval_results['metadatas']
        }
    
    def display_response(self, response: Dict):
        """
        Pretty-print a RAG response.
        
        Args:
            response: Response dictionary from query()
        """
        print(f"\n{'='*80}")
        print(f"❓ Question: {response['question']}")
        print(f"{'='*80}\n")
        
        print(f"💬 Answer:\n{response['answer']}\n")
        
        if response.get('sources'):
            print(f"📚 Sources ({response['num_sources']} documents):")
            for i, source in enumerate(response['sources'], 1):
                filename = source.get('filename', 'unknown')
                chunk_idx = source.get('chunk_index', '?')
                print(f"   {i}. {filename} (chunk {chunk_idx})")
        
        if response.get('distances'):
            print(f"\n🎯 Relevance scores (lower = more relevant):")
            for i, dist in enumerate(response['distances'], 1):
                print(f"   {i}. {dist:.4f}")
    
    def interactive_mode(self):
        """
        Run interactive Q&A session.
        """
        print("\n" + "="*80)
        print("🩺 Siddha Medicine RAG System - Interactive Mode")
        print("="*80)
        print("\nType your questions about Siddha medicine.")
        print("Commands: 'quit' or 'exit' to stop, 'history' to see chat history\n")
        
        chat_history = []
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if question.lower() == 'history':
                    print("\n📜 Chat History:")
                    for msg in chat_history:
                        role = "You" if msg['role'] == 'user' else "Bot"
                        print(f"   {role}: {msg['content'][:100]}...")
                    continue
                
                # Query the system (retrieve 10 chunks for better answers)
                response = self.query(question, top_k=10)
                
                # Display response
                self.display_response(response)
                
                # Update history
                chat_history.append({'role': 'user', 'content': question})
                chat_history.append({'role': 'assistant', 'content': response['answer']})
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def save_response(self, response: Dict, output_file: str):
        """
        Save response to JSON file.
        
        Args:
            response: Response dictionary
            output_file: Path to output file
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(response, f, indent=2, ensure_ascii=False)
        print(f"💾 Response saved to {output_file}")


def main():
    """
    Example usage and testing
    """
    # Initialize RAG system
    rag = SiddhaRAG()
    
    # Test queries
    test_questions = [
        "What are the medicinal properties of Neem?",
        "How is Tulsi used in treating respiratory conditions?",
        "What herbs are recommended for skin diseases in Siddha medicine?"
    ]
    
    # Query and display results
    for question in test_questions:
        response = rag.query(question, top_k=3)
        rag.display_response(response)
        print("\n" + "="*80 + "\n")
    
    # Optional: Run interactive mode
    # Uncomment to enable:
    # rag.interactive_mode()


if __name__ == "__main__":
    main()
