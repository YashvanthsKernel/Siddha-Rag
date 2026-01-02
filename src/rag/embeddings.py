"""
Embeddings Module
Generates vector embeddings using Sentence Transformers and stores them in ChromaDB.
"""

from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict, Optional
from tqdm import tqdm


class EmbeddingManager:
    """
    Manages embedding generation and vector database operations.
    Uses Sentence Transformers for embedding generation and ChromaDB for storage.
    """
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        db_path: str = "data/vectordb",
        collection_name: str = "siddha_knowledge"
    ):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: Name of the Sentence Transformer model
            db_path: Path to ChromaDB persistent storage
            collection_name: Name of the collection in ChromaDB
        """
        self.model_name = model_name
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Initialize Sentence Transformer model
        print(f"🔄 Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print(f"✅ Model loaded successfully!")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        
        print(f"✅ Initialized EmbeddingManager with model: {model_name}")
        print(f"   Database path: {db_path}")
        print(f"   Embedding dimensions: {self.model.get_sentence_embedding_dimension()}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as list of floats
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently using batch processing.
        
        Args:
            texts: List of input texts
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embedding vectors
        """
        print(f"\n🧠 Generating embeddings for {len(texts)} texts...")
        print(f"   Using batch size: {batch_size}")
        
        # Sentence Transformers handles batching internally and efficiently
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Convert numpy arrays to lists
        return [emb.tolist() for emb in embeddings]
    
    def create_collection(self, reset: bool = False):
        """
        Create or get the ChromaDB collection.
        
        Args:
            reset: If True, delete existing collection and create new one
            
        Returns:
            ChromaDB collection object
        """
        if reset:
            try:
                self.client.delete_collection(name=self.collection_name)
                print(f"🗑️  Deleted existing collection: {self.collection_name}")
            except:
                pass
        
        collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "Siddha medical knowledge base",
                "embedding_model": self.model_name
            }
        )
        
        print(f"✅ Collection ready: {self.collection_name}")
        return collection
    
    def store_chunks(
        self, 
        chunks: List[Dict], 
        reset_collection: bool = False,
        batch_size: int = 100
    ):
        """
        Generate embeddings for chunks and store in ChromaDB.
        
        Args:
            chunks: List of chunk dictionaries with 'text', 'chunk_id', and metadata
            reset_collection: Whether to reset the collection before storing
            batch_size: Number of chunks to process in each batch
            
        Returns:
            ChromaDB collection object
        """
        # Create or get collection
        collection = self.create_collection(reset=reset_collection)
        
        # Prepare data
        texts = [chunk['text'] for chunk in chunks]
        ids = [chunk['chunk_id'] for chunk in chunks]
        metadatas = [
            {
                'source': chunk.get('source', ''),
                'filename': chunk.get('filename', ''),
                'chunk_index': chunk.get('chunk_index', 0),
                'total_chunks': chunk.get('total_chunks', 0)
            } 
            for chunk in chunks
        ]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Store in batches
        print(f"\n💾 Storing {len(chunks)} chunks in ChromaDB...")
        
        for i in range(0, len(chunks), batch_size):
            batch_end = min(i + batch_size, len(chunks))
            
            collection.add(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end],
                documents=texts[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )
            
            print(f"   Stored batch {i//batch_size + 1}: {batch_end}/{len(chunks)} chunks")
        
        print(f"✅ Successfully stored {len(chunks)} chunks in vector database")
        
        # Verify storage
        count = collection.count()
        print(f"📊 Total documents in collection: {count}")
        
        return collection
    
    def get_collection(self):
        """
        Get existing collection.
        
        Returns:
            ChromaDB collection object or None if doesn't exist
        """
        try:
            collection = self.client.get_collection(name=self.collection_name)
            print(f"✅ Retrieved collection: {self.collection_name}")
            print(f"   Total documents: {collection.count()}")
            return collection
        except Exception as e:
            print(f"❌ Collection not found: {e}")
            return None
    
    def list_collections(self):
        """
        List all collections in the database.
        
        Returns:
            List of collection names
        """
        collections = self.client.list_collections()
        print(f"\n📚 Available collections:")
        for coll in collections:
            print(f"   - {coll.name}: {coll.count()} documents")
        return collections


def main():
    """
    Example usage and testing
    """
    # Sample chunks
    sample_chunks = [
        {
            'chunk_id': 'sample_chunk_0',
            'text': 'Neem (Azadirachta indica) is widely used in Siddha medicine for treating skin diseases.',
            'source': 'sample.pdf',
            'filename': 'sample.pdf',
            'chunk_index': 0,
            'total_chunks': 2
        },
        {
            'chunk_id': 'sample_chunk_1',
            'text': 'Tulsi (Ocimum sanctum) is considered sacred and is used for respiratory conditions.',
            'source': 'sample.pdf',
            'filename': 'sample.pdf',
            'chunk_index': 1,
            'total_chunks': 2
        }
    ]
    
    # Initialize embedding manager
    embedding_manager = EmbeddingManager()
    
    # Store chunks
    collection = embedding_manager.store_chunks(
        sample_chunks, 
        reset_collection=True
    )
    
    # List collections
    embedding_manager.list_collections()
    
    print("\n✅ Embedding manager test complete!")


if __name__ == "__main__":
    main()
