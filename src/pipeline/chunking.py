"""
Text Chunking Module
Splits large documents into smaller chunks for embedding and retrieval.
Uses LangChain's RecursiveCharacterTextSplitter for intelligent chunking.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import os
from typing import List, Dict


class TextChunker:
    """
    Handles document chunking with configurable chunk size and overlap.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # RecursiveCharacterTextSplitter tries to split on different separators
        # in order: paragraph breaks, newlines, sentences, words
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            is_separator_regex=False
        )
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split a single text into chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        return self.splitter.split_text(text)
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Split multiple documents into chunks with metadata.
        
        Args:
            documents: List of document dictionaries with 'text', 'filename', 'source' keys
            
        Returns:
            List of chunk dictionaries with text, metadata, and unique IDs
        """
        all_chunks = []
        
        for doc in documents:
            text = doc.get('text', '')
            filename = doc.get('filename', 'unknown')
            source = doc.get('source', 'unknown')
            
            # Split document into chunks
            chunks = self.chunk_text(text)
            
            # Create chunk objects with metadata
            for i, chunk in enumerate(chunks):
                chunk_obj = {
                    'chunk_id': f"{filename}_chunk_{i}",
                    'text': chunk,
                    'source': source,
                    'filename': filename,
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
                all_chunks.append(chunk_obj)
        
        return all_chunks
    
    def save_chunks(self, chunks: List[Dict], output_dir: str = "data/processed/chunks"):
        """
        Save chunks to a JSON file.
        
        Args:
            chunks: List of chunk dictionaries
            output_dir: Directory to save chunks
            
        Returns:
            Path to saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'chunks.json')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved {len(chunks)} chunks to {output_file}")
        return output_file
    
    def load_chunks(self, chunks_file: str = "data/processed/chunks/chunks.json") -> List[Dict]:
        """
        Load chunks from a JSON file.
        
        Args:
            chunks_file: Path to chunks JSON file
            
        Returns:
            List of chunk dictionaries
        """
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        print(f"✅ Loaded {len(chunks)} chunks from {chunks_file}")
        return chunks


def main():
    """
    Example usage and testing
    """
    # Sample documents
    sample_docs = [
        {
            'filename': 'sample1.txt',
            'text': 'Neem is a powerful medicinal plant in Siddha medicine. ' * 50,
            'source': 'data/raw/sample1.txt'
        },
        {
            'filename': 'sample2.txt',
            'text': 'Tulsi is known for its healing properties in respiratory conditions. ' * 50,
            'source': 'data/raw/sample2.txt'
        }
    ]
    
    # Initialize chunker
    chunker = TextChunker(chunk_size=200, chunk_overlap=50)
    
    # Chunk documents
    chunks = chunker.chunk_documents(sample_docs)
    
    # Display results
    print(f"\n📊 Chunking Results:")
    print(f"   - Total chunks created: {len(chunks)}")
    print(f"\n📝 Sample chunk:")
    print(f"   ID: {chunks[0]['chunk_id']}")
    print(f"   Text: {chunks[0]['text'][:100]}...")
    print(f"   Source: {chunks[0]['source']}")
    
    # Save chunks
    chunker.save_chunks(chunks)


if __name__ == "__main__":
    main()
