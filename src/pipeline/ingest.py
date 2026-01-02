import os
import json
import logging
from pathlib import Path
from typing import List, Dict

# Import our custom modules
from ocr_engine import OCREngine
from cleaner import TextCleaner
from graph_builder import GraphBuilder

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parents[2]  # Points to Siddha LLM root
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED_TEXT = BASE_DIR / "data" / "processed" / "cleaned_text"
DATA_ENTITIES = BASE_DIR / "data" / "processed" / "entities_and_relations.json"

# Ensure directories exist
DATA_PROCESSED_TEXT.mkdir(parents=True, exist_ok=True)


class DocumentIngester:
    """
    Document ingestion class for reading PDFs and DOCX files.
    Compatible with the main RAG pipeline.
    """
    
    def __init__(self, raw_data_dir: str = None):
        """
        Initialize the document ingester.
        
        Args:
            raw_data_dir: Directory containing raw documents (default: data/raw)
        """
        if raw_data_dir:
            self.raw_data_dir = Path(raw_data_dir)
        else:
            self.raw_data_dir = DATA_RAW
        
        self.ocr_engine = OCREngine()
    
    def read_pdf(self, file_path: str) -> str:
        """
        Extract text from PDF using OCR engine.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        return self.ocr_engine.extract_text(file_path)
    
    def read_docx(self, file_path: str) -> str:
        """
        Extract text from DOCX file.
        
        Args:
            file_path: Path to DOCX file
            
        Returns:
            Extracted text
        """
        try:
            from docx import Document
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        except Exception as e:
            logging.error(f"Error reading DOCX {file_path}: {e}")
            return ""
    
    def process_all_documents(self) -> List[Dict]:
        """
        Process all documents in the raw data directory.
        
        Returns:
            List of document dictionaries with text, filename, and source
        """
        documents = []
        
        # Get all PDF and DOCX files
        pdf_files = list(self.raw_data_dir.glob("*.pdf"))
        docx_files = list(self.raw_data_dir.glob("*.docx"))
        all_files = pdf_files + docx_files
        
        if not all_files:
            logging.warning(f"⚠️  No PDF or DOCX files found in {self.raw_data_dir}")
            return documents
        
        logging.info(f"📖 Found {len(all_files)} documents to process")
        
        for file_path in all_files:
            filename = file_path.name
            logging.info(f"   Reading: {filename}")
            
            # Extract text based on file type
            if filename.endswith('.pdf'):
                text = self.read_pdf(str(file_path))
            elif filename.endswith('.docx'):
                text = self.read_docx(str(file_path))
            else:
                continue
            
            if not text:
                logging.warning(f"   ⚠️  Empty content from {filename}")
                continue
            
            documents.append({
                'filename': filename,
                'text': text,
                'source': str(file_path)
            })
        
        logging.info(f"✅ Successfully processed {len(documents)} documents")
        return documents


def run_ingestion():
    """
    Main execution loop (legacy function):
    1. Scan data/raw for PDFs and images
    2. OCR -> Text
    3. Clean Text
    4. Save Cleaned Text
    5. Extract Graph Relations -> JSON
    """
    
    # Initialize Processors
    ocr = OCREngine()
    cleaner = TextCleaner()
    graph_builder = GraphBuilder()
    
    all_relations = []
    
    # Get list of PDF and image files
    pdf_files = list(DATA_RAW.glob("*.pdf"))
    image_files = list(DATA_RAW.glob("*.png")) + list(DATA_RAW.glob("*.jpg")) + list(DATA_RAW.glob("*.jpeg"))
    all_files = pdf_files + image_files
    
    if not all_files:
        logging.warning(f"⚠️  No PDF or image files found in {DATA_RAW}")
        logging.info(f"📂 Please add your Siddha medicine documents to: {DATA_RAW}")
        return

    logging.info(f"🚀 Starting Ingestion for {len(all_files)} files...")

    for file_path in all_files:
        filename = file_path.name
        file_stem = file_path.stem
        
        logging.info(f"--- Processing: {filename} ---")
        
        # STEP 1: OCR Extraction
        raw_text = ocr.extract_text(str(file_path))
        if not raw_text:
            logging.error(f"❌ Skipping {filename} due to empty OCR output.")
            continue
            
        # STEP 2: Cleaning
        cleaned_text = cleaner.clean(raw_text)
        
        # STEP 3: Save Cleaned Text (for VectorDB later)
        output_text_path = DATA_PROCESSED_TEXT / f"{file_stem}.txt"
        with open(output_text_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        logging.info(f"✅ Saved cleaned text to: {output_text_path}")
        
        # STEP 4: Graph Entity Extraction
        # In a real app, you might process text in chunks (e.g., 512 tokens) here
        relations = graph_builder.extract_relations(cleaned_text)
        
        # Add metadata (which file did this come from?)
        for rel in relations:
            rel["source_document"] = filename
            
        all_relations.extend(relations)
        logging.info(f"🔍 Extracted {len(relations)} relations from {filename}")

    # STEP 5: Save aggregated Graph Data
    # This JSON can be imported into Neo4j or NetworkX later
    with open(DATA_ENTITIES, "w", encoding="utf-8") as f:
        json.dump(all_relations, f, indent=4)
        
    logging.info(f"🎉 Ingestion Complete! Graph data saved to {DATA_ENTITIES}")
    logging.info(f"📊 Total relations extracted: {len(all_relations)}")

if __name__ == "__main__":
    run_ingestion()