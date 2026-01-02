import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import logging

class OCREngine:
    def __init__(self, tesseract_cmd=None):
        """
        OCR Engine using Tesseract.
        Args:
            tesseract_cmd: Path to tesseract executable (optional)
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        self.logger = logging.getLogger(__name__)

    def extract_text(self, file_path):
        """
        Extract text from PDF or image file.
        Args:
            file_path: Path to the file
        Returns:
            Extracted text as string
        """
        try:
            if file_path.lower().endswith('.pdf'):
                return self._extract_from_pdf(file_path)
            elif file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                return self._extract_from_image(file_path)
            else:
                self.logger.error(f"Unsupported file format: {file_path}")
                return ""
        except Exception as e:
            self.logger.error(f"Error extracting text from {file_path}: {e}")
            return ""

    def _extract_from_pdf(self, pdf_path):
        """Extract text from PDF using pdf2image and Tesseract"""
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path)
            text = ""
            
            for i, image in enumerate(images):
                self.logger.info(f"Processing page {i+1}/{len(images)}")
                # OCR each page
                page_text = pytesseract.image_to_string(image, lang='tam+eng')
                text += page_text + "\n\n"
            
            return text
        except Exception as e:
            self.logger.error(f"Error processing PDF {pdf_path}: {e}")
            return ""

    def _extract_from_image(self, image_path):
        """Extract text from image using Tesseract"""
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang='tam+eng')
            return text
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            return ""

if __name__ == "__main__":
    # Test OCR
    logging.basicConfig(level=logging.INFO)
    ocr = OCREngine()
    # Add test file path here if needed
    print("OCR Engine initialized successfully")
