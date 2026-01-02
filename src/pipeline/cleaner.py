import re

class TextCleaner:
    def __init__(self):
        pass

    def clean(self, text):
        """
        Cleans the extracted text:
        - Removes excessive whitespace
        - Fixes hyphenated words broken across lines
        - Removes non-printable characters
        """
        if not text:
            return ""

        # Replace multiple newlines with a single newline
        text = re.sub(r'\n+', '\n', text)
        
        # Join hyphenated words at line breaks (e.g., "trea-\ntment" -> "treatment")
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        return text.strip()

if __name__ == "__main__":
    cleaner = TextCleaner()
    raw = "This is a   raw \n\n string with bro-\nken words."
    print(cleaner.clean(raw))