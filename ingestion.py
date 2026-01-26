import os
from typing import Dict
import pymupdf4llm
from docx import Document
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

class IngestionEngine:
    def __init__(self):
        # We might add config here later, like chunk sizes or specific parser settings
        pass

    def load_pdf(self, file_path: str) -> str:
        # pymupdf4llm is handy because it converts the PDF directly to markdown including tables
        # which is much better for RAG than just plain text extraction.
        try:
            return pymupdf4llm.to_markdown(file_path)
        except Exception as e:
            print(f"Failed to read PDF {file_path}: {e}")
            return ""

    def load_docx(self, file_path: str) -> str:
        try:
            doc = Document(file_path)
            # Just joining paragraphs with newlines for now.
            # Might need to handle tables specifically if we find they are being lost or mangled.
            full_text = [para.text for para in doc.paragraphs]
            return '\n'.join(full_text)
        except Exception as e:
            print(f"Failed to read DOCX {file_path}: {e}")
            return ""

    def load_epub(self, file_path: str) -> str:
        try:
            # ebooklib can be a bit chatty with warnings, but it works well.
            book = epub.read_epub(file_path)
            full_text = []
            
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    # EPUB content is HTML, so we need BeautifulSoup to strip tags and get clean text
                    soup = BeautifulSoup(item.get_body_content(), 'html.parser')
                    full_text.append(soup.get_text())
            
            return '\n'.join(full_text)
        except Exception as e:
            print(f"Failed to read EPUB {file_path}: {e}")
            return ""

    def load_text(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Failed to read text file {file_path}: {e}")
            return ""

    def process_file(self, file_path: str) -> Dict[str, str]:
        # Quick check if file exists to avoid blowing up later
        if not os.path.exists(file_path):
            return {"source": file_path, "content": "", "error": "File not found"}

        ext = os.path.splitext(file_path)[1].lower()
        content = ""
        
        # Route to the right loader based on extension
        if ext == '.pdf':
            content = self.load_pdf(file_path)
        elif ext == '.docx':
            content = self.load_docx(file_path)
        elif ext == '.epub':
            content = self.load_epub(file_path)
        elif ext in ['.txt', '.md']:
            content = self.load_text(file_path)
        else:
            return {"source": file_path, "content": "", "error": f"Unsupported format: {ext}"}
            
        return {"source": file_path, "content": content}

    def process_directory(self, directory_path: str) -> List[Dict[str, str]]:
        """
        Walks through a directory and processes all supported files.
        """
        results = []
        if not os.path.exists(directory_path):
            print(f"Directory not found: {directory_path}")
            return results

        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Skip hidden files or unrelated stuff if needed
                if file.startswith('.'):
                    continue
                
                print(f"Processing: {file_path}")
                result = self.process_file(file_path)
                if result['content']:
                    results.append(result)
        
        return results

if __name__ == "__main__":
    # A quick sanity check to run this file directly
    print("Ingestion Engine ready.")
