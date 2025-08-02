import re
import logging
from pathlib import Path
from typing import List, Dict, Any
from io import BytesIO

# FastAPI imports
from fastapi import UploadFile, HTTPException

# Document processing
import fitz  # pymupdf
from docx import Document
import email
from email.policy import default

# Optional OCR support
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

from config import config

logger = logging.getLogger(__name__)

class AdvancedDocumentProcessor:
    """Enhanced document processor with OCR and better text extraction"""

    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.eml', '.txt']
        if OCR_AVAILABLE:
            self.supported_formats.extend(['.png', '.jpg', '.jpeg', '.tiff'])

    async def process_document(self, file: UploadFile) -> str:
        """Enhanced document processing with OCR for images"""
        if file.size > config.MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")

        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in self.supported_formats:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        content = await file.read()

        try:
            if file_extension == '.pdf':
                return self._extract_pdf_text(content)
            elif file_extension == '.docx':
                return self._extract_docx_text(content)
            elif file_extension == '.eml':
                return self._extract_email_text(content)
            elif file_extension == '.txt':
                return content.decode('utf-8')
            elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff']:
                return await self._extract_image_text(content)
        except Exception as e:
            logger.error(f"Error processing {file_extension} file: {str(e)}")
            raise HTTPException(status_code=422, detail=f"Error processing file: {str(e)}")

    def _extract_pdf_text(self, content: bytes) -> str:
        """Enhanced PDF text extraction with OCR fallback"""
        doc = fitz.open(stream=content, filetype="pdf")
        text = ""

        for page_num, page in enumerate(doc):
            # Try regular text extraction first
            page_text = page.get_text()

            # If no text found and OCR is available, try OCR on the page image
            if len(page_text.strip()) < 50 and OCR_AVAILABLE:
                try:
                    # Convert page to image and OCR
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")

                    image = Image.open(BytesIO(img_data))
                    ocr_text = pytesseract.image_to_string(image)
                    page_text = ocr_text

                except Exception as e:
                    logger.error(f"OCR failed for page {page_num}: {e}")

            text += page_text + "\n"

        doc.close()
        return self._clean_text(text)

    def _extract_docx_text(self, content: bytes) -> str:
        """Extract text from DOCX file"""
        doc = Document(BytesIO(content))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return self._clean_text(text)

    def _extract_email_text(self, content: bytes) -> str:
        """Extract text from email file"""
        msg = email.message_from_bytes(content, policy=default)
        text = ""

        # Extract basic headers
        text += f"Subject: {msg.get('Subject', 'No Subject')}\n"
        text += f"From: {msg.get('From', 'Unknown')}\n"
        text += f"Date: {msg.get('Date', 'Unknown')}\n\n"

        # Extract body
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    text += part.get_content() + "\n"
        else:
            text += msg.get_content() + "\n"

        return self._clean_text(text)

    async def _extract_image_text(self, content: bytes) -> str:
        """Extract text from images using OCR"""
        if not OCR_AVAILABLE:
            raise HTTPException(
                status_code=422,
                detail="OCR not available. Install pytesseract and tesseract-ocr"
            )

        try:
            image = Image.open(BytesIO(content))
            text = pytesseract.image_to_string(image)
            return self._clean_text(text)
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Error processing image: {str(e)}"
            )

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might interfere
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\"\'\/\\\$\%\&\*\+\=\@\#]', ' ', text)
        return text.strip()