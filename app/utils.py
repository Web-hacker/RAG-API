
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
from typing import List, Tuple

# Set path to Tesseract OCR executable (required for Windows systems)
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"


def hybrid_pdf_extraction(doc,
        # pdf_path: str ,
          ocr_threshold: int = 30) -> str:
    """
    Extracts text from PDFs using direct text extraction first. Falls back to OCR if not enough text is found.
    
    Args:
    - pdf_path: Path to the PDF file.
    - ocr_threshold: Minimum text length to accept before triggering OCR.

    Returns:
    - Combined text extracted using both methods.
    """
    # doc = fitz.open(pdf_path)
    complete_text = ""

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text().strip()
        

        if text and len(text) >= ocr_threshold:
            complete_text += text
        else:
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_text = pytesseract.image_to_string(img).strip()
            if ocr_text:
                complete_text += ocr_text

    return complete_text

def chunk_text(text: str, max_tokens: int = 500, overlap: int = 100) -> List[str]:
    """
    Splits long text into overlapping chunks for embedding.

    Args:
    - text: The text to chunk.
    - max_tokens: Maximum number of words per chunk.
    - overlap: Number of words to overlap between chunks.

    Returns:
    - A list of text chunks.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + max_tokens
        chunks.append(" ".join(words[start:end]))
        start += max_tokens - overlap

    return chunks

def extract_text_from_image(img
                            # ,path: str
                            ) -> str:
    """
    Uses OCR to extract text from an image file.

    Args:
    - path: Path to the image.

    Returns:
    - Extracted text.
    """
    try:
        # img = Image.open(path)
        return pytesseract.image_to_string(img)
    except Exception as e:
        print(f"Failed to process image {img}: {e}")
        return ""

def load_text_files_from_dir(
    dir_path: str, 
    allowed_exts={".md", ".txt", ".json", ".pdf"}
) -> List[Tuple[str, str]]:
    """
    Loads and extracts text content from all files in a directory recursively.

    Args:
    - dir_path: Path to the directory.
    - allowed_exts: Supported file extensions.

    Returns:
    - List of (file_path, extracted_text) tuples.
    """
    docs = []
    for filepath in Path(dir_path).rglob("*"):
        ext = filepath.suffix.lower()
        if ext in allowed_exts and filepath.is_file():
            try:
                if ext == ".pdf":
                    text = hybrid_pdf_extraction(str(filepath))
                elif ext in {".png", ".jpg", ".jpeg"}:
                    ocr_text = extract_text_from_image(str(filepath))
                    text = f"[OCR]\n{ocr_text.strip()}"
                else:
                    text = filepath.read_text(encoding="utf-8", errors="ignore")

                if text.strip():
                    docs.append((str(filepath), text))

            except Exception as e:
                print(f"Failed to read {filepath}: {e}")

    return docs

def load_files_from_file_path(
    file_path: str,
    allowed_exts={".md", ".txt", ".json", ".pdf"}
) -> List[Tuple[str, str]]:
    """
    Loads and extracts text from a single file path.

    Args:
    - file_path: Path to the file.
    - allowed_exts: Supported extensions.

    Returns:
    - List containing one (file_path, text) tuple if valid.
    """
    docs = []
    filepath = Path(file_path)
    ext = filepath.suffix.lower()

    if ext in allowed_exts and filepath.is_file():
        try:
            if ext == ".pdf":
                text = hybrid_pdf_extraction(str(filepath))
            elif ext in {".png", ".jpg", ".jpeg"}:
                ocr_text = extract_text_from_image(str(filepath))
                text = f"[OCR]\n{ocr_text.strip()}"
            else:
                text = filepath.read_text(encoding="utf-8", errors="ignore")

            if text.strip():
                docs.append((str(filepath), text))

        except Exception as e:
            print(f"Failed to read {filepath}: {e}")

    return docs


