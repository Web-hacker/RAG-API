# app/utils.py

from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import torch
import io
import os
import shutil

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def hybrid_pdf_extraction(pdf_path: str, ocr_threshold: int = 30) -> list[str]:
   
    doc = fitz.open(pdf_path)
    complete_text = ""

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text().strip()
        print(text)
        if text and len(text) >= ocr_threshold:
            complete_text += text
        else:
            # Render the page to an image
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))

            # OCR the image
            ocr_text = pytesseract.image_to_string(img).strip()
            print(ocr_text)
            if ocr_text:
                complete_text += ocr_text

    return complete_text


def chunk_text(text: str, max_tokens: int = 500, overlap: int = 100) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_tokens
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap
    return chunks

def extract_text_from_image(path: str) -> str:
    try:
        img = Image.open(path)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print(f"Failed to process image {path}: {e}")
        return ""




def load_text_files_from_dir(dir_path: str, allowed_exts={".md", ".txt", ".js",".html", ".css", ".cpp", ".py", ".json", ".pdf", ".png", ".jpg", ".jpeg"}) -> list[tuple[str, str]]:
    docs = []
    for filepath in Path(dir_path).rglob("*"):
        ext = filepath.suffix.lower()
        if ext in allowed_exts and filepath.is_file():
            try:
                if ext == ".pdf":
                    text = hybrid_pdf_extraction(str(filepath))
                elif ext in [".png", ".jpg", ".jpeg"]:
                    ocr_text = extract_text_from_image(str(filepath))
                    text = f"[OCR]\n{ocr_text.strip()}"
                else:
                    text = filepath.read_text(encoding="utf-8", errors="ignore")
                if text.strip():
                    docs.append((str(filepath), text))
            except Exception as e:
                print(f"Failed to read {filepath}: {e}")
    return docs

def load_files_from_file_path(file_path: str, allowed_exts={".md", ".txt", ".js",".html", ".css", ".cpp", ".py", ".json", ".pdf", ".png", ".jpg", ".jpeg"}) -> list[tuple[str, str]]:
    docs = []
    filepath = Path(file_path)
    ext = filepath.suffix.lower()
    if ext in allowed_exts and filepath.is_file():
            try:
                if ext == ".pdf":
                    text = hybrid_pdf_extraction(str(filepath))
                elif ext in [".png", ".jpg", ".jpeg"]:
                    ocr_text = extract_text_from_image(str(filepath))
                    text = f"[OCR]\n{ocr_text.strip()}"
                else:
                    text = filepath.read_text(encoding="utf-8", errors="ignore")
                if text.strip():
                    docs.append((str(filepath), text))
            except Exception as e:
                print(f"Failed to read {filepath}: {e}")
    return docs


def cleanup_repo(local_path: str):
    if os.path.exists(local_path):
        print(f"Cleaning up cloned repo at {local_path}...")
        shutil.rmtree(local_path)