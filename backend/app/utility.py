from io import BytesIO
from pypdf import PdfReader

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    texts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        texts.append(page_text)
    return "\n".join(texts).strip()
