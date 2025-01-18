import os
import fitz  # PyMuPDF
from PyPDF2 import PdfReader, PdfWriter
import streamlit as st
from io import BytesIO

from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

@st.cache_data
def get_pdf_text(pdf_path: str) -> str:
    """
    Extract text from the given PDF file using PyPDF2.
    """
    text = ""
    try:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        st.error(f"Lỗi trích xuất văn bản từ '{pdf_path}': {e}")
    return text

@st.cache_data
def get_text_chunks(text: str):
    """
    Split large text into smaller chunks for ingestion into retriever or vector store.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=5000,
        chunk_overlap=500,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_documents(chunks):
    """
    Create Document objects (LangChain) from chunks of text.
    """
    docs = [Document(page_content=chunk) for chunk in chunks]
    return docs

def split_pdf_into_pages(pdf_path: str, output_dir: str):
    """
    Split a multi-page PDF into individual PDF files (one file per page).
    """
    try:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        os.makedirs(output_dir, exist_ok=True)
        for i in range(total_pages):
            writer = PdfWriter()
            writer.add_page(reader.pages[i])
            output_filename = f"page_{i+1:03}.pdf"
            with open(os.path.join(output_dir, output_filename), "wb") as f:
                writer.write(f)
        st.success(f"Đã tách thành công {total_pages} trang từ '{pdf_path}' vào '{output_dir}'.")
    except Exception as e:
        st.error(f"Lỗi tách trang PDF: {e}")

def parse_data_detail(file_path: str):
    """
    Parse a plain text file (data_detail.txt) for sections info.
    Format for each line (example): CODE#start_page#end_page#Section Name
    """
    sections = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('#')
                if len(parts) < 4:
                    st.warning(f"Bỏ qua dòng không hợp lệ trong data_detail.txt: {line}")
                    continue
                code = parts[0].strip()
                start_page = int(parts[1].strip())
                end_page = int(parts[2].strip())
                name = parts[3].strip()
                sections.append({
                    'code': code,
                    'start': start_page,
                    'end': end_page,
                    'name': name
                })
    except FileNotFoundError:
        st.error(f"Không tìm thấy tệp '{file_path}'. Vui lòng đảm bảo nó tồn tại.")
    except Exception as e:
        st.error(f"Lỗi đọc tệp '{file_path}': {e}")
    return sections

def get_page_numbers(section):
    """
    Return the list of page numbers for the given section dict.
    """
    return list(range(section['start'], section['end'] + 1))

def pymupdf_parse_page(pdf_path: str, page_number: int = 0) -> str:
    """
    Extract text from a single page using PyMuPDF.
    """
    text = ""
    try:
        with fitz.open(pdf_path) as file:
            if page_number < 0 or page_number >= file.page_count:
                st.error(f"Số trang {page_number + 1} vượt quá phạm vi cho tệp '{pdf_path}'.")
                return ""
            page = file.load_page(page_number)
            text += page.get_text()
    except Exception as e:
        st.error(f"Lỗi mở tệp PDF '{pdf_path}': {e}")
        return ""
    # If you're concerned about super large text, you can limit:
    text = text[:230000]  
    return text

def pymupdf_render_page_as_image(pdf_path: str, page_number: int = 0, zoom: float = 1.5) -> bytes:
    """
    Render a PDF page as image bytes using PyMuPDF.
    """
    try:
        with fitz.open(pdf_path) as doc:
            if page_number < 0 or page_number >= doc.page_count:
                st.error(f"Số trang {page_number + 1} vượt quá phạm vi cho tệp '{pdf_path}'.")
                return b""
            page = doc.load_page(page_number)
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            return img_bytes
    except Exception as e:
        st.error(f"Lỗi chuyển đổi trang PDF '{pdf_path}': {e}")
        return b""
