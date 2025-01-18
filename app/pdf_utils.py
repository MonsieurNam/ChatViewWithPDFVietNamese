# app/pdf_utils.py

import os
from io import BytesIO
import fitz  # PyMuPDF
from PyPDF2 import PdfReader, PdfWriter
import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

@st.cache_data
def get_pdf_text(pdf_path):
    text = ""
    try:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        st.error(f"Lỗi trích xuất văn bản từ '{pdf_path}': {e}")
    return text

@st.cache_data
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=5000,
        chunk_overlap=500,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_documents(chunks):
    return [Document(page_content=chunk) for chunk in chunks]

def get_bm25_retriever(docs):
    from langchain_community.retrievers import BM25Retriever
    return BM25Retriever.from_documents(docs)

def split_pdf_into_pages(pdf_path, output_dir):
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

def pymupdf_parse_page(pdf_path: str, page_number: int = 0) -> str:
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
    return text[:230000]

def pymupdf_render_page_as_image(pdf_path: str, page_number: int = 0, zoom: float = 1.5) -> bytes:
    try:
        with fitz.open(pdf_path) as doc:
            if page_number < 0 or page_number >= doc.page_count:
                st.error(f"Số trang {page_number + 1} vượt quá phạm vi cho tệp '{pdf_path}'.")
                return b""
            page = doc.load_page(page_number)
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            return pix.tobytes("png")
    except Exception as e:
        st.error(f"Lỗi chuyển đổi trang PDF '{pdf_path}': {e}")
        return b""
