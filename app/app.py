import os
import streamlit as st
from io import BytesIO

# Local imports
from st_config import load_env_vars  # If you separate config
from pdf_processing import (
    get_pdf_text,
    get_text_chunks,
    create_documents,
    split_pdf_into_pages,
    parse_data_detail,
    get_page_numbers,
    pymupdf_parse_page,
    pymupdf_render_page_as_image
)
from chain import (
    get_bm25_retriever,
    get_conversation_chain,
    handle_userinput,
    clear_chat_history
)
from groq_wrapper import GroqWrapper
from htmlTemplates import css, user_template, bot_template, loading_template

def initialize_session_state(total_pages=0):
    if 'total_pages' not in st.session_state:
        st.session_state.total_pages = total_pages

def main():
    # Optionally load .env variables
    load_env_vars()
    
    st.set_page_config(page_title="CHAT&VIEWWITHPDF", layout="wide")
    st.write(css, unsafe_allow_html=True)

    # Initialize session_state variables
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Title
    st.title("CHAT&VIEWWITHPDF")

    # Sidebar: Upload Files
    st.sidebar.header("📥 Upload Tệp Cần Thiết")
    uploaded_data_pdf = st.sidebar.file_uploader("Upload `data.pdf` (pdf với hình và text)", type=["pdf"], key="data_pdf")
    uploaded_data_content_pdf = st.sidebar.file_uploader("Upload `data_content.pdf` (pdf chỉ chứa text)", type=["pdf"], key="data_content_pdf")
    uploaded_data_detail = st.sidebar.file_uploader("Upload `data_detail.txt`", type=["txt"], key="data_detail_txt")

    # Check if all 3 files are uploaded
    if uploaded_data_pdf and uploaded_data_content_pdf and uploaded_data_detail:
        # Create directories if they don't exist
        os.makedirs("uploaded_data/pdf_pages", exist_ok=True)

        # Save uploaded files
        with open("uploaded_data/data.pdf", "wb") as f:
            f.write(uploaded_data_pdf.getbuffer())

        with open("uploaded_data/data_content.pdf", "wb") as f:
            f.write(uploaded_data_content_pdf.getbuffer())

        with open("uploaded_data/data_detail.txt", "wb") as f:
            f.write(uploaded_data_detail.getbuffer())

        st.sidebar.success("Tất cả các tệp đã được upload thành công!")
        
        # Split data.pdf into individual pages
        split_pdf_into_pages("uploaded_data/data.pdf", "uploaded_data/pdf_pages")

        # Parse data_detail.txt
        sections = parse_data_detail("uploaded_data/data_detail.txt")

        if not sections:
            st.error("Không tìm thấy phần hợp lệ. Vui lòng kiểm tra tệp data_detail.txt của bạn.")
            return

        # Extract text from data_content.pdf for chatbot
        raw_text = get_pdf_text("uploaded_data/data_content.pdf")
        text_chunks = get_text_chunks(raw_text)
        docs = create_documents(text_chunks)
        retriever = get_bm25_retriever(docs)
        st.session_state.retriever = retriever

        # Initialize conversation chain
        if st.session_state.conversation is None:
            st.session_state.conversation = get_conversation_chain(st.session_state.retriever)

        # Sidebar: Chọn Phần Chính
        st.sidebar.header("Chọn Phần Chính")
        main_sections = [
            section for section in sections 
            if len(section['code']) == 1 and 
               (section['name'].startswith("NHỮNG QUỐC GIA") or 
                section['name'].startswith("XÂY DỰNG"))
        ]
        selected_main_section = st.sidebar.selectbox(
            "Chọn một phần chính:",
            [section['name'] for section in main_sections]
        )

        # Find the selected main section details
        selected_main_section_details = next(
            (s for s in sections if s['name'] == selected_main_section),
            None
        )

        # Handle sub-section selection if main section is "XÂY DỰNG..."
        if selected_main_section and selected_main_section.startswith("XÂY DỰNG"):
            st.sidebar.header("Chọn Phần Con")
            sub_sections = [
                section for section in sections
                if (section['start'] >= selected_main_section_details['start'] and
                    section['end'] <= selected_main_section_details['end'] and
                    len(section['code']) > 1)
            ]
            selected_sub_section_name = st.sidebar.selectbox(
                "Chọn một phần con:",
                [section['name'] for section in sub_sections]
            )

            # Find details of the selected sub-section
            selected_sub_section = next(
                (s for s in sections if s['name'] == selected_sub_section_name),
                None
            )

            if not selected_sub_section:
                st.error("Không tìm thấy phần đã chọn.")
                return

            # Get page numbers for the selected sub-section
            page_numbers = get_page_numbers(selected_sub_section)
        else:
            # If the main section is not "XÂY DỰNG", use the main section itself
            selected_sub_section = selected_main_section_details
            page_numbers = get_page_numbers(selected_main_section_details)

        total_pages = len(page_numbers)
        initialize_session_state(total_pages=total_pages)

        # Sidebar: Display Options
        st.sidebar.header("Tùy Chọn Hiển Thị")
        show_text = st.sidebar.checkbox("Hiển Thị Văn Bản Đã Trích Xuất", value=True)
        zoom_factor = st.sidebar.slider("Mức Thu Phóng", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

        # Main Content: Display PDF Pages
        st.header(f"📄 {selected_sub_section['name'] if selected_sub_section else selected_main_section}")

        data_dir = "uploaded_data/pdf_pages"
        for page_num in page_numbers:
            pdf_filename = f"page_{page_num:03}.pdf"
            pdf_path = os.path.join(data_dir, pdf_filename)

            if not os.path.isfile(pdf_path):
                st.error(f"Không tìm thấy tệp PDF '{pdf_filename}' trong '{data_dir}'.")
                continue

            # Show PDF page as image
            try:
                img_bytes = pymupdf_render_page_as_image(pdf_path, page_number=0, zoom=zoom_factor)
                if img_bytes:
                    st.image(img_bytes, caption=f"Trang {page_num}", use_column_width=True)
                else:
                    st.error("Không thể hiển thị hình ảnh trang.")
            except Exception as e:
                st.error(f"Lỗi hiển thị trang PDF: {e}")

            st.markdown("---")

        # Sidebar: Page Information
        st.sidebar.header("Thông Tin Trang")
        for page_num in page_numbers:
            with st.sidebar.expander(f"📄 Trang {page_num}"):
                pdf_filename = f"page_{page_num:03}.pdf"
                pdf_path = os.path.join(data_dir, pdf_filename)

                if not os.path.isfile(pdf_path):
                    st.error(f"Không tìm thấy tệp PDF '{pdf_filename}' trong '{data_dir}'.")
                    continue

                # Extract page text
                page_text = pymupdf_parse_page(pdf_path)

                # Show text if needed
                if show_text:
                    st.text_area("Văn Bản Đã Trích Xuất:", value=page_text, height=150)

                download_col1, download_col2 = st.columns(2)
                with download_col1:
                    # Download the page PDF
                    try:
                        with open(pdf_path, 'rb') as f:
                            pdf_bytes = f.read()
                        st.download_button(
                            label="📥 Tải Xuống Trang PDF",
                            data=pdf_bytes,
                            file_name=pdf_filename,
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"Lỗi chuẩn bị tải xuống PDF: {e}")

                if show_text:
                    with download_col2:
                        # Download extracted text
                        try:
                            buffer = BytesIO()
                            buffer.write(page_text.encode('utf-8'))
                            buffer.seek(0)
                            safe_section_name = "".join(
                                c for c in selected_sub_section['name'] if c.isalnum() or c in (' ', '_', '-')
                            ).rstrip()
                            download_filename = f"{safe_section_name}_Trang_{page_num}.txt"
                            st.download_button(
                                label="📄 Tải Xuống Văn Bản",
                                data=buffer,
                                file_name=download_filename,
                                mime="text/plain"
                            )
                        except Exception as e:
                            st.error(f"Lỗi chuẩn bị tải xuống văn bản: {e}")

        # Sidebar: Chat Functionality
        st.sidebar.header("💬 Chat với Tài liệu")
        user_question = st.sidebar.text_input("Đặt câu hỏi về tài liệu của bạn:", key="user_input")

        if user_question:
            handle_userinput(user_question)

        # Display chat messages
        st.sidebar.markdown("---")
        if st.session_state.messages:
            st.sidebar.header("📜 Lịch Sử Trò Chuyện")
            # Display the messages in reverse chronological order
            for message in st.session_state.messages[::-1]:
                if message["role"] == "assistant":
                    st.sidebar.markdown(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
                else:
                    st.sidebar.markdown(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)

        if st.sidebar.button("🧹 Xóa lịch sử trò chuyện"):
            clear_chat_history()
            st.experimental_rerun()

    else:
        st.sidebar.warning("Vui lòng upload đầy đủ 3 tệp: `data.pdf`, `data_content.pdf`, và `data_detail.txt`.")


if __name__ == "__main__":
    main()
