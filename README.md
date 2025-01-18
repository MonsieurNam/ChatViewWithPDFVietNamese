# 📚 PDF Chat & View Assistant (Optimized for Vietnamese)

## Welcome to the PDF Chat & View Assistant!
This application allows you to:

- Upload PDF files for easy viewing (page by page).
- Chat with your PDFs in Vietnamese via an AI powered by LangChain, Groq's LLM, and BM25 Retrieval.

Test app : [https://chatviewwithpdfvietnamese.streamlit.app/](https://chatviewwithpdfvietnamese.streamlit.app/)


<p align="center">
  <img src="https://github.com/MonsieurNam/ChatViewWithPDFVietNamese/blob/main/image/Screenshot%202025-01-18%20220154.png" alt="Centered image" width="1200" height="400" />
</p>

<p align="center">
  <img src="https://github.com/MonsieurNam/ChatViewWithPDFVietNamese/blob/main/image/Screenshot%202025-01-18%20220114.png" alt="Centered image" width="1200" height="400" />
</p>


---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Streamlit
- Groq API Token
- (Optional but recommended) A virtual environment via venv, conda, etc.

---

### Installation

#### Clone the Repository
```bash
git clone https://github.com/MonsieurNam/ChatViewWithPDFVietNamese.git
```
Or download the repo as a .zip and extract.

#### Install Required Packages
```bash
cd ChatViewWithPDFVietNamese
pip install -r requirements.txt
```

#### Set up Environment Variables
Create a `.env` file in the app/ folder and add your Groq API Token:
```env
GROQ_API_TOKEN="your_groq_api_key_here"
```
Alternatively, you can set `GROQ_API_TOKEN` directly in your environment.


## 🎯 How to Use

### Run the Application
From the project root directory:
```bash
streamlit run app/app.py
```

### Upload PDFs
On the Streamlit sidebar, you will see three upload prompts:

- `data.pdf` – The original PDF with images and text.
- `data_content.pdf` – A PDF containing only the text for the chatbot.
- `data_detail.txt` – A text file describing PDF structure in lines of the form:
  ```plaintext
  <section_code>#<start_page>#<end_page>#<section_name>
  ```
  For example:
  ```plaintext
  a#1#9#NHỮNG QUỐC GIA ĐẦU TIÊN TRÊN LÃNH THỔ VIỆT NAM
  b#10#82#XÂY DỰNG VÀ BẢO VỆ ĐẤT NƯỚC VIỆT NAM
  b1#10#16#Đấu tranh giành độc lập thời kì Bắc thuộc
  ```

The app automatically splits `data.pdf` into page-level PDFs, which are displayed on the main page.

### Ask Questions
Once the documents are processed, a chat interface appears in the sidebar.
Type your questions in Vietnamese (or your desired language), and the assistant will reply based on your PDF’s content.

### View & Download
- **View**: Each page is displayed as an image in the main section.
- **Toggle**: "Hiển Thị Văn Bản Đã Trích Xuất" reveals the raw text extracted from each page.
- **Download**: Use "Tải Xuống Trang PDF" or "Tải Xuống Văn Bản" to download individual pages or their extracted text.

---

## 🌟 Customization Options

### System Prompt (in `groq_wrapper.py`):
Adjust the `system_prompt` property to change the assistant’s overall style or constraints.

### Temperature:
Change temperature in `_call()` for more or less creative responses (lower = more deterministic).

---

## 💡 Key Components

- **Streamlit**: Front-end interface and container for the entire workflow.
- **BM25Retriever**: A classical text retrieval method to quickly find relevant sections in PDF text.
- **LangChain**: Manages the conversational chain and memory.
- **Groq API**: Provides the LLM for natural language responses.

---

## ⚙️ Workflow

1. **Uploads**: PDFs + TXT structure are uploaded.
2. **Splitting & Parsing**:
   - `data.pdf` is split into individual pages (`pdf_processing.split_pdf_into_pages`).
   - `data_content.pdf` is extracted into text chunks (`get_pdf_text`, `get_text_chunks`).
3. **BM25 Retrieval**:
   - Text chunks are indexed with BM25 to find relevant sections on user queries.
4. **Conversation**:
   - `ConversationalRetrievalChain` (LangChain) uses `BM25Retriever` + the custom `Groq LLM` to answer user questions.
5. **Display**:
   - Pages rendered as images (`pymupdf_render_page_as_image`) and shown on the main page.
   - Chat interface in the sidebar with a conversation history.

---

## 🔮 Ongoing Development

- **Automatic Content Extraction**: Automate extraction of text & structure, reducing manual steps.
- **Professional File Splitting**: Support advanced PDF splitting strategies for complex layouts.
- **Multi-Language Optimization**: Enhance performance in Vietnamese and beyond.
- **Real-Time Collaboration**: Multi-user editing and chat.

---

## Happy Chatting & Viewing!
If you find this tool helpful, feel free to ⭐ star the repo or contribute via pull requests.

For any issues or questions, open an issue on GitHub.
