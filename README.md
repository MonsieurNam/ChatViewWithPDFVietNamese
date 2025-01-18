# 📚 PDF Chat&View Assistant optimize for Vietnamese !

Welcome to the **PDF Chat&View Assistant**! This application allows you to upload PDF files for viewing and chat with them using the power of AI. Powered by **LangChain**, and **Groq's LLM**, it brings your static documents to life by giving you answers to your questions based on their content!

Test app : https://chatviewwithpdfvietnamese.streamlit.app/


<p align="center">
  <img src="https://github.com/MonsieurNam/ChatViewWithPDFVietNamese/blob/main/image/Screenshot%202025-01-18%20220154.png" alt="Centered image" width="1200" height="400" />
</p>

<p align="center">
  <img src="https://github.com/MonsieurNam/ChatViewWithPDFVietNamese/blob/main/image/Screenshot%202025-01-18%20220114.png" alt="Centered image" width="1200" height="400" />
</p>

## 🚀 Getting Started

### Prerequisites

To get started with the **PDF Chat&View Assistant**, you will need to have:

- **Python 3.8+**
- **Streamlit**
- **Hugging Face Transformers**
- **Groq API Token**

### Installation

1. **Clone the Repository**

   ```bash
   https://github.com/MonsieurNam/ChatViewWithPDFVietNamese.git
   ```

2. **Install the Required Packages**

   Install all dependencies by running:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables**

   Create a `.env` file in the root directory and add your **Groq API Token**:

   ```plaintext
   GROQ_API_TOKEN="your_groq_api_key_here"
   ```
### 🎯 How to Use

1. **Run the Application**

   Start the application using the following command:

   ```bash
   streamlit run app.py 
   ```

2. **Upload PDFs**

   Use the document uploader on the right side of the page to upload your PDF files.
   Attention to pre-prepare 3 file:
   - Your pdf file.pdf
   - Your text content of pdf file.pdf
   - The structure of pdf file.txt: <mã phần>#<trang bắt đầu>#<trang kết thúc>#<tên phần>
   + EX:
   + a#001#009#NHỮNG QUỐC GIA ĐẦU TIÊN TRÊN LÃNH THỔ VIỆT NAM
   + b#010#082#XÂY DỰNG VÀ BẢO VỆ ĐẤT NƯỚC VIỆT NAM
   + b1#010#016#Đấu tranh giành độc lập thời kì Bắc thuộc
   .......


3. **Ask Questions**

   Once your documents are processed, you can start asking questions based on the content of your PDFs in the chat interface at sidebar.

4. **Customize the AI**

   Modify the system prompt to change how the AI responds. For example, you can ask it to be more casual, technical, or creative!


## 🛠️ Project Architecture

### Key Components:

- **Streamlit**: Frontend interface and chat system.
- **LangChain**: Manages conversational logic and connects your queries with document content.
- **Hugging Face Embeddings**: Transforms document text into meaningful vector representations.
- **Groq API**: The powerhouse behind generating intelligent responses based on document data.

### Workflow

1. **Upload PDFs**: The user uploads PDFs that need to be processed.
2. **Text Processing**: Text is extracted and split into chunks for easier embedding.
3. **Viewing and Conversational Retrieval**: View the PDF file and using the AI searches through the document to retrieve relevant information and answers questions.

## 🌟 Customization Options

- **System Prompt**: Customize the behavior of the AI by changing the system prompt.
- **Temperature**: Adjust the temperature to control the creativity of responses.

## Technologies Used:
- **Python**: Core programming language.
- **Streamlit**: For the web interface.
- **PyPDF2**: For extracting text from PDFs.
- **LangChain**: For conversational AI and document vectorization.
- **Groq API**: Powers the conversational responses.
  
## 🔮 Customization Option
- **Automatic Content Extraction**: Automate the extraction of text and structure from uploaded PDFs, eliminating the need for manual preparation.
- **Professional File Splitting**: Implement more advanced file splitting techniques to handle complex PDFs with multi-level hierarchies or irregular layouts.
- **Multi-Language Optimization**: Enhance the AI’s performance in Vietnamese and other Southeast Asian languages for more accurate and nuanced responses.
- **Real-Time Collaboration**: Enable multiple users to work with the same document simultaneously, with synchronized chat and edits.
- **Improved Search Features**: Add advanced search capabilities, such as fuzzy matching and contextual highlighting, to refine document navigation.

