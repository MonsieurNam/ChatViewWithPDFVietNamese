import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.retrievers import BM25Retriever
from langdetect import detect

from groq_wrapper import GroqWrapper

def get_bm25_retriever(docs):
    """
    Create a BM25Retriever from a list of Document objects.
    """
    retriever = BM25Retriever.from_documents(docs)
    return retriever

def get_conversation_chain(retriever):
    """
    Create a ConversationalRetrievalChain using a custom LLM (GroqWrapper).
    """
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    llm = GroqWrapper()  # Make sure your GROQ_API_TOKEN is set
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    """
    Process the user input within the conversation chain,
    handle language detection, and store messages in session.
    """
    modified_question = user_question

    # Display loading message
    with st.spinner('Đang xử lý...'):
        # Save a placeholder for a loading animation in the sidebar (optional)
        response = st.session_state.conversation({'question': modified_question})
        st.session_state.chat_history = response['chat_history']

    ai_response = st.session_state.chat_history[-1].content

    # Check if the AI responded in Vietnamese
    try:
        language = detect(ai_response)
    except:
        language = 'unknown'

    if language != 'vi':
        st.warning("AI đã trả lời bằng ngôn ngữ khác. Đang yêu cầu AI trả lời lại bằng tiếng Việt...")
        modified_question = user_question + " Vui lòng trả lời bằng tiếng Việt."
        response = st.session_state.conversation({'question': modified_question})
        st.session_state.chat_history = response['chat_history']
        ai_response = st.session_state.chat_history[-1].content

    # Update message history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    st.session_state.messages.append({"role": "user", "content": user_question})

def clear_chat_history():
    """
    Clear the conversation/chat history from Streamlit session_state.
    """
    st.session_state.messages = []
    st.session_state.chat_history = []
