import os
import streamlit as st
from dotenv import load_dotenv

def load_env_vars():
    """
    Load environment variables from .env file.
    """
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_TOKEN")
    if not groq_api_key:
        st.error("GROQ_API_TOKEN is not set in .env or environment variables.")
    return groq_api_key
