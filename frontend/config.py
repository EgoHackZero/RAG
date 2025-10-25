# Frontend Configuration
import streamlit as st

# Theme configuration
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styles
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
    }
    .css-1d391kg {
        padding-top: 1rem;
    }
    .stMarkdown {
        font-size: 16px;
    }
    .css-1v0mbdj {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)