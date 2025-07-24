"""
===============================================================================
ESSENTRA - Agentic RAG Chatbot
===============================================================================

Author: Tirumala Manav
Email: tirumalamanav@example.com
GitHub: https://github.com/TirumalaManav
LinkedIn: https://linkedin.com/in/tirumalamanav

Project: ESSENTRA - Advanced Agentic RAG Chatbot
Repository: https://github.com/TirumalaManav/essentra-ai
Created: 2025-07-23
Last Modified: 2025-07-23 17:57:58

License: MIT License
Copyright (c) 2025 Tirumala Manav
""" 


import streamlit as st
import sys
import logging
import asyncio
from pathlib import Path
from datetime import datetime, UTC
import uuid

# Add src directory to sys.path
from pathlib import Path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Imports from your source modules (unchanged)
try:
    from config import setup_application, get_config, is_competition_mode, get_user
    from ui_components import get_api_keys, set_environment_from_secrets, initialize_session_state
    from memory import MemoryFactory, ConversationTurn
    from document_processor import UniversalDocumentProcessor
    from llm_clients import GeminiClient, TavilyWebSearchClient, LLMClientFactory
    from agents import AgentManager
    from langgraph_workflow import LangGraphWorkflow, WorkflowFactory

    print("✅ All ESSENTRA components imported successfully!")
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.stop()

# ============== PAGE & LOGGER CONFIG =============
st.set_page_config(
    page_title="ESSENTRA",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============== SYSTEM INITIALIZATION =============
@st.cache_resource
def initialize_essentra_system():
    try:
        logger.info("🚀 Initializing ESSENTRA...")

        config = setup_application()
        api_keys = get_api_keys()
        set_environment_from_secrets()
        memory = MemoryFactory.create_production_memory({
            "persist_directory": "./data/memory",
            "enable_persistence": True,
            "enable_analytics": True
        })
        doc_processor = UniversalDocumentProcessor(vector_db_path="./data/vector_store")
        llm_factory = LLMClientFactory()
        gemini_client = llm_factory.create_gemini_client()
        tavily_client = llm_factory.create_tavily_client()
        agent_manager = AgentManager()
        agents = agent_manager.initialize_agents()
        agent_manager.inject_dependencies(
            document_processor=doc_processor,
            llm_client=gemini_client,
            web_search_client=tavily_client
        )
        workflow = WorkflowFactory.create_production_workflow(
            document_processor=doc_processor,
            llm_client=gemini_client,
            web_search_client=tavily_client
        )
        return {
            'config': config,
            'memory': memory,
            'doc_processor': doc_processor,
            'gemini_client': gemini_client,
            'tavily_client': tavily_client,
            'agent_manager': agent_manager,
            'workflow': workflow,
            'api_keys': api_keys
        }
    except Exception as e:
        logger.error(f"❌ System initialization failed: {str(e)}")
        return None

# ============== FILE PROCESSING =============
def process_uploaded_file(uploaded_file, system_components):
    try:
        doc_processor = system_components['doc_processor']
        temp_file_path = Path(f"./temp/{uploaded_file.name}")
        temp_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        async def process_doc():
            return await doc_processor.process_document(
                file_path=temp_file_path,
                session_id=st.session_state.session_id,
                chunk_size=500,
                chunk_overlap=50
            )

        with st.spinner(f"🔄 Processing {uploaded_file.name}..."):
            result = asyncio.run(process_doc())

        try:
            temp_file_path.unlink()
        except:
            pass

        if result.success:
            if 'processed_files' not in st.session_state:
                st.session_state.processed_files = []
            file_info = {
                'name': uploaded_file.name,
                'size': len(uploaded_file.getvalue()),
                'chunks': result.chunks_created,
                'processed_at': datetime.now(UTC).isoformat()
            }
            st.session_state.processed_files.append(file_info)
            st.success(f"✅ Processed {uploaded_file.name} ({result.chunks_created} chunks)")
            return True
        else:
            st.error(f"❌ Failed to process {uploaded_file.name}")
            return False
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return False

# ============== MESSAGE PROCESSING =============
def process_user_message(user_input, system_components, uploaded_file=None):
    try:
        workflow = system_components['workflow']

        if uploaded_file:
            process_uploaded_file(uploaded_file, system_components)

        async def process_workflow():
            return await workflow.process_user_request(
                user_query=user_input,
                session_id=st.session_state.session_id,
                uploaded_files=st.session_state.get('processed_files', [])
            )

        with st.spinner("🤖 ESSENTRA is thinking..."):
            result = asyncio.run(process_workflow())

        if result["success"]:
            response_text = result["response"]
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now(UTC).isoformat()
            })
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.now(UTC).isoformat()
            })
            return True
        else:
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now(UTC).isoformat()
            })
            st.session_state.messages.append({
                "role": "assistant",
                "content": result.get("response", "Sorry, I encountered an error."),
                "timestamp": datetime.now(UTC).isoformat()
            })
            return False
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return False

# ============== MAIN APPLICATION =============
def main():
    system_components = initialize_essentra_system()
    if not system_components:
        st.error("❌ Failed to initialize ESSENTRA")
        st.stop()

    initialize_session_state()

    # ---- Minimal, Clean Custom CSS ----
    st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .main { padding-top: 2rem; max-width: 800px; margin: 0 auto;}
    .essentra-title {
        animation: glow 2s ease-in-out infinite alternate;
    }
    @keyframes glow {
        0% { text-shadow: 0 0 20px #00d4aa; }
        50% { text-shadow: 0 0 30px #00d4aa, 0 0 40px #00d4aa; }
        100% { text-shadow: 0 0 20px #00d4aa; }
    }
    .developer-name {
        animation: glow 2s ease-in-out infinite alternate;
        color: #00d4aa; font-weight: 500;
    }
    .status-indicator {
        text-align: center;
        margin: 1rem 0;
        padding: 0.5rem;
        border-radius: 8px;
        background: rgba(0,212,170,0.1);
        border: 1px solid rgba(0,212,170,0.3);
    }
    .welcome-box {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
        border: 1px solid rgba(0, 212, 170, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 2rem auto;
        max-width: 600px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    /* Hide Streamlit's default file uploader display */
    section[data-testid="stFileUploader"] > label, section[data-testid="stFileUploader"] > div:first-child {display: none !important;}
    /* Style plus button uploader under chat */
    .custom-upload-btn {
        font-size: 28px;
        background: #00d4aa;
        color: #fff;
        border: none;
        border-radius: 50%;
        width: 44px;
        height: 44px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 12px auto 0 auto;
        box-shadow: 0 2px 8px rgba(0,0,0,0.10);
        transition: background 0.3s;
    }
    .custom-upload-btn:hover {
        background: #11ffa2; color: #222;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---- HEADER ----
    st.markdown("""
    <div style='text-align: center; margin-bottom: 3rem;'>
        <h1 class='essentra-title' style='font-size: 3.5rem; margin: 0; color: #00d4aa; font-weight: 700;'>
            🤖 ESSENTRA
        </h1>
        <p style='font-size: 1.1rem; color: #888; margin: 0.5rem 0; font-weight: 300;'>
            Your Intelligent Document Assistant
        </p>
        <p class='developer-name' style='font-size: 0.9rem; margin: 0;'>
            <strong>TirumalaManav</strong> - Advanced RAG System with LangGraph
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ---- STATUS ----
    st.markdown("""
    <div class='status-indicator'>
        🚀 <strong>ESSENTRA System is fully operational with Advanced RAG!</strong>
    </div>
    """, unsafe_allow_html=True)

    # ---- WELCOME ----
    if not st.session_state.messages:
        st.markdown("""
        <div class='welcome-box'>
            <div style='display: flex; align-items: center; margin-bottom: 1rem;'>
                <div style='background: #00d4aa; width: 32px; height: 32px; border-radius: 50%;
                            display: flex; align-items: center; justify-content: center; margin-right: 12px;'>
                    🤖
                </div>
                <strong style='color: #fff; font-size: 1.1rem;'>Hi! I'm ESSENTRA, your intelligent document assistant.</strong>
            </div>
            <p style='margin: 0; color: #ccc; line-height: 1.5;'>
                I can help you analyze documents, answer questions from your uploaded files, and provide intelligent insights.
                I'm equipped with advanced RAG capabilities and LangGraph workflow orchestration.
            </p>
            <p style='margin: 1rem 0 0 0; color: #00d4aa; font-weight: 500;'>
                What can I help you with today?
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ---- CHAT MESSAGES ----
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ---- CHAT INPUT & CUSTOM "PLUS" UPLOAD BUTTON ----
    # Use empty columns to mock a visual box -- you can adjust or group them further!
    col_main = st.container()
    with col_main:
        user_input = st.chat_input("Message ESSENTRA...")

        # Invisible file uploader, set to no label/layout
        uploaded_file = st.file_uploader(
            "",
            type=['pdf', 'docx', 'txt', 'md', 'csv', 'xlsx'],
            label_visibility="collapsed",
            key="hidden_uploader"
        )

        # Custom '+' icon button below the chat input, inside same visual container
        st.markdown("""
            <button class="custom-upload-btn" onclick="document.querySelector('input[type=file]').click();">+</button>
            """, unsafe_allow_html=True)

    # ---- FILE UPLOAD HANDLING ----
    if uploaded_file and uploaded_file not in st.session_state.get('processed_files_names', []):
        if 'processed_files_names' not in st.session_state:
            st.session_state.processed_files_names = []
        st.session_state.processed_files_names.append(uploaded_file)
        success = process_uploaded_file(uploaded_file, system_components)
        if success:
            st.rerun()

    if user_input:
        success = process_user_message(user_input, system_components)
        st.rerun()

    # ---- NEW CHAT BUTTON ----
    if st.session_state.messages:
        if st.button("🆕 New Chat", key="new_chat_btn"):
            st.session_state.messages = []
            st.session_state.processed_files = []
            st.session_state.processed_files_names = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()

    st.markdown("""
    <div style='text-align: center; margin-top: 4rem; padding: 2rem; color: #666; font-size: 0.8rem;'>
        <span class='developer-name'>🤖 ESSENTRA v1.0 • Built with ❤️ by TirumalaManav</span>
    </div>
    """, unsafe_allow_html=True)

# ============== RUN APPLICATION ==============
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"💥 Critical Error: {str(e)}")
        if st.button("🔄 Restart"):
            st.cache_resource.clear()
            st.rerun()
