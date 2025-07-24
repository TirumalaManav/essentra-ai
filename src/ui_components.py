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
import os
import time
import uuid
from datetime import datetime, UTC
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import base64
import logging

# Streamlit imports
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== API KEY MANAGEMENT (DUAL MODE) ====================

def get_api_keys() -> Dict[str, str]:
    """Get API keys from either Streamlit secrets or environment variables"""
    api_keys = {}

    try:
        # Try Streamlit secrets first (for cloud deployment)
        if hasattr(st, 'secrets'):
            api_keys = {
                'GEMINI_API_KEY': st.secrets.get('GEMINI_API_KEY', ''),
                'TAVILY_API_KEY': st.secrets.get('TAVILY_API_KEY', ''),
                'USER_NAME': st.secrets.get('USER_NAME', 'TIRUMALAMANAV')
            }
            logger.info("🔑 Using Streamlit Cloud secrets")
        else:
            # Fallback to environment variables (for local development)
            api_keys = {
                'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY', ''),
                'TAVILY_API_KEY': os.getenv('TAVILY_API_KEY', ''),
                'USER_NAME': os.getenv('USER_NAME', 'TIRUMALAMANAV')
            }
            logger.info("🔑 Using environment variables")

    except Exception as e:
        # Final fallback to environment variables
        api_keys = {
            'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY', ''),
            'TAVILY_API_KEY': os.getenv('TAVILY_API_KEY', ''),
            'USER_NAME': os.getenv('USER_NAME', 'TIRUMALAMANAV')
        }
        logger.warning(f"⚠️ Secrets fallback: {str(e)}")

    # Validate keys
    missing_keys = [k for k, v in api_keys.items() if not v and k != 'USER_NAME']
    if missing_keys:
        logger.error(f"❌ Missing API keys: {missing_keys}")
        st.error(f"Missing API keys: {missing_keys}. Please configure in Streamlit Cloud secrets or .env file.")

    return api_keys

def set_environment_from_secrets():
    """Set environment variables from Streamlit secrets for other modules"""
    api_keys = get_api_keys()

    for key, value in api_keys.items():
        if value:
            os.environ[key] = value

    logger.info("🔧 Environment variables synchronized")

# ==================== SESSION STATE MANAGEMENT ====================

def initialize_session_state():
    """Initialize Streamlit session state"""
    defaults = {
        'messages': [],
        'session_id': str(uuid.uuid4()),
        'uploaded_files': [],
        'processing': False,
        'user_name': get_api_keys().get('USER_NAME', 'TIRUMALAMANAV'),
        'chat_history': [],
        'current_file_content': None,
        'last_response_time': 0.0,
        'total_queries': 0
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    logger.info(f"📱 Session initialized: {st.session_state.session_id[:8]}...")

# ==================== STYLING FUNCTIONS ====================

def load_custom_css():
    """Load custom CSS for ESSENTRA styling"""
    css = """
    <style>
    /* Main App Styling */
    .main {
        padding: 0rem 1rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom chat container */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 1rem;
    }

    /* Message styling */
    .user-message {
        background: #f0f0f0;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 18px;
        border-top-right-radius: 5px;
    }

    .assistant-message {
        background: #ffffff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 18px;
        border-top-left-radius: 5px;
        border: 1px solid #e0e0e0;
    }

    /* File upload styling */
    .uploadedFile {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 1rem 0;
    }

    /* Title styling */
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 300;
        margin: 2rem 0;
        color: #333;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 3rem;
    }

    /* Processing indicator */
    .processing {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #666;
        font-style: italic;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #fafafa;
    }

    /* Chat input */
    .stChatInput > div > div > textarea {
        border-radius: 25px;
        border: 1px solid #ddd;
        padding: 1rem 1.5rem;
    }

    /* Attachment button */
    .attachment-btn {
        background: none;
        border: none;
        font-size: 1.2rem;
        cursor: pointer;
        padding: 0.5rem;
        border-radius: 50%;
    }

    .attachment-btn:hover {
        background-color: #f0f0f0;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ==================== HEADER COMPONENT ====================

def render_header():
    """Render simple, clean header"""
    st.markdown(
        """
        <div class="main-title">
            🤖 ESSENTRA
        </div>
        <div class="subtitle">
            Your Intelligent Document Assistant
        </div>
        """,
        unsafe_allow_html=True
    )

# ==================== SIDEBAR COMPONENT ====================

def render_sidebar():
    """Render minimal sidebar with chat history"""
    with st.sidebar:
        # New Chat Button
        if st.button("🆕 New chat", use_container_width=True):
            # Clear current conversation
            st.session_state.messages = []
            st.session_state.uploaded_files = []
            st.session_state.current_file_content = None
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()

        st.markdown("---")

        # Chat History
        st.subheader("📝 Recent")

        # Display recent chat sessions
        if st.session_state.chat_history:
            for i, chat in enumerate(st.session_state.chat_history[-5:]):  # Last 5 chats
                chat_preview = chat.get('title', 'Untitled Chat')[:30] + "..."
                if st.button(chat_preview, key=f"chat_{i}"):
                    # Load selected chat (implement if needed)
                    st.info("Chat history loading - feature coming soon!")
        else:
            st.write("No previous chats")

        st.markdown("---")

        # User info (minimal)
        st.markdown(f"👤 **{st.session_state.user_name}**")

        # Simple stats (optional, minimal)
        if st.session_state.total_queries > 0:
            st.markdown(f"💬 Queries: {st.session_state.total_queries}")
            if st.session_state.last_response_time > 0:
                st.markdown(f"⚡ Last: {st.session_state.last_response_time:.1f}s")

# ==================== FILE UPLOAD COMPONENT ====================

def render_file_upload() -> Optional[UploadedFile]:
    """Render simple file upload"""
    uploaded_file = st.file_uploader(
        "📎 Attach a document",
        type=['pdf', 'docx', 'txt', 'md'],
        help="Upload PDF, DOCX, TXT, or MD files",
        label_visibility="collapsed"
    )

    if uploaded_file:
        # Show file info
        file_size = len(uploaded_file.getvalue()) / 1024  # KB
        st.success(f"📄 **{uploaded_file.name}** ({file_size:.1f} KB)")

        # Add to session state if new
        if uploaded_file not in st.session_state.uploaded_files:
            st.session_state.uploaded_files.append(uploaded_file)
            logger.info(f"📎 File uploaded: {uploaded_file.name}")

    return uploaded_file

# ==================== MESSAGE DISPLAY COMPONENT ====================

def render_message(role: str, content: str, metadata: Optional[Dict] = None):
    """Render individual chat message"""

    with st.chat_message(role):
        if role == "user":
            st.markdown(content)

        elif role == "assistant":
            # Main response
            st.markdown(content)

            # Optional metadata (minimal display)
            if metadata:
                with st.expander("ℹ️ Details", expanded=False):
                    if 'sources_used' in metadata and metadata['sources_used']:
                        st.markdown("**📚 Sources:**")
                        for source in metadata['sources_used'][:3]:  # Max 3 sources
                            st.markdown(f"• {source}")

                    if 'processing_time' in metadata:
                        st.markdown(f"⏱️ Response time: {metadata['processing_time']:.1f}s")

                    if 'confidence_score' in metadata:
                        confidence = metadata['confidence_score']
                        st.markdown(f"🎯 Confidence: {confidence:.1%}")

# ==================== CHAT DISPLAY COMPONENT ====================

def render_chat_history():
    """Render complete chat history"""
    if not st.session_state.messages:
        # Empty state
        st.markdown(
            """
            <div class="main-title" style="margin-top: 4rem;">
                What can I help with?
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    # Display all messages
    for message in st.session_state.messages:
        render_message(
            role=message["role"],
            content=message["content"],
            metadata=message.get("metadata")
        )

# ==================== PROCESSING INDICATOR ====================

def show_processing_indicator():
    """Show processing indicator"""
    with st.chat_message("assistant"):
        with st.spinner("🤖 ESSENTRA is thinking..."):
            time.sleep(0.1)  # Brief pause for visual effect

# ==================== INPUT PROCESSING ====================

def process_user_input(user_input: str, uploaded_file: Optional[UploadedFile] = None) -> Dict[str, Any]:
    """Process user input and return response data"""

    # This function will be called by the main app
    # It should integrate with your RAG system

    start_time = time.time()

    try:
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now(UTC).isoformat()
        })

        # Process file if uploaded
        file_info = None
        if uploaded_file:
            file_info = {
                "name": uploaded_file.name,
                "type": uploaded_file.type,
                "size": len(uploaded_file.getvalue())
            }

        # Placeholder response (replace with your RAG system)
        response_data = {
            "response": f"Thank you for your question: '{user_input}'. " +
                       (f"I've analyzed your uploaded file '{uploaded_file.name}'. " if uploaded_file else "") +
                       "This is a placeholder response. Your RAG system will provide the actual intelligent response here.",
            "sources_used": ["document.pdf", "web_search"] if uploaded_file else ["web_search"],
            "confidence_score": 0.85,
            "processing_time": time.time() - start_time,
            "model_used": "gemini-1.5-flash"
        }

        # Add assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_data["response"],
            "metadata": {
                "sources_used": response_data["sources_used"],
                "processing_time": response_data["processing_time"],
                "confidence_score": response_data["confidence_score"]
            },
            "timestamp": datetime.now(UTC).isoformat()
        })

        # Update session stats
        st.session_state.total_queries += 1
        st.session_state.last_response_time = response_data["processing_time"]

        logger.info(f"💬 Query processed in {response_data['processing_time']:.2f}s")

        return response_data

    except Exception as e:
        logger.error(f"❌ Error processing input: {str(e)}")

        # Add error message
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"I apologize, but I encountered an error while processing your request: {str(e)}",
            "metadata": {"error": True},
            "timestamp": datetime.now(UTC).isoformat()
        })

        return {"error": str(e)}

# ==================== MAIN UI ORCHESTRATOR ====================

def render_main_ui():
    """Render the complete ESSENTRA UI"""

    # Initialize session state
    initialize_session_state()

    # Set environment variables from secrets
    set_environment_from_secrets()

    # Load custom CSS
    load_custom_css()

    # Render sidebar
    render_sidebar()

    # Main content area
    with st.container():
        # File upload (when no conversation started)
        uploaded_file = None
        if not st.session_state.messages:
            uploaded_file = render_file_upload()

        # Chat history
        render_chat_history()

        # File upload in conversation (smaller, inline)
        if st.session_state.messages:
            uploaded_file = render_file_upload()

    # Chat input
    if user_input := st.chat_input("Message ESSENTRA"):
        # Show processing indicator
        with st.spinner("🤖 Processing..."):
            # Process the input
            process_user_input(user_input, uploaded_file)

        # Rerun to update the display
        st.rerun()

# ==================== UTILITY FUNCTIONS ====================

def export_chat_history() -> str:
    """Export chat history as markdown"""
    if not st.session_state.messages:
        return "No conversation to export."

    export_lines = [
        f"# ESSENTRA Chat Export",
        f"**User:** {st.session_state.user_name}",
        f"**Session:** {st.session_state.session_id}",
        f"**Exported:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC",
        "",
        "---",
        ""
    ]

    for i, message in enumerate(st.session_state.messages):
        role = "**User**" if message["role"] == "user" else "**ESSENTRA**"
        export_lines.append(f"## {role}")
        export_lines.append(message["content"])
        export_lines.append("")

    return "\n".join(export_lines)

def clear_chat_history():
    """Clear current chat history"""
    st.session_state.messages = []
    st.session_state.uploaded_files = []
    st.session_state.current_file_content = None
    logger.info("🧹 Chat history cleared")

def get_session_stats() -> Dict[str, Any]:
    """Get current session statistics"""
    return {
        "session_id": st.session_state.session_id,
        "total_messages": len(st.session_state.messages),
        "total_queries": st.session_state.total_queries,
        "uploaded_files": len(st.session_state.uploaded_files),
        "last_response_time": st.session_state.last_response_time,
        "user_name": st.session_state.user_name
    }

# ==================== ERROR HANDLING ====================

def handle_api_error(error: Exception) -> str:
    """Handle API errors gracefully"""
    error_msg = str(error)

    if "api key" in error_msg.lower():
        return "⚠️ API configuration issue. Please check your API keys in Streamlit Cloud secrets."
    elif "timeout" in error_msg.lower():
        return "⏱️ Request timeout. Please try again with a shorter query."
    elif "rate limit" in error_msg.lower():
        return "🚦 Rate limit reached. Please wait a moment before trying again."
    else:
        return f"❌ An error occurred: {error_msg}"

# ==================== TESTING UTILITIES ====================

def create_sample_conversation():
    """Create sample conversation for testing"""
    sample_messages = [
        {
            "role": "user",
            "content": "What is artificial intelligence?",
            "timestamp": datetime.now(UTC).isoformat()
        },
        {
            "role": "assistant",
            "content": "Artificial Intelligence (AI) is a field of computer science focused on creating systems that can perform tasks typically requiring human intelligence, such as learning, reasoning, problem-solving, perception, and language understanding.",
            "metadata": {
                "sources_used": ["ai_textbook.pdf"],
                "processing_time": 1.2,
                "confidence_score": 0.9
            },
            "timestamp": datetime.now(UTC).isoformat()
        },
        {
            "role": "user",
            "content": "Can you analyze this document about machine learning?",
            "timestamp": datetime.now(UTC).isoformat()
        },
        {
            "role": "assistant",
            "content": "I'd be happy to analyze your document about machine learning! Please upload the document using the 📎 attachment button, and I'll provide a comprehensive analysis of its contents.",
            "metadata": {
                "sources_used": [],
                "processing_time": 0.8,
                "confidence_score": 0.8
            },
            "timestamp": datetime.now(UTC).isoformat()
        }
    ]

    st.session_state.messages = sample_messages
    st.session_state.total_queries = 2
    st.session_state.last_response_time = 1.0

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # Configure Streamlit page
    st.set_page_config(
        page_title="ESSENTRA - Your Intelligent Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="auto"
    )

    # Render the main UI
    render_main_ui()
