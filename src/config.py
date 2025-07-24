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


import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, UTC
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging for this module
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION DATACLASSES ====================

@dataclass
class APIConfig:
    """API configuration settings"""
    # Gemini Configuration
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
    gemini_temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.7")))
    gemini_max_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "2048")))

    # Tavily Configuration
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    tavily_max_results: int = field(default_factory=lambda: int(os.getenv("MAX_SEARCH_RESULTS", "5")))

    # Rate Limiting
    gemini_rate_limit: float = field(default_factory=lambda: float(os.getenv("GEMINI_RATE_LIMIT", "0.1")))
    tavily_rate_limit: float = field(default_factory=lambda: float(os.getenv("TAVILY_RATE_LIMIT", "0.5")))

    # Retry Configuration
    max_retries: int = field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "3")))
    retry_timeout: int = field(default_factory=lambda: int(os.getenv("RETRY_TIMEOUT", "30")))

    def validate(self) -> List[str]:
        """Validate API configuration and return any errors"""
        errors = []

        if not self.gemini_api_key:
            errors.append("GEMINI_API_KEY is required")
        elif not self.gemini_api_key.startswith("AIza"):
            errors.append("GEMINI_API_KEY appears to be invalid format")

        if not self.tavily_api_key:
            errors.append("TAVILY_API_KEY is required")
        elif not self.tavily_api_key.startswith("tvly-"):
            errors.append("TAVILY_API_KEY appears to be invalid format")

        if not 0.0 <= self.gemini_temperature <= 2.0:
            errors.append("TEMPERATURE must be between 0.0 and 2.0")

        if self.gemini_max_tokens <= 0:
            errors.append("MAX_TOKENS must be positive")

        if self.tavily_max_results <= 0:
            errors.append("MAX_SEARCH_RESULTS must be positive")

        return errors

@dataclass
class DatabaseConfig:
    """Database and storage configuration"""
    # ChromaDB Configuration
    chroma_persist_directory: str = field(default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db"))
    chroma_collection_name: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION", "documents"))

    # Vector Database Settings
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    vector_dimension: int = field(default_factory=lambda: int(os.getenv("VECTOR_DIMENSION", "384")))

    # Document Processing
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "400")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50")))
    max_file_size_mb: int = field(default_factory=lambda: int(os.getenv("MAX_FILE_SIZE_MB", "100")))

    # Memory Configuration
    memory_persist_directory: str = field(default_factory=lambda: os.getenv("MEMORY_PERSIST_DIR", "./data/memory"))
    max_conversation_turns: int = field(default_factory=lambda: int(os.getenv("MAX_CONVERSATION_TURNS", "50")))

    def get_chroma_path(self) -> Path:
        """Get ChromaDB path as Path object"""
        return Path(self.chroma_persist_directory)

    def get_memory_path(self) -> Path:
        """Get memory path as Path object"""
        return Path(self.memory_persist_directory)

    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        self.get_chroma_path().mkdir(parents=True, exist_ok=True)
        self.get_memory_path().mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Database directories ensured")

@dataclass
class LangGraphConfig:
    """LangGraph workflow configuration"""
    # Workflow Settings
    enable_memory: bool = field(default_factory=lambda: os.getenv("ENABLE_MEMORY", "true").lower() == "true")
    enable_web_search: bool = field(default_factory=lambda: os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true")
    enable_document_retrieval: bool = field(default_factory=lambda: os.getenv("ENABLE_DOC_RETRIEVAL", "true").lower() == "true")

    # Performance Settings
    max_workflow_time: int = field(default_factory=lambda: int(os.getenv("MAX_WORKFLOW_TIME", "120")))  # seconds
    max_context_length: int = field(default_factory=lambda: int(os.getenv("MAX_CONTEXT_LENGTH", "8000")))

    # Intent Detection Thresholds
    web_search_threshold: float = field(default_factory=lambda: float(os.getenv("WEB_SEARCH_THRESHOLD", "0.7")))
    document_retrieval_threshold: float = field(default_factory=lambda: float(os.getenv("DOC_RETRIEVAL_THRESHOLD", "0.8")))

    # Quality Control
    min_confidence_score: float = field(default_factory=lambda: float(os.getenv("MIN_CONFIDENCE", "0.3")))
    max_retrieval_results: int = field(default_factory=lambda: int(os.getenv("MAX_RETRIEVAL_RESULTS", "5")))

@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_format: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    log_file: Optional[str] = field(default_factory=lambda: os.getenv("LOG_FILE"))

    # Competition Logging
    enable_performance_logging: bool = field(default_factory=lambda: os.getenv("ENABLE_PERF_LOGGING", "true").lower() == "true")
    enable_trace_logging: bool = field(default_factory=lambda: os.getenv("ENABLE_TRACE_LOGGING", "true").lower() == "true")

    def setup_logging(self):
        """Setup application logging"""
        level = getattr(logging, self.log_level.upper(), logging.INFO)

        # Configure root logger
        logging.basicConfig(
            level=level,
            format=self.log_format,
            handlers=self._get_handlers()
        )

        logger.info(f"✅ Logging configured: {self.log_level}")

    def _get_handlers(self) -> List[logging.Handler]:
        """Get logging handlers"""
        handlers = [logging.StreamHandler()]

        if self.log_file:
            # Ensure log directory exists
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(logging.Formatter(self.log_format))
            handlers.append(file_handler)

        return handlers

@dataclass
class SecurityConfig:
    """Security and privacy configuration"""
    # API Key Security
    mask_api_keys_in_logs: bool = field(default_factory=lambda: os.getenv("MASK_API_KEYS", "true").lower() == "true")

    # Session Security
    session_timeout_minutes: int = field(default_factory=lambda: int(os.getenv("SESSION_TIMEOUT", "60")))
    max_sessions_per_user: int = field(default_factory=lambda: int(os.getenv("MAX_SESSIONS", "5")))

    # File Upload Security
    allowed_file_extensions: List[str] = field(default_factory=lambda:
        os.getenv("ALLOWED_EXTENSIONS", "pdf,docx,txt,md").split(","))
    scan_uploads: bool = field(default_factory=lambda: os.getenv("SCAN_UPLOADS", "true").lower() == "true")

    # Data Privacy
    auto_delete_sessions: bool = field(default_factory=lambda: os.getenv("AUTO_DELETE_SESSIONS", "false").lower() == "true")
    data_retention_days: int = field(default_factory=lambda: int(os.getenv("DATA_RETENTION_DAYS", "30")))

@dataclass
class PerformanceConfig:
    """Performance optimization settings"""
    # Concurrency
    max_concurrent_requests: int = field(default_factory=lambda: int(os.getenv("MAX_CONCURRENT", "10")))
    max_agent_workers: int = field(default_factory=lambda: int(os.getenv("MAX_AGENT_WORKERS", "5")))

    # Caching
    enable_response_caching: bool = field(default_factory=lambda: os.getenv("ENABLE_CACHING", "true").lower() == "true")
    cache_ttl_minutes: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL", "30")))

    # Memory Management
    max_memory_usage_mb: int = field(default_factory=lambda: int(os.getenv("MAX_MEMORY_MB", "2048")))
    enable_memory_monitoring: bool = field(default_factory=lambda: os.getenv("ENABLE_MEM_MONITOR", "true").lower() == "true")

    # Competition Optimizations
    enable_fast_mode: bool = field(default_factory=lambda: os.getenv("ENABLE_FAST_MODE", "false").lower() == "true")
    priority_processing: bool = field(default_factory=lambda: os.getenv("PRIORITY_PROCESSING", "true").lower() == "true")

# ==================== MAIN CONFIG CLASS ====================

class AppConfig:
    """Main application configuration manager"""

    def __init__(self):
        self.user = os.getenv("USER_NAME", "TIRUMALAMANAV")
        self.version = "1.0.0"
        self.competition_mode = os.getenv("COMPETITION_MODE", "true").lower() == "true"
        self.debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"

        # Initialize all configuration sections
        self.api = APIConfig()
        self.database = DatabaseConfig()
        self.langgraph = LangGraphConfig()
        self.logging = LoggingConfig()
        self.security = SecurityConfig()
        self.performance = PerformanceConfig()

        # Initialize timestamp
        self.initialized_at = datetime.now(UTC)

        logger.info(f"🎯 Configuration initialized for user: {self.user}")

    def validate_all(self) -> Dict[str, List[str]]:
        """Validate all configuration sections"""
        validation_errors = {
            "api": self.api.validate(),
            "general": []
        }

        # General validations
        if not self.user:
            validation_errors["general"].append("USER_NAME is required")

        return {k: v for k, v in validation_errors.items() if v}

    def setup_environment(self):
        """Setup the complete environment"""
        logger.info(f"🔧 Setting up environment for {self.user}...")

        # Setup logging first
        self.logging.setup_logging()

        # Ensure directories
        self.database.ensure_directories()

        # Validate configuration
        errors = self.validate_all()
        if errors:
            logger.error(f"❌ Configuration validation failed: {errors}")
            raise ValueError(f"Configuration validation failed: {errors}")

        logger.info(f"✅ Environment setup completed successfully!")

        if self.competition_mode:
            logger.info(f"COMPETITION MODE ENABLED - Maximum performance activated!")

        if self.debug_mode:
            logger.info(f"🔍 DEBUG MODE ENABLED - Detailed logging activated!")

    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            "user": self.user,
            "version": self.version,
            "competition_mode": self.competition_mode,
            "debug_mode": self.debug_mode,
            "initialized_at": self.initialized_at.isoformat(),
            "api": {
                "gemini_model": self.api.gemini_model,
                "gemini_max_tokens": self.api.gemini_max_tokens,
                "tavily_max_results": self.api.tavily_max_results,
                "has_gemini_key": bool(self.api.gemini_api_key),
                "has_tavily_key": bool(self.api.tavily_api_key)
            },
            "database": {
                "chroma_collection": self.database.chroma_collection_name,
                "chunk_size": self.database.chunk_size,
                "embedding_model": self.database.embedding_model
            },
            "langgraph": {
                "memory_enabled": self.langgraph.enable_memory,
                "web_search_enabled": self.langgraph.enable_web_search,
                "doc_retrieval_enabled": self.langgraph.enable_document_retrieval
            },
            "performance": {
                "max_concurrent": self.performance.max_concurrent_requests,
                "caching_enabled": self.performance.enable_response_caching,
                "fast_mode": self.performance.enable_fast_mode
            }
        }

    def export_config(self, file_path: Optional[str] = None) -> str:
        """Export configuration to JSON file"""
        config_data = self.get_summary()

        if file_path:
            config_path = Path(file_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)

            logger.info(f"📄 Configuration exported to: {config_path}")
            return str(config_path)
        else:
            return json.dumps(config_data, indent=2)

    def mask_sensitive_data(self, data: str) -> str:
        """Mask sensitive information in logs/output"""
        if not self.security.mask_api_keys_in_logs:
            return data

        masked = data

        # Mask API keys
        if self.api.gemini_api_key and self.api.gemini_api_key in masked:
            masked = masked.replace(self.api.gemini_api_key, "AIza****MASKED****")

        if self.api.tavily_api_key and self.api.tavily_api_key in masked:
            masked = masked.replace(self.api.tavily_api_key, "tvly-****MASKED****")

        return masked

# ==================== GLOBAL CONFIG INSTANCE ====================

# Create global configuration instance
config = AppConfig()

# ==================== HELPER FUNCTIONS ====================

def get_config() -> AppConfig:
    """Get the global configuration instance"""
    return config

def setup_application() -> AppConfig:
    """Setup the complete application with configuration"""
    config.setup_environment()
    return config

def is_competition_mode() -> bool:
    """Check if running in competition mode"""
    return config.competition_mode

def is_debug_mode() -> bool:
    """Check if running in debug mode"""
    return config.debug_mode

def get_user() -> str:
    """Get current user name"""
    return config.user

def validate_environment() -> bool:
    """Validate the environment is properly configured"""
    try:
        errors = config.validate_all()
        return len(errors) == 0
    except Exception:
        return False

# ==================== TESTING FUNCTIONALITY ====================

def test_configuration():
    """Test configuration setup and validation"""
    print("🧪 Testing Configuration System")
    print(f"👤 User: {config.user}")
    print(f"📅 Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 60)

    try:
        # Test validation
        print("🔧 Testing configuration validation...")
        errors = config.validate_all()

        if errors:
            print("❌ Configuration validation errors found:")
            for section, section_errors in errors.items():
                print(f"  📋 {section.title()}:")
                for error in section_errors:
                    print(f"    ❌ {error}")
        else:
            print("✅ Configuration validation passed!")

        # Test environment setup
        print("\n🌍 Testing environment setup...")
        config.setup_environment()
        print("✅ Environment setup completed!")

        # Test configuration summary
        print("\n📊 Configuration Summary:")
        summary = config.get_summary()
        for section, data in summary.items():
            if isinstance(data, dict):
                print(f"  📋 {section.title()}:")
                for key, value in data.items():
                    print(f"    • {key}: {value}")
            else:
                print(f"  • {section}: {data}")

        # Test export
        print("\n💾 Testing configuration export...")
        exported_json = config.export_config()
        print("✅ Configuration export successful!")

        print(f"\n🎉 ALL CONFIGURATION TESTS PASSED!")

        return True

    except Exception as e:
        print(f"\n❌ Configuration test failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Run configuration tests
    test_configuration()
