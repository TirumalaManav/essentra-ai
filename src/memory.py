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
import json
import pickle
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta, UTC
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
import uuid
import hashlib
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)

# ==================== ENHANCED DATA MODELS ====================

class ConversationTurn(BaseModel):
    """Enhanced conversation turn with metadata"""
    turn_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_message: str
    assistant_response: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    sources_used: List[str] = Field(default_factory=list)
    retrieval_context: List[Dict[str, Any]] = Field(default_factory=list)
    session_id: str

    # Enhanced metadata
    response_time: float = 0.0
    confidence_score: float = 0.0
    user_satisfaction: Optional[int] = None  # 1-5 rating
    model_used: str = "gemini-1.5-flash"
    intent_detected: str = ""
    routing_decision: str = ""

    # Analytics data
    tokens_used: int = 0
    search_results_count: int = 0
    retrieval_results_count: int = 0

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def to_context_string(self) -> str:
        """Convert to context string for memory retrieval"""
        return f"User: {self.user_message}\nAssistant: {self.assistant_response}"

    def get_summary(self) -> str:
        """Get brief summary of the turn"""
        user_preview = self.user_message[:50] + "..." if len(self.user_message) > 50 else self.user_message
        return f"Turn {self.turn_id[:8]}: {user_preview} (confidence: {self.confidence_score:.1%})"

@dataclass
class SessionMetrics:
    """Session-level analytics"""
    session_id: str
    total_turns: int = 0
    total_response_time: float = 0.0
    average_confidence: float = 0.0
    total_tokens_used: int = 0
    unique_sources: int = 0
    created_at: datetime = None
    last_activity: datetime = None
    user_satisfaction_avg: float = 0.0

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(UTC)
        if self.last_activity is None:
            self.last_activity = datetime.now(UTC)

    def update_from_turn(self, turn: ConversationTurn):
        """Update metrics from new conversation turn"""
        self.total_turns += 1
        self.total_response_time += turn.response_time
        self.total_tokens_used += turn.tokens_used
        self.last_activity = datetime.now(UTC)

        # Update averages
        if self.total_turns > 0:
            all_confidence = [self.average_confidence] * (self.total_turns - 1) + [turn.confidence_score]
            self.average_confidence = sum(all_confidence) / len(all_confidence)

@dataclass
class MemoryStatistics:
    """Global memory system statistics"""
    total_sessions: int = 0
    total_turns: int = 0
    total_tokens_processed: int = 0
    average_session_length: float = 0.0
    top_intents: Dict[str, int] = None
    top_sources: Dict[str, int] = None
    memory_size_mb: float = 0.0

    def __post_init__(self):
        if self.top_intents is None:
            self.top_intents = {}
        if self.top_sources is None:
            self.top_sources = {}

# ==================== ADVANCED MEMORY SYSTEM ====================

class ConversationMemory:
    """Production-grade conversation memory with persistence and analytics"""

    def __init__(self,
                 max_history: int = 50,
                 memory_window_hours: int = 168,  # 1 week
                 persist_directory: str = "./data/memory",
                 enable_persistence: bool = True,
                 enable_analytics: bool = True):

        # Core configuration
        self.max_history = max_history
        self.memory_window = timedelta(hours=memory_window_hours)
        self.persist_directory = Path(persist_directory)
        self.enable_persistence = enable_persistence
        self.enable_analytics = enable_analytics

        # Data storage
        self.conversations: Dict[str, List[ConversationTurn]] = {}
        self.session_metrics: Dict[str, SessionMetrics] = {}
        self.global_statistics = MemoryStatistics()

        # Setup persistence
        if self.enable_persistence:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

        logger.info(f"🧠 ConversationMemory initialized")
        logger.info(f"📊 Max history: {max_history}, Window: {memory_window_hours}h")
        logger.info(f"💾 Persistence: {enable_persistence}, Analytics: {enable_analytics}")

    def add_turn(self, session_id: str, turn: ConversationTurn) -> bool:
        """Add conversation turn with enhanced processing"""
        try:
            # Initialize session if new
            if session_id not in self.conversations:
                self.conversations[session_id] = []
                self.session_metrics[session_id] = SessionMetrics(session_id=session_id)

            # Add turn
            self.conversations[session_id].append(turn)

            # Update analytics
            if self.enable_analytics:
                self._update_analytics(session_id, turn)

            # Cleanup old turns
            self._cleanup_old_turns(session_id)

            # Persist changes
            if self.enable_persistence:
                self._save_session_to_disk(session_id)

            logger.info(f"🔄 Turn added to session {session_id[:8]}... (total: {len(self.conversations[session_id])})")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to add turn: {str(e)}")
            return False

    def get_recent_context(self, session_id: str, turns: int = 3,
                          include_metadata: bool = False) -> str:
        """Get enhanced recent conversation context"""
        if session_id not in self.conversations:
            return ""

        recent_turns = self.conversations[session_id][-turns:]
        if not recent_turns:
            return ""

        context_parts = []

        if include_metadata:
            # Add session context header
            metrics = self.session_metrics.get(session_id)
            if metrics:
                context_parts.append(f"Session Context (ID: {session_id[:8]}...)")
                context_parts.append(f"Total turns: {metrics.total_turns}, Avg confidence: {metrics.average_confidence:.1%}")
                context_parts.append("---")

        # Add conversation turns
        for i, turn in enumerate(recent_turns, 1):
            turn_header = f"Turn {i}" + (f" ({turn.intent_detected})" if turn.intent_detected else "")
            context_parts.append(f"**{turn_header}:**")
            context_parts.append(f"User: {turn.user_message}")
            context_parts.append(f"Assistant: {turn.assistant_response}")

            if include_metadata and turn.sources_used:
                context_parts.append(f"Sources: {', '.join(turn.sources_used[:3])}")

            context_parts.append("")  # Empty line for readability

        return "\n".join(context_parts)

    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive conversation summary"""
        if session_id not in self.conversations:
            return {"error": "Session not found"}

        turns = self.conversations[session_id]
        metrics = self.session_metrics.get(session_id, SessionMetrics(session_id=session_id))

        if not turns:
            return {"session_id": session_id, "turns": 0, "summary": "No conversation turns"}

        # Analyze conversation
        total_user_chars = sum(len(turn.user_message) for turn in turns)
        total_assistant_chars = sum(len(turn.assistant_response) for turn in turns)

        all_sources = []
        for turn in turns:
            all_sources.extend(turn.sources_used)
        unique_sources = list(set(all_sources))

        all_intents = [turn.intent_detected for turn in turns if turn.intent_detected]
        intent_counts = defaultdict(int)
        for intent in all_intents:
            intent_counts[intent] += 1

        # Recent activity
        last_turn = turns[-1] if turns else None
        time_since_last = (datetime.now(UTC) - last_turn.timestamp).total_seconds() / 60 if last_turn else 0

        summary = {
            "session_id": session_id,
            "session_metrics": asdict(metrics),
            "conversation_stats": {
                "total_turns": len(turns),
                "total_user_chars": total_user_chars,
                "total_assistant_chars": total_assistant_chars,
                "avg_user_message_length": total_user_chars / len(turns) if turns else 0,
                "avg_assistant_response_length": total_assistant_chars / len(turns) if turns else 0,
                "unique_sources_used": len(unique_sources),
                "top_intents": dict(intent_counts),
                "minutes_since_last_activity": round(time_since_last, 2)
            },
            "timeline": [
                {
                    "turn_id": turn.turn_id,
                    "timestamp": turn.timestamp.isoformat(),
                    "user_preview": turn.user_message[:100] + "..." if len(turn.user_message) > 100 else turn.user_message,
                    "confidence": turn.confidence_score,
                    "response_time": turn.response_time,
                    "intent": turn.intent_detected
                }
                for turn in turns[-10:]  # Last 10 turns
            ]
        }

        return summary

    def search_conversations(self, query: str, session_id: Optional[str] = None,
                           max_results: int = 5) -> List[Dict[str, Any]]:
        """Search through conversation history"""
        query_lower = query.lower()
        results = []

        # Determine sessions to search
        sessions_to_search = [session_id] if session_id else list(self.conversations.keys())

        for sid in sessions_to_search:
            if sid not in self.conversations:
                continue

            for turn in self.conversations[sid]:
                # Search in user message and assistant response
                user_match = query_lower in turn.user_message.lower()
                assistant_match = query_lower in turn.assistant_response.lower()

                if user_match or assistant_match:
                    relevance_score = 0.0

                    # Calculate relevance
                    if user_match:
                        relevance_score += 0.6
                    if assistant_match:
                        relevance_score += 0.4

                    # Boost recent conversations
                    days_old = (datetime.now(UTC) - turn.timestamp).days
                    recency_boost = max(0, (7 - days_old) / 7 * 0.2)
                    relevance_score += recency_boost

                    results.append({
                        "session_id": sid,
                        "turn_id": turn.turn_id,
                        "timestamp": turn.timestamp.isoformat(),
                        "user_message": turn.user_message,
                        "assistant_response": turn.assistant_response[:200] + "..." if len(turn.assistant_response) > 200 else turn.assistant_response,
                        "relevance_score": relevance_score,
                        "confidence": turn.confidence_score,
                        "sources_used": turn.sources_used
                    })

        # Sort by relevance and return top results
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:max_results]

    def get_session_list(self, active_only: bool = False,
                        sort_by: str = "last_activity") -> List[Dict[str, Any]]:
        """Get list of all sessions with metadata"""
        sessions = []

        for session_id, metrics in self.session_metrics.items():
            if session_id not in self.conversations:
                continue

            turns = self.conversations[session_id]

            # Filter active sessions
            if active_only:
                time_since_last = datetime.now(UTC) - metrics.last_activity
                if time_since_last > timedelta(hours=24):
                    continue

            session_info = {
                "session_id": session_id,
                "turn_count": len(turns),
                "created_at": metrics.created_at.isoformat(),
                "last_activity": metrics.last_activity.isoformat(),
                "average_confidence": metrics.average_confidence,
                "total_tokens": metrics.total_tokens_used,
                "first_message_preview": turns[0].user_message[:50] + "..." if turns else "",
                "last_message_preview": turns[-1].user_message[:50] + "..." if turns else "",
                "time_since_last_activity_minutes": (datetime.now(UTC) - metrics.last_activity).total_seconds() / 60
            }

            sessions.append(session_info)

        # Sort sessions
        if sort_by == "last_activity":
            sessions.sort(key=lambda x: x["last_activity"], reverse=True)
        elif sort_by == "turn_count":
            sessions.sort(key=lambda x: x["turn_count"], reverse=True)
        elif sort_by == "created_at":
            sessions.sort(key=lambda x: x["created_at"], reverse=True)

        return sessions

    def get_global_statistics(self) -> MemoryStatistics:
        """Get comprehensive memory system statistics"""
        stats = MemoryStatistics()

        stats.total_sessions = len(self.conversations)
        stats.total_turns = sum(len(turns) for turns in self.conversations.values())

        if stats.total_sessions > 0:
            stats.average_session_length = stats.total_turns / stats.total_sessions

        # Analyze all turns for patterns
        all_intents = defaultdict(int)
        all_sources = defaultdict(int)
        total_tokens = 0

        for turns in self.conversations.values():
            for turn in turns:
                if turn.intent_detected:
                    all_intents[turn.intent_detected] += 1

                for source in turn.sources_used:
                    all_sources[source] += 1

                total_tokens += turn.tokens_used

        stats.total_tokens_processed = total_tokens
        stats.top_intents = dict(sorted(all_intents.items(), key=lambda x: x[1], reverse=True)[:10])
        stats.top_sources = dict(sorted(all_sources.items(), key=lambda x: x[1], reverse=True)[:10])

        # Calculate memory size
        try:
            memory_size = self._calculate_memory_size()
            stats.memory_size_mb = memory_size
        except Exception:
            stats.memory_size_mb = 0.0

        return stats

    def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """Clean up sessions older than specified days"""
        cutoff_date = datetime.now(UTC) - timedelta(days=days_old)
        sessions_to_remove = []

        for session_id, metrics in self.session_metrics.items():
            if metrics.last_activity < cutoff_date:
                sessions_to_remove.append(session_id)

        # Remove old sessions
        for session_id in sessions_to_remove:
            if session_id in self.conversations:
                del self.conversations[session_id]
            if session_id in self.session_metrics:
                del self.session_metrics[session_id]

            # Remove from disk
            if self.enable_persistence:
                self._remove_session_from_disk(session_id)

        logger.info(f"🧹 Cleaned up {len(sessions_to_remove)} old sessions")
        return len(sessions_to_remove)

    def export_conversation(self, session_id: str, format: str = "json") -> Optional[str]:
        """Export conversation in various formats"""
        if session_id not in self.conversations:
            return None

        turns = self.conversations[session_id]

        if format == "json":
            data = {
                "session_id": session_id,
                "exported_at": datetime.now(UTC).isoformat(),
                "conversation": [turn.dict() for turn in turns]
            }
            return json.dumps(data, indent=2, default=str)

        elif format == "markdown":
            lines = [f"# Conversation Export: {session_id}", f"Exported: {datetime.now(UTC).isoformat()}", ""]

            for i, turn in enumerate(turns, 1):
                lines.append(f"## Turn {i}")
                lines.append(f"**Time:** {turn.timestamp.isoformat()}")
                lines.append(f"**User:** {turn.user_message}")
                lines.append(f"**Assistant:** {turn.assistant_response}")
                if turn.sources_used:
                    lines.append(f"**Sources:** {', '.join(turn.sources_used)}")
                lines.append("")

            return "\n".join(lines)

        elif format == "txt":
            lines = [f"Conversation Export: {session_id}", f"Exported: {datetime.now(UTC).isoformat()}", "=" * 50, ""]

            for i, turn in enumerate(turns, 1):
                lines.append(f"Turn {i} - {turn.timestamp.isoformat()}")
                lines.append(f"User: {turn.user_message}")
                lines.append(f"Assistant: {turn.assistant_response}")
                lines.append("-" * 30)

            return "\n".join(lines)

        return None

    # ==================== PRIVATE METHODS ====================

    def _update_analytics(self, session_id: str, turn: ConversationTurn):
        """Update analytics data"""
        if session_id in self.session_metrics:
            self.session_metrics[session_id].update_from_turn(turn)

    def _cleanup_old_turns(self, session_id: str):
        """Enhanced cleanup with analytics preservation"""
        if session_id not in self.conversations:
            return

        cutoff_time = datetime.now(UTC) - self.memory_window

        # Filter by time
        valid_turns = [
            turn for turn in self.conversations[session_id]
            if turn.timestamp > cutoff_time
        ]

        # Keep only last N turns
        self.conversations[session_id] = valid_turns[-self.max_history:]

        logger.debug(f"🧹 Cleaned session {session_id[:8]}... (kept {len(self.conversations[session_id])} turns)")

    def _save_session_to_disk(self, session_id: str):
        """Save session data to disk"""
        if not self.enable_persistence:
            return

        try:
            session_file = self.persist_directory / f"session_{session_id}.json"

            data = {
                "session_id": session_id,
                "conversation": [turn.dict() for turn in self.conversations[session_id]],
                "metrics": asdict(self.session_metrics.get(session_id, SessionMetrics(session_id=session_id))),
                "saved_at": datetime.now(UTC).isoformat()
            }

            with open(session_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"❌ Failed to save session {session_id}: {str(e)}")

    def _load_from_disk(self):
        """Load all sessions from disk"""
        if not self.persist_directory.exists():
            return

        loaded_sessions = 0

        try:
            for session_file in self.persist_directory.glob("session_*.json"):
                try:
                    with open(session_file, 'r') as f:
                        data = json.load(f)

                    session_id = data["session_id"]

                    # Load conversation turns
                    turns = []
                    for turn_data in data["conversation"]:
                        # Handle datetime parsing
                        if isinstance(turn_data["timestamp"], str):
                            turn_data["timestamp"] = datetime.fromisoformat(turn_data["timestamp"].replace('Z', '+00:00'))

                        turns.append(ConversationTurn(**turn_data))

                    self.conversations[session_id] = turns

                    # Load metrics
                    if "metrics" in data:
                        metrics_data = data["metrics"]
                        if isinstance(metrics_data["created_at"], str):
                            metrics_data["created_at"] = datetime.fromisoformat(metrics_data["created_at"].replace('Z', '+00:00'))
                        if isinstance(metrics_data["last_activity"], str):
                            metrics_data["last_activity"] = datetime.fromisoformat(metrics_data["last_activity"].replace('Z', '+00:00'))

                        self.session_metrics[session_id] = SessionMetrics(**metrics_data)

                    loaded_sessions += 1

                except Exception as e:
                    logger.warning(f"⚠️ Failed to load session from {session_file}: {str(e)}")

            if loaded_sessions > 0:
                logger.info(f"📂 Loaded {loaded_sessions} sessions from disk")

        except Exception as e:
            logger.error(f"❌ Failed to load sessions from disk: {str(e)}")

    def _remove_session_from_disk(self, session_id: str):
        """Remove session file from disk"""
        try:
            session_file = self.persist_directory / f"session_{session_id}.json"
            if session_file.exists():
                session_file.unlink()
        except Exception as e:
            logger.warning(f"⚠️ Failed to remove session file {session_id}: {str(e)}")

    def _calculate_memory_size(self) -> float:
        """Calculate approximate memory usage in MB"""
        try:
            total_size = 0

            # Calculate size of conversations
            conversations_str = json.dumps(
                {sid: [turn.dict() for turn in turns] for sid, turns in self.conversations.items()},
                default=str
            )
            total_size += len(conversations_str.encode('utf-8'))

            # Calculate size of metrics
            metrics_str = json.dumps(
                {sid: asdict(metrics) for sid, metrics in self.session_metrics.items()},
                default=str
            )
            total_size += len(metrics_str.encode('utf-8'))

            return total_size / (1024 * 1024)  # Convert to MB

        except Exception:
            return 0.0

# ==================== MEMORY FACTORY ====================

class MemoryFactory:
    """Factory for creating memory instances"""

    @staticmethod
    def create_production_memory(config_dict: Optional[Dict[str, Any]] = None) -> ConversationMemory:
        """Create production-ready memory instance"""
        default_config = {
            "max_history": 50,
            "memory_window_hours": 168,  # 1 week
            "persist_directory": "./data/memory",
            "enable_persistence": True,
            "enable_analytics": True
        }

        if config_dict:
            default_config.update(config_dict)

        return ConversationMemory(**default_config)

    @staticmethod
    def create_test_memory() -> ConversationMemory:
        """Create memory instance for testing"""
        return ConversationMemory(
            max_history=10,
            memory_window_hours=1,
            persist_directory="./test_data/memory",
            enable_persistence=False,
            enable_analytics=True
        )

# ==================== TESTING FUNCTIONALITY ====================

def create_sample_conversation(memory: ConversationMemory, session_id: str) -> str:
    """Create sample conversation for testing"""
    sample_turns = [
        {
            "user_message": "What is artificial intelligence?",
            "assistant_response": "Artificial intelligence (AI) is a field of computer science focused on creating systems that can perform tasks typically requiring human intelligence, such as learning, reasoning, and problem-solving.",
            "sources_used": ["ai_textbook.pdf", "wikipedia_ai"],
            "confidence_score": 0.9,
            "intent_detected": "information_request",
            "response_time": 1.2,
            "tokens_used": 150
        },
        {
            "user_message": "How does machine learning work?",
            "assistant_response": "Machine learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed. It works by training algorithms on data to identify patterns and make predictions.",
            "sources_used": ["ml_guide.pdf", "coursera_ml"],
            "confidence_score": 0.85,
            "intent_detected": "explanation_request",
            "response_time": 1.5,
            "tokens_used": 180
        },
        {
            "user_message": "Can you give me an example of neural networks?",
            "assistant_response": "Neural networks are computing systems inspired by biological neural networks. A simple example is image recognition - a neural network can be trained on thousands of cat photos to learn patterns and then identify cats in new images.",
            "sources_used": ["neural_nets.pdf"],
            "confidence_score": 0.8,
            "intent_detected": "example_request",
            "response_time": 1.1,
            "tokens_used": 140
        }
    ]

    for turn_data in sample_turns:
        turn = ConversationTurn(
            session_id=session_id,
            **turn_data
        )
        memory.add_turn(session_id, turn)

    return f"Created sample conversation with {len(sample_turns)} turns"

if __name__ == "__main__":
    print(f"👤 User: TIRUMALAMANAV")
    print(f"📅 Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 60)

    # Create memory instance
    memory = MemoryFactory.create_test_memory()
    test_session = "test_session_MANAV"

    # Create sample data
    result = create_sample_conversation(memory, test_session)
    print(f"✅ {result}")

    # Test context retrieval
    context = memory.get_recent_context(test_session, turns=2, include_metadata=True)
    print(f"\n📖 Recent context preview:")
    print(context[:200] + "..." if len(context) > 200 else context)

    # Test summary
    summary = memory.get_conversation_summary(test_session)
    print(f"\n📊 Conversation summary:")
    print(f"  • Total turns: {summary['conversation_stats']['total_turns']}")
    print(f"  • Avg confidence: {summary['session_metrics']['average_confidence']:.1%}")
    print(f"  • Sources used: {summary['conversation_stats']['unique_sources_used']}")


