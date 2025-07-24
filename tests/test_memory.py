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
import sys
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime, timedelta, UTC
import uuid

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from memory import (
        ConversationMemory, ConversationTurn, SessionMetrics,
        MemoryStatistics, MemoryFactory, create_sample_conversation
    )
    print("✅ Successfully imported memory modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# ==================== MEMORY TESTER ====================

class MemoryTester:
    """Comprehensive memory system testing"""

    def __init__(self):
        self.test_results = []
        self.session_id = f"memory_test_{int(datetime.now(UTC).timestamp())}"
        self.test_dir = None

        print(f"🧠 Memory tester initialized")
        print(f"👤 User: TIRUMALAMANAV")
        print(f"🆔 Session: {self.session_id}")

    def setup_test_environment(self):
        """Setup temporary test environment"""
        self.test_dir = tempfile.mkdtemp(prefix="memory_test_")
        print(f"📁 Test directory: {self.test_dir}")
        return True

    def cleanup_test_environment(self):
        """Cleanup test environment"""
        if self.test_dir and Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)
            print(f"🧹 Cleaned up test directory")

    def test_conversation_turn_model(self):
        """Test ConversationTurn data model"""
        print("\n📝 Testing ConversationTurn Model...")

        try:
            # Test basic turn creation
            turn = ConversationTurn(
                user_message="What is AI?",
                assistant_response="AI is artificial intelligence.",
                session_id="test_session",
                sources_used=["source1.pdf"],
                confidence_score=0.85,
                response_time=1.5
            )

            # Test properties
            assert turn.user_message == "What is AI?"
            assert turn.assistant_response == "AI is artificial intelligence."
            assert turn.confidence_score == 0.85
            assert len(turn.turn_id) > 0
            assert isinstance(turn.timestamp, datetime)

            print("  ✅ Basic turn creation: PASSED")

            # Test methods
            context_str = turn.to_context_string()
            assert "User: What is AI?" in context_str
            assert "Assistant: AI is artificial intelligence." in context_str

            summary = turn.get_summary()
            assert "85.0%" in summary
            assert turn.turn_id[:8] in summary

            print("  ✅ Turn methods: PASSED")

            # Test JSON serialization
            turn_dict = turn.dict()
            assert isinstance(turn_dict, dict)
            assert turn_dict["user_message"] == "What is AI?"

            print("  ✅ JSON serialization: PASSED")

            self.test_results.append({'test': 'conversation_turn_model', 'passed': True})
            return True

        except Exception as e:
            print(f"  ❌ ConversationTurn test error: {str(e)}")
            self.test_results.append({'test': 'conversation_turn_model', 'passed': False, 'error': str(e)})
            return False

    def test_basic_memory_operations(self):
        """Test basic memory operations"""
        print("\n🧠 Testing Basic Memory Operations...")

        try:
            # Create memory instance
            memory = ConversationMemory(
                max_history=5,
                memory_window_hours=24,
                persist_directory=f"{self.test_dir}/memory",
                enable_persistence=False
            )

            test_session = "test_basic_session"

            # Test adding turns
            for i in range(3):
                turn = ConversationTurn(
                    user_message=f"Question {i+1}",
                    assistant_response=f"Answer {i+1}",
                    session_id=test_session,
                    confidence_score=0.8 + (i * 0.05)
                )

                success = memory.add_turn(test_session, turn)
                assert success is True

            print("  ✅ Adding turns: PASSED")

            # Test session exists
            assert test_session in memory.conversations
            assert len(memory.conversations[test_session]) == 3

            print("  ✅ Session management: PASSED")

            # Test context retrieval
            context = memory.get_recent_context(test_session, turns=2)
            assert "Question 2" in context
            assert "Question 3" in context
            assert "Answer 2" in context
            assert "Answer 3" in context

            print("  ✅ Context retrieval: PASSED")

            # Test context with metadata
            context_meta = memory.get_recent_context(test_session, turns=2, include_metadata=True)
            assert "Session Context" in context_meta
            assert "Total turns: 3" in context_meta

            print("  ✅ Context with metadata: PASSED")

            self.test_results.append({'test': 'basic_memory_operations', 'passed': True})
            return True

        except Exception as e:
            print(f"  ❌ Basic memory operations error: {str(e)}")
            self.test_results.append({'test': 'basic_memory_operations', 'passed': False, 'error': str(e)})
            return False

    def test_memory_analytics(self):
        """Test memory analytics features"""
        print("\n📊 Testing Memory Analytics...")

        try:
            memory = ConversationMemory(
                max_history=10,
                enable_persistence=False,
                enable_analytics=True
            )

            test_session = "analytics_test_session"

            # Add turns with various metrics
            turns_data = [
                {"user_message": "What is AI?", "confidence_score": 0.9, "intent_detected": "information", "tokens_used": 100},
                {"user_message": "How does ML work?", "confidence_score": 0.85, "intent_detected": "explanation", "tokens_used": 120},
                {"user_message": "Give me examples", "confidence_score": 0.8, "intent_detected": "examples", "tokens_used": 90}
            ]

            for turn_data in turns_data:
                turn = ConversationTurn(
                    session_id=test_session,
                    assistant_response="Test response",
                    **turn_data
                )
                memory.add_turn(test_session, turn)

            print("  ✅ Adding turns with analytics: PASSED")

            # Test session metrics
            assert test_session in memory.session_metrics
            metrics = memory.session_metrics[test_session]

            assert metrics.total_turns == 3
            assert metrics.total_tokens_used == 310  # 100 + 120 + 90
            assert 0.8 <= metrics.average_confidence <= 0.9

            print("  ✅ Session metrics: PASSED")

            # Test conversation summary
            summary = memory.get_conversation_summary(test_session)

            assert summary["session_id"] == test_session
            assert summary["conversation_stats"]["total_turns"] == 3
            assert summary["conversation_stats"]["unique_sources_used"] == 0  # No sources in test
            assert len(summary["timeline"]) == 3

            # Check intent analysis
            top_intents = summary["conversation_stats"]["top_intents"]
            assert "information" in top_intents
            assert "explanation" in top_intents
            assert "examples" in top_intents

            print("  ✅ Conversation summary: PASSED")

            # Test global statistics
            stats = memory.get_global_statistics()

            assert stats.total_sessions == 1
            assert stats.total_turns == 3
            assert stats.total_tokens_processed == 310

            print("  ✅ Global statistics: PASSED")

            self.test_results.append({'test': 'memory_analytics', 'passed': True})
            return True

        except Exception as e:
            print(f"  ❌ Memory analytics error: {str(e)}")
            self.test_results.append({'test': 'memory_analytics', 'passed': False, 'error': str(e)})
            return False

    def test_memory_persistence(self):
        """Test memory persistence features"""
        print("\n💾 Testing Memory Persistence...")

        try:
            persist_dir = f"{self.test_dir}/persistence_test"

            # Create memory with persistence enabled
            memory1 = ConversationMemory(
                max_history=10,
                persist_directory=persist_dir,
                enable_persistence=True
            )

            test_session = "persistence_test_session"

            # Add some turns
            for i in range(3):
                turn = ConversationTurn(
                    user_message=f"Persistent question {i+1}",
                    assistant_response=f"Persistent answer {i+1}",
                    session_id=test_session,
                    sources_used=[f"source_{i}.pdf"],
                    confidence_score=0.8
                )
                memory1.add_turn(test_session, turn)

            print("  ✅ Adding turns with persistence: PASSED")

            # Verify file creation
            session_file = Path(persist_dir) / f"session_{test_session}.json"
            assert session_file.exists()

            print("  ✅ Session file creation: PASSED")

            # Create new memory instance and load from disk
            memory2 = ConversationMemory(
                max_history=10,
                persist_directory=persist_dir,
                enable_persistence=True
            )

            # Verify data loaded
            assert test_session in memory2.conversations
            assert len(memory2.conversations[test_session]) == 3
            assert memory2.conversations[test_session][0].user_message == "Persistent question 1"
            assert test_session in memory2.session_metrics

            print("  ✅ Data loading from disk: PASSED")

            # Test context retrieval from loaded data
            context = memory2.get_recent_context(test_session, turns=2)
            assert "Persistent question 2" in context
            assert "Persistent question 3" in context

            print("  ✅ Context from loaded data: PASSED")

            self.test_results.append({'test': 'memory_persistence', 'passed': True})
            return True

        except Exception as e:
            print(f"  ❌ Memory persistence error: {str(e)}")
            self.test_results.append({'test': 'memory_persistence', 'passed': False, 'error': str(e)})
            return False

    def test_memory_search(self):
        """Test memory search functionality"""
        print("\n🔍 Testing Memory Search...")

        try:
            memory = ConversationMemory(enable_persistence=False)

            # Create multiple sessions with different content
            sessions_data = {
                "ai_session": [
                    {"user_message": "What is artificial intelligence?", "assistant_response": "AI is machine intelligence"},
                    {"user_message": "How does AI work?", "assistant_response": "AI uses algorithms and data"}
                ],
                "ml_session": [
                    {"user_message": "Explain machine learning", "assistant_response": "ML is a subset of AI"},
                    {"user_message": "What are neural networks?", "assistant_response": "Networks inspired by the brain"}
                ]
            }

            for session_id, turns_data in sessions_data.items():
                for turn_data in turns_data:
                    turn = ConversationTurn(
                        session_id=session_id,
                        **turn_data
                    )
                    memory.add_turn(session_id, turn)

            print("  ✅ Creating searchable data: PASSED")

            # Test search functionality
            ai_results = memory.search_conversations("artificial intelligence", max_results=5)
            assert len(ai_results) > 0
            assert any("artificial intelligence" in result["user_message"].lower() for result in ai_results)

            print("  ✅ Search by query: PASSED")

            # Test session-specific search
            ml_results = memory.search_conversations("machine learning", session_id="ml_session", max_results=5)
            assert len(ml_results) > 0
            assert all(result["session_id"] == "ml_session" for result in ml_results)

            print("  ✅ Session-specific search: PASSED")

            # Test relevance scoring
            for result in ai_results:
                assert "relevance_score" in result
                assert 0 <= result["relevance_score"] <= 1

            print("  ✅ Relevance scoring: PASSED")

            self.test_results.append({'test': 'memory_search', 'passed': True})
            return True

        except Exception as e:
            print(f"  ❌ Memory search error: {str(e)}")
            self.test_results.append({'test': 'memory_search', 'passed': False, 'error': str(e)})
            return False

    def test_export_functionality(self):
        """Test conversation export features"""
        print("\n📤 Testing Export Functionality...")

        try:
            memory = ConversationMemory(enable_persistence=False)
            test_session = "export_test_session"

            # Add test data
            create_sample_conversation(memory, test_session)

            print("  ✅ Creating exportable data: PASSED")

            # Test JSON export
            json_export = memory.export_conversation(test_session, format="json")
            assert json_export is not None

            json_data = json.loads(json_export)
            assert json_data["session_id"] == test_session
            assert "conversation" in json_data
            assert len(json_data["conversation"]) > 0

            print("  ✅ JSON export: PASSED")

            # Test Markdown export
            md_export = memory.export_conversation(test_session, format="markdown")
            assert md_export is not None
            assert f"# Conversation Export: {test_session}" in md_export
            assert "**User:**" in md_export
            assert "**Assistant:**" in md_export

            print("  ✅ Markdown export: PASSED")

            # Test Text export
            txt_export = memory.export_conversation(test_session, format="txt")
            assert txt_export is not None
            assert f"Conversation Export: {test_session}" in txt_export
            assert "User:" in txt_export
            assert "Assistant:" in txt_export

            print("  ✅ Text export: PASSED")

            # Test non-existent session
            none_export = memory.export_conversation("non_existent", format="json")
            assert none_export is None

            print("  ✅ Non-existent session handling: PASSED")

            self.test_results.append({'test': 'export_functionality', 'passed': True})
            return True

        except Exception as e:
            print(f"  ❌ Export functionality error: {str(e)}")
            self.test_results.append({'test': 'export_functionality', 'passed': False, 'error': str(e)})
            return False

    def test_memory_factory(self):
        """Test memory factory functionality"""
        print("\n🏭 Testing Memory Factory...")

        try:
            # Test  memory creation
            prod_memory = MemoryFactory.create_production_memory()

            assert isinstance(prod_memory, ConversationMemory)
            assert prod_memory.max_history == 50
            assert prod_memory.enable_persistence is True
            assert prod_memory.enable_analytics is True

            print("  ✅ Memory creation: PASSED")

            # Test custom configuration
            custom_config = {
                "max_history": 100,
                "memory_window_hours": 72,
                "enable_persistence": False
            }

            custom_memory = MemoryFactory.create_production_memory(custom_config)

            assert custom_memory.max_history == 100
            assert custom_memory.memory_window.total_seconds() == 72 * 3600
            assert custom_memory.enable_persistence is False

            print("  ✅ Custom configuration: PASSED")

            # Test test memory creation
            test_memory = MemoryFactory.create_test_memory()

            assert isinstance(test_memory, ConversationMemory)
            assert test_memory.max_history == 10
            assert test_memory.enable_persistence is False

            print("  ✅ Test memory creation: PASSED")

            self.test_results.append({'test': 'memory_factory', 'passed': True})
            return True

        except Exception as e:
            print(f"  ❌ Memory factory error: {str(e)}")
            self.test_results.append({'test': 'memory_factory', 'passed': False, 'error': str(e)})
            return False

    def print_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "="*80)
        print("📋 COMPREHENSIVE MEMORY SYSTEM TEST SUMMARY")
        print("="*80)
        print(f"👤 User: TIRUMALAMANAV")
        print(f"⏰ Time: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"🆔 Session: {self.session_id}")

        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['passed'])

        print(f"\n📊 OVERALL RESULTS:")
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")

        print(f"\n📋 DETAILED RESULTS:")
        for result in self.test_results:
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            test_name = result['test'].replace('_', ' ').title()
            print(f"  {status} - {test_name}")

            if 'error' in result:
                print(f"    Error: {result['error']}")

        print("\n" + "="*80)

        if passed_tests == total_tests:
            print("🎉 ALL MEMORY SYSTEM TESTS PASSED!")
            print("🧠 Memory system is  READY!")
        else:
            print("⚠️ Some memory tests failed.")
            print("💡 Review failed tests and fix issues.")

def main():
    """Main test runner"""
    print("🧠 MEMORY SYSTEM COMPREHENSIVE TEST SUITE")
    print(f"👤 User: TIRUMALAMANAV")
    print(f"📅 Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("🎯 Testing enhanced conversation memory system")
    print("=" * 80)

    tester = MemoryTester()

    try:
        # Setup test environment
        tester.setup_test_environment()

        # Run all tests
        tests = [
            tester.test_conversation_turn_model,
            tester.test_basic_memory_operations,
            tester.test_memory_analytics,
            tester.test_memory_persistence,
            tester.test_memory_search,
            tester.test_export_functionality,
            tester.test_memory_factory
        ]

        print(f"\n🚀 Running {len(tests)} comprehensive memory tests...")

        for i, test in enumerate(tests, 1):
            print(f"\n🔄 Running memory test {i}/{len(tests)}...")
            test()

    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Test suite error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        tester.cleanup_test_environment()
        tester.print_summary()

if __name__ == "__main__":
    main()
