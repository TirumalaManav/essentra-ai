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



import asyncio
import sys
import os
from pathlib import Path
import time
import uuid
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import json

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from llm_clients import (
        GeminiClient,
        TavilyWebSearchClient,
        LLMClientFactory,
        APIMetrics
    )
    print("✅ Successfully imported LLM client modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you have installed required dependencies:")
    print("   pip install google-generativeai tavily-python python-dotenv aiohttp backoff")
    sys.exit(1)

# ==================== MOCK CLASSES ====================

class MockGeminiResponse:
    """Mock Gemini API response"""
    def __init__(self, text: str):
        self.text = text

class MockTavilyResponse:
    """Mock Tavily API response"""
    def __init__(self, results: list):
        self.results = results

# ==================== LLMCLIENT TESTER ====================

class LLMClientTester:
    """Comprehensive LLM client testing suite"""

    def __init__(self):
        self.test_results = []
        self.session_id = f"test_session_llm_{int(time.time())}"
        self.test_start_time = datetime.utcnow()

        print(f"🧪 Initialized LLM client tester")
        print(f"👤 User: TIRUMALAMANAV")
        print(f"🆔 Session: {self.session_id}")
        print(f"⏰ Start time: {self.test_start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")

    async def test_gemini_client_mock(self):
        """Test Gemini client with mocked API calls"""
        print("\n🤖 Testing Gemini Client (Mock Mode)...")

        try:
            # Test with mocked environment
            with patch.dict(os.environ, {
                'GEMINI_API_KEY': 'mock_key_for_testing',
                'GEMINI_MODEL': 'gemini-1.5-flash',
                'MAX_TOKENS': '2048',
                'TEMPERATURE': '0.7'
            }):
                # Mock the genai module
                with patch('llm_clients.genai') as mock_genai:
                    # Setup mock model
                    mock_model = Mock()
                    mock_response = MockGeminiResponse("This is a mock response from Gemini 1.5 Flash. The AI provides comprehensive information about artificial intelligence and machine learning concepts.")
                    mock_model.generate_content.return_value = mock_response
                    mock_genai.GenerativeModel.return_value = mock_model

                    # Create client
                    client = GeminiClient()

                    # Test basic response generation
                    print("  📝 Testing basic response generation...")
                    response = await client.generate_response("What is artificial intelligence?")

                    if response and len(response) > 50:
                        print(f"    ✅ Basic generation: PASSED")
                        print(f"    📏 Response length: {len(response)} chars")

                        # Test contextual generation
                        print("  📚 Testing contextual generation...")
                        contexts = [
                            {
                                "content": "Artificial intelligence is the simulation of human intelligence processes by machines.",
                                "source": "ai_textbook.pdf",
                                "similarity_score": 0.9,
                                "rank": 1
                            },
                            {
                                "content": "Machine learning is a subset of AI that enables computers to learn without being explicitly programmed.",
                                "source": "ml_guide.pdf",
                                "similarity_score": 0.85,
                                "rank": 2
                            }
                        ]

                        contextual_result = await client.generate_with_context(
                            "Explain AI and ML", contexts, "Previous discussion about technology"
                        )

                        if contextual_result and contextual_result.get("response"):
                            print(f"    ✅ Contextual generation: PASSED")
                            print(f"    🎯 Confidence: {contextual_result.get('confidence_score', 0):.1%}")
                            print(f"    📚 Sources used: {len(contextual_result.get('sources_used', []))}")

                            # Test metrics
                            metrics = client.get_metrics()
                            print(f"    📊 API calls made: {metrics['total_requests']}")
                            print(f"    ✅ Success rate: {metrics['success_rate']}")

                            test_passed = True
                        else:
                            print(f"    ❌ Contextual generation: FAILED")
                            test_passed = False
                    else:
                        print(f"    ❌ Basic generation: FAILED")
                        test_passed = False

            self.test_results.append({
                'test': 'gemini_client_mock',
                'passed': test_passed,
                'timestamp': datetime.utcnow()
            })

            return test_passed

        except Exception as e:
            print(f"    ❌ Gemini mock test error: {str(e)}")
            self.test_results.append({
                'test': 'gemini_client_mock',
                'passed': False,
                'error': str(e)
            })
            return False

    async def test_tavily_client_mock(self):
        """Test Tavily client with mocked API calls"""
        print("\n🌐 Testing Tavily Client (Mock Mode)...")

        try:
            with patch.dict(os.environ, {
                'TAVILY_API_KEY': 'mock_tavily_key',
                'MAX_SEARCH_RESULTS': '5'
            }):
                # Mock the TavilyClient
                with patch('llm_clients.TavilyClient') as mock_tavily_class:
                    mock_tavily_instance = Mock()
                    mock_search_response = {
                        "results": [
                            {
                                "title": "Latest AI Research Breakthroughs 2025",
                                "content": "Recent advances in artificial intelligence include improved large language models and breakthrough applications in healthcare.",
                                "url": "https://ai-research.com/2025-breakthroughs",
                                "published_date": "2025-07-20",
                                "score": 0.95
                            },
                            {
                                "title": "Machine Learning Applications in Industry",
                                "content": "Machine learning is transforming various industries through predictive analytics and automation technologies.",
                                "url": "https://ml-industry.com/applications",
                                "published_date": "2025-07-19",
                                "score": 0.88
                            },
                            {
                                "title": "AI Ethics and Future Considerations",
                                "content": "As AI systems become more powerful, ethical considerations and responsible development practices are crucial.",
                                "url": "https://ai-ethics.org/future-considerations",
                                "published_date": "2025-07-18",
                                "score": 0.82
                            }
                        ]
                    }

                    mock_tavily_instance.search.return_value = mock_search_response
                    mock_tavily_class.return_value = mock_tavily_instance

                    # Create client
                    client = TavilyWebSearchClient()

                    # Test basic search
                    print("  🔍 Testing basic web search...")
                    search_results = await client.search("latest AI developments 2025")

                    if search_results and len(search_results) > 0:
                        print(f"    ✅ Basic search: PASSED")
                        print(f"    📊 Results found: {len(search_results)}")
                        print(f"    🏆 Top result: {search_results[0].get('title', 'N/A')}")

                        # Test enhanced search with context
                        print("  🧠 Testing enhanced search with context...")
                        enhanced_result = await client.search_with_context(
                            "AI breakthroughs",
                            "Previous discussion about machine learning and neural networks",
                            max_results=3
                        )

                        if enhanced_result and enhanced_result.get("search_results"):
                            print(f"    ✅ Enhanced search: PASSED")
                            print(f"    📈 Search quality: {enhanced_result.get('search_quality', 'unknown')}")
                            print(f"    ⏱️ Processing time: {enhanced_result.get('processing_time', 0):.2f}s")

                            # Test result scoring and processing
                            first_result = enhanced_result["search_results"][0]
                            print(f"    🎯 Top result relevance: {first_result.get('composite_score', 0):.1%}")
                            print(f"    📝 Source type: {first_result.get('source_type', 'unknown')}")

                            # Test metrics
                            metrics = client.get_metrics()
                            print(f"    📊 Search calls made: {metrics['total_requests']}")
                            print(f"    ✅ Success rate: {metrics['success_rate']}")

                            test_passed = True
                        else:
                            print(f"    ❌ Enhanced search: FAILED")
                            test_passed = False
                    else:
                        print(f"    ❌ Basic search: FAILED")
                        test_passed = False

            self.test_results.append({
                'test': 'tavily_client_mock',
                'passed': test_passed,
                'timestamp': datetime.utcnow()
            })

            return test_passed

        except Exception as e:
            print(f"    ❌ Tavily mock test error: {str(e)}")
            self.test_results.append({
                'test': 'tavily_client_mock',
                'passed': False,
                'error': str(e)
            })
            return False

    async def test_client_factory(self):
        """Test LLM client factory functionality"""
        print("\n🏭 Testing LLM Client Factory...")

        try:
            print("  🔧 Testing client creation...")

            # Test connection testing
            with patch.dict(os.environ, {
                'GEMINI_API_KEY': 'test_key',
                'TAVILY_API_KEY': 'test_key'
            }):
                with patch('llm_clients.genai'), patch('llm_clients.TavilyClient'):
                    connections = LLMClientFactory.test_client_connections()

                    if "gemini" in connections and "tavily" in connections:
                        print(f"    ✅ Connection testing: PASSED")
                        print(f"    🤖 Gemini status: {connections['gemini']}")
                        print(f"    🌐 Tavily status: {connections['tavily']}")

                        # Test client creation
                        clients = LLMClientFactory.create_all_clients()

                        if "gemini" in clients and "tavily" in clients:
                            print(f"    ✅ All clients creation: PASSED")
                            test_passed = True
                        else:
                            print(f"    ❌ All clients creation: FAILED")
                            test_passed = False
                    else:
                        print(f"    ❌ Connection testing: FAILED")
                        test_passed = False

            self.test_results.append({
                'test': 'client_factory',
                'passed': test_passed,
                'timestamp': datetime.utcnow()
            })

            return test_passed

        except Exception as e:
            print(f"    ❌ Factory test error: {str(e)}")
            self.test_results.append({
                'test': 'client_factory',
                'passed': False,
                'error': str(e)
            })
            return False

    async def test_api_metrics(self):
        """Test API metrics functionality"""
        print("\n📊 Testing API Metrics...")

        try:
            # Create metrics instance
            metrics = APIMetrics()

            # Test initial state
            print("  📈 Testing initial metrics state...")
            if metrics.total_requests == 0 and metrics.get_success_rate() == 0.0:
                print(f"    ✅ Initial state: PASSED")

                # Test success updates
                print("  ✅ Testing success metric updates...")
                metrics.update_success(1.5, 100)  # 1.5s response, 100 tokens
                metrics.update_success(2.0, 150)  # 2.0s response, 150 tokens

                if (metrics.total_requests == 2 and
                    metrics.successful_requests == 2 and
                    metrics.get_success_rate() == 100.0):
                    print(f"    ✅ Success updates: PASSED")
                    print(f"    📊 Success rate: {metrics.get_success_rate():.1f}%")
                    print(f"    ⏱️ Avg response time: {metrics.average_response_time:.2f}s")
                    print(f"    🔢 Total tokens: {metrics.total_tokens_used}")

                    # Test failure updates
                    print("  ❌ Testing failure metric updates...")
                    metrics.update_failure(0.5)  # Failed request

                    if (metrics.total_requests == 3 and
                        metrics.failed_requests == 1 and
                        66.0 <= metrics.get_success_rate() <= 67.0):  # 2/3 = 66.67%
                        print(f"    ✅ Failure updates: PASSED")
                        print(f"    📊 Final success rate: {metrics.get_success_rate():.1f}%")
                        test_passed = True
                    else:
                        print(f"    ❌ Failure updates: FAILED")
                        test_passed = False
                else:
                    print(f"    ❌ Success updates: FAILED")
                    test_passed = False
            else:
                print(f"    ❌ Initial state: FAILED")
                test_passed = False

            self.test_results.append({
                'test': 'api_metrics',
                'passed': test_passed,
                'timestamp': datetime.utcnow()
            })

            return test_passed

        except Exception as e:
            print(f"    ❌ Metrics test error: {str(e)}")
            self.test_results.append({
                'test': 'api_metrics',
                'passed': False,
                'error': str(e)
            })
            return False

    async def test_real_api_integration(self):
        """Test real API integration (if keys are available)"""
        print("\n🔑 Testing Real API Integration...")

        # Check if real API keys are available
        gemini_key = os.getenv("GEMINI_API_KEY")
        tavily_key = os.getenv("TAVILY_API_KEY")

        if not gemini_key or not tavily_key:
            print("  ⚠️ Real API keys not found - skipping real API tests")
            print("  💡 Set GEMINI_API_KEY and TAVILY_API_KEY for real API testing")
            self.test_results.append({
                'test': 'real_api_integration',
                'passed': True,  # Not a failure, just skipped
                'skipped': True,
                'reason': 'API keys not available'
            })
            return True

        try:
            print("  🔑 Real API keys found - testing live integration...")

            # Test Gemini real API
            print("  🤖 Testing real Gemini API...")
            try:
                gemini_client = GeminiClient()
                gemini_response = await gemini_client.generate_response(
                    "Hello! Please respond with a brief test message to confirm API connectivity."
                )

                if gemini_response and len(gemini_response) > 10:
                    print(f"    ✅ Real Gemini API: PASSED")
                    print(f"    📝 Response preview: {gemini_response[:100]}...")
                    gemini_success = True
                else:
                    print(f"    ❌ Real Gemini API: FAILED")
                    gemini_success = False
            except Exception as e:
                print(f"    ❌ Real Gemini API error: {str(e)}")
                gemini_success = False

            # Test Tavily real API
            print("  🌐 Testing real Tavily API...")
            try:
                tavily_client = TavilyWebSearchClient()
                search_results = await tavily_client.search("AI news 2025", max_results=2)

                if search_results and len(search_results) > 0:
                    print(f"    ✅ Real Tavily API: PASSED")
                    print(f"    📊 Results found: {len(search_results)}")
                    print(f"    🏆 Top result: {search_results[0].get('title', 'N/A')[:50]}...")
                    tavily_success = True
                else:
                    print(f"    ❌ Real Tavily API: FAILED")
                    tavily_success = False
            except Exception as e:
                print(f"    ❌ Real Tavily API error: {str(e)}")
                tavily_success = False

            # Overall real API test result
            test_passed = gemini_success and tavily_success

            if test_passed:
                print(f"  🎉 Real API integration: ALL PASSED!")
                print(f"  🚀-ready for competition!")
            else:
                print(f"  ⚠️ Some real API tests failed")
                print(f"  💡 Check API keys and network connectivity")

            self.test_results.append({
                'test': 'real_api_integration',
                'passed': test_passed,
                'gemini_success': gemini_success,
                'tavily_success': tavily_success,
                'timestamp': datetime.utcnow()
            })

            return test_passed

        except Exception as e:
            print(f"  ❌ Real API integration error: {str(e)}")
            self.test_results.append({
                'test': 'real_api_integration',
                'passed': False,
                'error': str(e)
            })
            return False

    def print_comprehensive_summary(self):
        """Print detailed test summary"""
        print("\n" + "="*80)
        print("📋 COMPREHENSIVE LLM CLIENT TEST SUMMARY")
        print("="*80)
        print(f"👤 User: TIRUMALAMANAV")
        print(f"⏰ Test completed: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"🆔 Session: {self.session_id}")
        print(f"⏱️ Total test duration: {(datetime.utcnow() - self.test_start_time).total_seconds():.2f}s")

        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['passed'])
        skipped_tests = sum(1 for result in self.test_results if result.get('skipped', False))

        print(f"\n📊 OVERALL RESULTS:")
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests - skipped_tests}")
        print(f"Skipped: {skipped_tests}")
        print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")

        print(f"\n📋 DETAILED RESULTS:")
        for result in self.test_results:
            if result.get('skipped', False):
                status = "⏭️ SKIP"
                reason = f" - {result.get('reason', 'Unknown')}"
            elif result['passed']:
                status = "✅ PASS"
                reason = ""
            else:
                status = "❌ FAIL"
                reason = f" - {result.get('error', 'Unknown error')}"

            test_name = result['test'].replace('_', ' ').title()
            print(f"  {status} - {test_name}{reason}")

            # Show additional details for specific tests
            if result['test'] == 'real_api_integration' and not result.get('skipped', False):
                gemini_status = "✅" if result.get('gemini_success', False) else "❌"
                tavily_status = "✅" if result.get('tavily_success', False) else "❌"
                print(f"    {gemini_status} Gemini 1.5 Flash API")
                print(f"    {tavily_status} Tavily Search API")

        print("\n" + "="*80)

        if passed_tests == total_tests:
            print("🎉 ALL LLM CLIENT TESTS PASSED!")
            print("🔥 Gemini 1.5 Flash + Tavily integration PERFECT!")
        else:
            success_rate = (passed_tests/total_tests)*100
            if success_rate >= 80:
                print("EXCELLENT test performance!")
                print("System is ready with minor issues!")
            elif success_rate >= 60:
                print("GOOD test performance!")
                print("Some optimizations recommended.")
            else:
                print("Some critical issues need attention.")
                print("Review failed tests and fix issues.")


async def main():
    """Main test runner for LLM clients"""
    print("🧪 LLM CLIENT COMPREHENSIVE TEST SUITE")
    print(f"👤 User: TIRUMALAMANAV")
    print(f"📅 Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("🎯 Testing LLM clients")
    print("=" * 80)

    tester = LLMClientTester()

    try:
        # Run all tests
        tests = [
            tester.test_api_metrics(),
            tester.test_gemini_client_mock(),
            tester.test_tavily_client_mock(),
            tester.test_client_factory(),
            tester.test_real_api_integration()
        ]

        print(f"\n🚀 Running {len(tests)} comprehensive LLM client tests...")

        for i, test in enumerate(tests, 1):
            print(f"\n🔄 Running LLM test {i}/{len(tests)}...")
            await test
            await asyncio.sleep(0.1)  # Brief pause between tests

    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Test suite error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        tester.print_comprehensive_summary()

if __name__ == "__main__":
    # Ensure proper environment
    if not os.path.exists("src"):
        print("❌ Please run this test from the project root directory")
        print("   Current directory should contain 'src' folder")
        sys.exit(1)

    # Run the comprehensive LLM client test suite
    asyncio.run(main())
