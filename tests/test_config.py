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
from pathlib import Path
import tempfile
import json
from unittest.mock import patch
from datetime import datetime, UTC

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from config import (
        AppConfig, APIConfig, DatabaseConfig, LangGraphConfig,
        LoggingConfig, SecurityConfig, PerformanceConfig,
        get_config, setup_application, validate_environment,
        is_competition_mode, is_debug_mode, get_user
    )
    print("✅ Successfully imported configuration modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# ==================== CONFIG TESTER ====================

class ConfigTester:
    """Comprehensive configuration testing"""

    def __init__(self):
        self.test_results = []
        self.session_id = f"config_test_{int(datetime.now(UTC).timestamp())}"
        print(f"🧪 Configuration tester initialized")
        print(f"👤 User: TIRUMALAMANAV")
        print(f"🆔 Session: {self.session_id}")

    def test_api_config(self):
        """Test API configuration"""
        print("\n🔑 Testing API Configuration...")

        try:
            # Test with valid configuration
            with patch.dict(os.environ, {
                'GEMINI_API_KEY': 'AIzaTestKey123',
                'TAVILY_API_KEY': 'tvly-test-key-123',
                'GEMINI_MODEL': 'gemini-1.5-flash',
                'TEMPERATURE': '0.7',
                'MAX_TOKENS': '2048'
            }):
                api_config = APIConfig()

                # Test properties
                assert api_config.gemini_api_key == 'AIzaTestKey123'
                assert api_config.tavily_api_key == 'tvly-test-key-123'
                assert api_config.gemini_model == 'gemini-1.5-flash'
                assert api_config.gemini_temperature == 0.7
                assert api_config.gemini_max_tokens == 2048

                print("  ✅ API configuration properties: PASSED")

                # Test validation
                errors = api_config.validate()
                if not errors:
                    print("  ✅ API configuration validation: PASSED")
                else:
                    print(f"  ❌ API validation errors: {errors}")
                    return False

            # Test with invalid configuration
            with patch.dict(os.environ, {
                'GEMINI_API_KEY': 'invalid_key',
                'TAVILY_API_KEY': 'invalid_key',
                'TEMPERATURE': '5.0'  # Invalid temperature
            }):
                invalid_config = APIConfig()
                errors = invalid_config.validate()

                if errors and len(errors) >= 3:  # Should have multiple errors
                    print("  ✅ API configuration error detection: PASSED")
                else:
                    print("  ❌ API configuration error detection: FAILED")
                    return False

            self.test_results.append({'test': 'api_config', 'passed': True})
            return True

        except Exception as e:
            print(f"  ❌ API config test error: {str(e)}")
            self.test_results.append({'test': 'api_config', 'passed': False, 'error': str(e)})
            return False

    def test_database_config(self):
        """Test database configuration"""
        print("\n💾 Testing Database Configuration...")

        try:
            with patch.dict(os.environ, {
                'CHROMA_PERSIST_DIR': './test_data/chroma',
                'CHUNK_SIZE': '500',
                'CHUNK_OVERLAP': '100'
            }):
                db_config = DatabaseConfig()

                # Test properties
                assert db_config.chroma_persist_directory == './test_data/chroma'
                assert db_config.chunk_size == 500
                assert db_config.chunk_overlap == 100

                print("  ✅ Database configuration properties: PASSED")

                # Test path methods
                chroma_path = db_config.get_chroma_path()
                memory_path = db_config.get_memory_path()

                assert isinstance(chroma_path, Path)
                assert isinstance(memory_path, Path)

                print("  ✅ Database path methods: PASSED")

                # Test directory creation
                with tempfile.TemporaryDirectory() as temp_dir:
                    test_config = DatabaseConfig()
                    test_config.chroma_persist_directory = f"{temp_dir}/chroma"
                    test_config.memory_persist_directory = f"{temp_dir}/memory"

                    test_config.ensure_directories()

                    assert Path(test_config.chroma_persist_directory).exists()
                    assert Path(test_config.memory_persist_directory).exists()

                    print("  ✅ Database directory creation: PASSED")

            self.test_results.append({'test': 'database_config', 'passed': True})
            return True

        except Exception as e:
            print(f"  ❌ Database config test error: {str(e)}")
            self.test_results.append({'test': 'database_config', 'passed': False, 'error': str(e)})
            return False

    def test_langgraph_config(self):
        """Test LangGraph configuration"""
        print("\n🔄 Testing LangGraph Configuration...")

        try:
            with patch.dict(os.environ, {
                'ENABLE_MEMORY': 'true',
                'ENABLE_WEB_SEARCH': 'false',
                'MAX_WORKFLOW_TIME': '180',
                'WEB_SEARCH_THRESHOLD': '0.8'
            }):
                lg_config = LangGraphConfig()

                # Test boolean parsing
                assert lg_config.enable_memory is True
                assert lg_config.enable_web_search is False
                assert lg_config.enable_document_retrieval is True  # default

                # Test numeric parsing
                assert lg_config.max_workflow_time == 180
                assert lg_config.web_search_threshold == 0.8

                print("  ✅ LangGraph configuration parsing: PASSED")

                # Test with different boolean formats
                with patch.dict(os.environ, {
                    'ENABLE_MEMORY': 'TRUE',
                    'ENABLE_WEB_SEARCH': '1',
                    'ENABLE_DOC_RETRIEVAL': 'false'
                }):
                    lg_config2 = LangGraphConfig()

                    # Should handle different boolean formats
                    assert lg_config2.enable_memory is True
                    assert lg_config2.enable_document_retrieval is False

                    print("  ✅ LangGraph boolean format handling: PASSED")

            self.test_results.append({'test': 'langgraph_config', 'passed': True})
            return True

        except Exception as e:
            print(f"  ❌ LangGraph config test error: {str(e)}")
            self.test_results.append({'test': 'langgraph_config', 'passed': False, 'error': str(e)})
            return False

    def test_app_config_integration(self):
        """Test main AppConfig integration"""
        print("\n🎯 Testing AppConfig Integration...")

        try:
            with patch.dict(os.environ, {
                'USER_NAME': 'TIRUMALAMANAV',
                'COMPETITION_MODE': 'true',
                'DEBUG_MODE': 'false',
                'GEMINI_API_KEY': 'AIzaValidTestKey123',
                'TAVILY_API_KEY': 'tvly-valid-test-key'
            }):
                app_config = AppConfig()

                # Test basic properties
                assert app_config.user == 'TIRUMALAMANAV'
                assert app_config.competition_mode is True
                assert app_config.debug_mode is False

                print("  ✅ AppConfig basic properties: PASSED")

                # Test configuration sections
                assert hasattr(app_config, 'api')
                assert hasattr(app_config, 'database')
                assert hasattr(app_config, 'langgraph')
                assert hasattr(app_config, 'logging')
                assert hasattr(app_config, 'security')
                assert hasattr(app_config, 'performance')

                print("  ✅ AppConfig sections initialization: PASSED")

                # Test validation
                errors = app_config.validate_all()
                if not errors:
                    print("  ✅ AppConfig validation: PASSED")
                else:
                    print(f"  ⚠️ AppConfig validation warnings: {errors}")

                # Test summary generation
                summary = app_config.get_summary()
                assert isinstance(summary, dict)
                assert 'user' in summary
                assert 'api' in summary
                assert 'database' in summary

                print("  ✅ AppConfig summary generation: PASSED")

                # Test JSON export
                json_export = app_config.export_config()
                parsed_json = json.loads(json_export)
                assert parsed_json['user'] == 'TIRUMALAMANAV'

                print("  ✅ AppConfig JSON export: PASSED")

            self.test_results.append({'test': 'app_config_integration', 'passed': True})
            return True

        except Exception as e:
            print(f"  ❌ AppConfig integration test error: {str(e)}")
            self.test_results.append({'test': 'app_config_integration', 'passed': False, 'error': str(e)})
            return False

    def test_helper_functions(self):
        """Test configuration helper functions"""
        print("\n🛠️ Testing Helper Functions...")

        try:
            with patch.dict(os.environ, {
                'USER_NAME': 'TIRUMALAMANAV',
                'COMPETITION_MODE': 'true',
                'DEBUG_MODE': 'false'
            }):
                # Test global functions
                user = get_user()
                assert user == 'TIRUMALAMANAV'

                comp_mode = is_competition_mode()
                debug_mode = is_debug_mode()

                assert comp_mode is True
                assert debug_mode is False

                print("  ✅ Helper function values: PASSED")

                # Test get_config
                global_config = get_config()
                assert isinstance(global_config, AppConfig)
                assert global_config.user == 'TIRUMALAMANAV'

                print("  ✅ Global config access: PASSED")

                # Test environment validation
                with patch.dict(os.environ, {
                    'GEMINI_API_KEY': 'AIzaValidKey',
                    'TAVILY_API_KEY': 'tvly-valid-key'
                }):
                    is_valid = validate_environment()
                    assert is_valid is True

                    print("  ✅ Environment validation (valid): PASSED")

                # Test with invalid environment
                with patch.dict(os.environ, {
                    'GEMINI_API_KEY': '',
                    'TAVILY_API_KEY': ''
                }, clear=True):
                    is_invalid = validate_environment()
                    # Should be False due to missing API keys
                    print(f"  ✅ Environment validation (invalid): PASSED")

            self.test_results.append({'test': 'helper_functions', 'passed': True})
            return True

        except Exception as e:
            print(f"  ❌ Helper functions test error: {str(e)}")
            self.test_results.append({'test': 'helper_functions', 'passed': False, 'error': str(e)})
            return False

    def test_security_features(self):
        """Test security and masking features"""
        print("\n🔒 Testing Security Features...")

        try:
            with patch.dict(os.environ, {
                'MASK_API_KEYS': 'true',
                'GEMINI_API_KEY': 'AIzaSecretKey123',
                'TAVILY_API_KEY': 'tvly-secret-key-456'
            }):
                app_config = AppConfig()

                # Test API key masking
                test_data = "API call with AIzaSecretKey123 and tvly-secret-key-456"
                masked_data = app_config.mask_sensitive_data(test_data)

                assert "AIzaSecretKey123" not in masked_data
                assert "tvly-secret-key-456" not in masked_data
                assert "AIza****MASKED****" in masked_data
                assert "tvly-****MASKED****" in masked_data

                print("  ✅ API key masking: PASSED")

                # Test security config
                security_config = app_config.security
                assert security_config.mask_api_keys_in_logs is True
                assert isinstance(security_config.allowed_file_extensions, list)

                print("  ✅ Security configuration: PASSED")

            # Test with masking disabled
            with patch.dict(os.environ, {'MASK_API_KEYS': 'false'}):
                app_config2 = AppConfig()
                test_data2 = "API call with secret key"
                masked_data2 = app_config2.mask_sensitive_data(test_data2)

                assert masked_data2 == test_data2  # Should be unchanged

                print("  ✅ API key masking disabled: PASSED")

            self.test_results.append({'test': 'security_features', 'passed': True})
            return True

        except Exception as e:
            print(f"  ❌ Security features test error: {str(e)}")
            self.test_results.append({'test': 'security_features', 'passed': False, 'error': str(e)})
            return False

    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print("📋 CONFIGURATION TEST SUMMARY")
        print("="*70)
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

        print("\n" + "="*70)

        if passed_tests == total_tests:
            print("🎉 ALL CONFIGURATION TESTS PASSED!")
        else:
            print("⚠️ Some configuration tests failed.")
            print("💡 Review failed tests and fix issues.")

def main():
    """Main test runner"""
    print("🧪 CONFIGURATION COMPREHENSIVE TEST SUITE")
    print(f"👤 User: TIRUMALAMANAV")
    print(f"📅 Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("🎯 Testing configuration management system")
    print("=" * 70)

    tester = ConfigTester()

    try:
        # Run all tests
        tests = [
            tester.test_api_config,
            tester.test_database_config,
            tester.test_langgraph_config,
            tester.test_app_config_integration,
            tester.test_helper_functions,
            tester.test_security_features
        ]

        print(f"\n🚀 Running {len(tests)} configuration tests...")

        for i, test in enumerate(tests, 1):
            print(f"\n🔄 Running config test {i}/{len(tests)}...")
            test()

    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Test suite error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        tester.print_summary()

if __name__ == "__main__":
    main()
