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

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from document_processor import UniversalDocumentProcessor
    print("✅ Successfully imported document_processor")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

async def test_with_sample_data():
    print("🚀 Testing Document Processor with Sample Data")
    print(f"👤 User: TIRUMALAMANAV")
    print(f"⏰ Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)

    # Initialize processor
    processor = UniversalDocumentProcessor()
    session_id = "test_session_MANAV"

    # Your sample files (as specified in your document_processor.py)
    test_files = [
        "./sample_data/3.pdf",
        "./sample_data/1.docx",
        "./sample_data/5.csv",
        "./sample_data/6.pptx",
        "./sample_data/11.txt"
    ]

    print("📁 Testing with your sample files:")
    for file in test_files:
        if Path(file).exists():
            print(f"  ✅ Found: {file}")
        else:
            print(f"  ⚠️ Not found: {file}")

    print("\n📄 Processing Documents...")

    successful_files = 0
    total_chunks = 0

    for file_path in test_files:
        if not Path(file_path).exists():
            print(f"⏭️ Skipping {file_path} (not found)")
            continue

        print(f"\n🔄 Processing: {Path(file_path).name}")

        try:
            start_time = time.time()
            result = await processor.process_document(file_path, session_id)
            processing_time = time.time() - start_time

            if result.success:
                print(f"  ✅ Success!")
                print(f"  📊 Chunks created: {result.chunks_created}")
                print(f"  ⏱️ Time: {processing_time:.2f}s")
                print(f"  📏 File size: {result.metadata.file_size} bytes")
                print(f"  🏷️ Type: {result.metadata.file_type}")

                successful_files += 1
                total_chunks += result.chunks_created
            else:
                print(f"  ❌ Failed: {result.error_message}")

        except Exception as e:
            print(f"  💥 Exception: {str(e)}")

    print(f"\n📊 PROCESSING SUMMARY:")
    print(f"  Files processed successfully: {successful_files}/{len([f for f in test_files if Path(f).exists()])}")
    print(f"  Total chunks created: {total_chunks}")

    # Test retrieval if we have data
    if total_chunks > 0:
        print(f"\n🔍 Testing Retrieval...")

        test_queries = [
            "What is this document about?",
            "key information",
            "data summary"
        ]

        for query in test_queries:
            print(f"\n  Query: '{query}'")
            try:
                results = await processor.retrieve_similar_chunks(query, session_id, n_results=3)

                if results:
                    print(f"    ✅ Found {len(results)} results")
                    for i, result in enumerate(results[:2], 1):
                        score = result['similarity_score']
                        source = result['metadata'].get('source_file', 'unknown')
                        preview = result['content'][:60] + "..."
                        print(f"      {i}. Score: {score:.3f} | {source}")
                        print(f"         Preview: {preview}")
                else:
                    print(f"    ⚠️ No results found")

            except Exception as e:
                print(f"    ❌ Retrieval error: {str(e)}")

    # Session stats
    print(f"\n📈 Session Statistics:")
    stats = processor.get_session_stats(session_id)
    print(f"  Session ID: {stats['session_id']}")
    print(f"  Total files: {stats['total_files']}")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Files: {stats['files']}")

    print(f"\n🎉 Test completed successfully!")
    return successful_files > 0

if __name__ == "__main__":
    # Check if we're in the right directory
    if not os.path.exists("src"):
        print("❌ Please run from project root directory")
        print("   Current directory should contain 'src' folder")
        sys.exit(1)

    if not os.path.exists("sample_data"):
        print("❌ sample_data folder not found")
        print("   Make sure your sample files are in sample_data/")
        sys.exit(1)

    # Run the test
    success = asyncio.run(test_with_sample_data())

    if success:
        print("\n✅ DOCUMENT PROCESSOR TEST PASSED!")
    else:
        print("\n❌ Some issues found. Please check the errors above.")
