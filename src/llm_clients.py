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
import asyncio
import logging
import time
import json
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import aiohttp
import google.generativeai as genai
from tavily import TavilyClient
from dotenv import load_dotenv
import backoff

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== PERFORMANCE METRICS ====================

@dataclass
class APIMetrics:
    """Track API performance metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    average_response_time: float = 0.0
    total_tokens_used: int = 0
    last_request_time: Optional[datetime] = None

    def update_success(self, response_time: float, tokens_used: int = 0):
        """Update metrics for successful request"""
        self.total_requests += 1
        self.successful_requests += 1
        self.total_response_time += response_time
        self.average_response_time = self.total_response_time / self.total_requests
        self.total_tokens_used += tokens_used
        self.last_request_time = datetime.utcnow()

    def update_failure(self, response_time: float = 0.0):
        """Update metrics for failed request"""
        self.total_requests += 1
        self.failed_requests += 1
        self.total_response_time += response_time
        if self.total_requests > 0:
            self.average_response_time = self.total_response_time / self.total_requests
        self.last_request_time = datetime.utcnow()

    def get_success_rate(self) -> float:
        """Get success rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

# ==================== GEMINI 1.5 FLASH CLIENT ====================

class GeminiClient:

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "2048"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))

        if not self.api_key:
            raise ValueError("❌ GEMINI_API_KEY not found in environment variables")

        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

        # Performance configuration
        self.generation_config = {
            'temperature': self.temperature,
            'max_output_tokens': self.max_tokens,
            'top_p': 0.8,
            'top_k': 40
        }

        # Metrics tracking
        self.metrics = APIMetrics()

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

        logger.info(f"🤖 Gemini client initialized: {self.model_name}")
        logger.info(f"🎯 Configuration: temp={self.temperature}, max_tokens={self.max_tokens}")

    @backoff.on_exception(
        backoff.expo,
        (Exception,),
        max_tries=3,
        max_time=30
    )
    async def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using Gemini 1.5 Flash with retry logic"""
        start_time = time.time()

        try:
            # Rate limiting
            await self._rate_limit()

            # Enhance prompt with context if provided
            enhanced_prompt = self._enhance_prompt(prompt, context)

            logger.info(f"🧠 Generating response with Gemini 1.5 Flash...")
            logger.info(f"📝 Prompt length: {len(enhanced_prompt)} chars")

            # Generate response
            response = await asyncio.to_thread(
                self.model.generate_content,
                enhanced_prompt,
                generation_config=self.generation_config
            )

            # Process response
            if response.text:
                generated_text = response.text.strip()

                # Calculate metrics
                response_time = time.time() - start_time
                token_count = self._estimate_tokens(generated_text)

                # Update metrics
                self.metrics.update_success(response_time, token_count)

                logger.info(f"✅ Gemini response generated successfully!")
                logger.info(f"📏 Response length: {len(generated_text)} chars")
                logger.info(f"🔢 Estimated tokens: {token_count}")
                logger.info(f"⏱️ Response time: {response_time:.2f}s")

                return generated_text

            else:
                # Handle empty response
                response_time = time.time() - start_time
                self.metrics.update_failure(response_time)

                logger.warning("⚠️ Empty response from Gemini")
                return self._create_fallback_response(prompt)

        except Exception as e:
            response_time = time.time() - start_time
            self.metrics.update_failure(response_time)

            error_msg = str(e)
            logger.error(f"❌ Gemini API error: {error_msg}")

            # Return intelligent error response
            return self._create_error_response(prompt, error_msg)

    async def generate_with_context(self, query: str, contexts: List[Dict[str, Any]],
                                  conversation_history: str = "") -> Dict[str, Any]:
        """Generate contextual response with sources"""
        start_time = time.time()

        try:
            # Format context for Gemini
            context_text = self._format_contexts(contexts)

            # Create comprehensive prompt
            prompt = self._create_contextual_prompt(
                query, context_text, conversation_history
            )

            # Generate response
            response_text = await self.generate_response(prompt)

            # Extract sources
            sources_used = [ctx.get("source", "Unknown") for ctx in contexts[:3]]

            # Calculate confidence based on context quality
            confidence = self._calculate_confidence(contexts, response_text)

            processing_time = time.time() - start_time

            result = {
                "response": response_text,
                "sources_used": sources_used,
                "context_count": len(contexts),
                "confidence_score": confidence,
                "processing_time": processing_time,
                "model_used": self.model_name,
                "timestamp": datetime.utcnow().isoformat()
            }

            logger.info(f"📚 Contextual response generated with {len(contexts)} sources")
            logger.info(f"🎯 Confidence score: {confidence:.1%}")

            return result

        except Exception as e:
            logger.error(f"❌ Contextual generation error: {str(e)}")
            return {
                "response": self._create_error_response(query, str(e)),
                "sources_used": [],
                "context_count": 0,
                "confidence_score": 0.0,
                "processing_time": time.time() - start_time,
                "error": str(e)
            }

    def _enhance_prompt(self, prompt: str, context: str = "") -> str:
        """Enhance prompt with system instructions and context"""
        system_prompt = f"""You are an intelligent AI assistant helping TIRUMALAMANAV with their queries.

**Instructions:**
- Provide accurate, comprehensive, and well-structured responses
- Use markdown formatting for better readability
- Include relevant examples when helpful
- Be precise and informative
- Maintain a professional but friendly tone

Current Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
"""

        if context:
            enhanced = f"{system_prompt}\n\n**Context Information:**\n{context}\n\n**User Query:**\n{prompt}\n\n**Response:**"
        else:
            enhanced = f"{system_prompt}\n\n**User Query:**\n{prompt}\n\n**Response:**"

        return enhanced

    def _format_contexts(self, contexts: List[Dict[str, Any]]) -> str:
        """Format context documents for optimal LLM consumption"""
        if not contexts:
            return ""

        formatted_parts = []
        for i, ctx in enumerate(contexts[:5], 1):  # Top 5 contexts
            content = ctx.get("content", "")
            source = ctx.get("source", "Unknown")
            score = ctx.get("similarity_score", 0.0)

            formatted_parts.append(
                f"**Source {i} ({source}) - Relevance: {score:.1%}**\n{content}"
            )

        return "\n\n".join(formatted_parts)

    def _create_contextual_prompt(self, query: str, context_text: str,
                                conversation_history: str = "") -> str:
        """Create comprehensive contextual prompt"""

        prompt_parts = [
            f"You are helping TIRUMALAMANAV with an intelligent response based on provided context.",
            "",
            "**Context Information:**",
            context_text,
        ]

        if conversation_history:
            prompt_parts.extend([
                "",
                "**Previous Conversation:**",
                conversation_history,
            ])

        prompt_parts.extend([
            "",
            f"**Current Question:**",
            query,
            "",
            "**Instructions:**",
            "- Provide a comprehensive answer based on the context provided",
            "- Include specific details and examples from the sources",
            "- Use clear markdown formatting with headers and bullet points",
            "- If the context doesn't fully answer the question, mention what's available",
            "- Be accurate and cite information appropriately",
            "",
            "**Your Response:**"
        ])

        return "\n".join(prompt_parts)

    def _calculate_confidence(self, contexts: List[Dict[str, Any]], response: str) -> float:
        """Calculate confidence score based on context quality and response"""
        if not contexts:
            return 0.3

        # Base confidence from similarity scores
        avg_similarity = sum(ctx.get("similarity_score", 0.0) for ctx in contexts) / len(contexts)

        # Boost for multiple sources
        source_diversity = len(set(ctx.get("source", "") for ctx in contexts))
        diversity_boost = min(source_diversity * 0.1, 0.2)

        # Response quality (length and structure)
        response_quality = min(len(response) / 500, 1.0) * 0.1

        # Final confidence
        confidence = avg_similarity + diversity_boost + response_quality
        return min(confidence, 0.95)  # Cap at 95%

    def _create_fallback_response(self, prompt: str) -> str:
        """Create fallback response for empty API responses"""
        return f"""I apologize, but I'm having difficulty generating a response to your question: "{prompt[:100]}..."

This might be due to:
- Temporary API limitations
- Complex query processing requirements
- Network connectivity issues

**Suggestions:**
- Try rephrasing your question
- Break complex questions into smaller parts
- Check if you need to upload relevant documents

Please try again, and I'll do my best to provide a helpful response!

*Session: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC*"""

    def _create_error_response(self, prompt: str, error: str) -> str:
        """Create user-friendly error response"""
        return f"""I encountered an issue while processing your request: "{prompt[:100]}..."

**Technical Details:** {error}

**What you can do:**
- Wait a moment and try again
- Simplify your question
- Check your internet connection
- Contact support if the issue persists

I'm working to resolve this and provide you with the best possible assistance!

*Error logged at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC*"""

    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            await asyncio.sleep(sleep_time)

        self.last_request_time = current_time

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        return len(text.split()) + len(text) // 4

    def get_metrics(self) -> Dict[str, Any]:
        """Get API performance metrics"""
        return {
            **asdict(self.metrics),
            "model_name": self.model_name,
            "success_rate": f"{self.metrics.get_success_rate():.1f}%"
        }

# ==================== TAVILY WEB SEARCH CLIENT ====================

class TavilyWebSearchClient:
    def __init__(self):
        self.api_key = os.getenv("TAVILY_API_KEY")
        self.max_results = int(os.getenv("MAX_SEARCH_RESULTS", "5"))

        if not self.api_key:
            raise ValueError("❌ TAVILY_API_KEY not found in environment variables")

        # Initialize Tavily client
        self.client = TavilyClient(api_key=self.api_key)

        # Metrics tracking
        self.metrics = APIMetrics()

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.5  # 500ms between searches

        logger.info(f"🌐 Tavily client initialized successfully")
        logger.info(f"🔍 Max results per search: {self.max_results}")

    @backoff.on_exception(
        backoff.expo,
        (Exception,),
        max_tries=3,
        max_time=30
    )
    async def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """Perform web search with advanced result processing"""
        start_time = time.time()

        try:
            # Rate limiting
            await self._rate_limit()

            # Use provided max_results or default
            results_limit = max_results or self.max_results

            logger.info(f"🌐 Performing web search: '{query[:50]}...'")
            logger.info(f"🔍 Max results: {results_limit}")

            # Perform search
            search_results = await asyncio.to_thread(
                self.client.search,
                query=query,
                max_results=results_limit,
                search_depth="advanced",
                include_domains=None,
                exclude_domains=["ads", "spam"],
                include_answer=True,
                include_raw_content=True
            )

            # Process and enhance results
            processed_results = self._process_search_results(search_results, query)

            # Update metrics
            response_time = time.time() - start_time
            self.metrics.update_success(response_time)

            logger.info(f"✅ Web search completed successfully!")
            logger.info(f"📊 Found {len(processed_results)} relevant results")
            logger.info(f"⏱️ Search time: {response_time:.2f}s")

            return processed_results

        except Exception as e:
            response_time = time.time() - start_time
            self.metrics.update_failure(response_time)

            error_msg = str(e)
            logger.error(f"❌ Tavily search error: {error_msg}")

            # Return fallback results
            return self._create_fallback_results(query, error_msg)

    async def search_with_context(self, query: str, context: str = "",
                                max_results: Optional[int] = None) -> Dict[str, Any]:
        """Enhanced search with context and metadata"""
        start_time = time.time()

        try:
            # Enhance query with context
            enhanced_query = self._enhance_search_query(query, context)

            # Perform search
            results = await self.search(enhanced_query, max_results)

            # Calculate relevance and quality scores
            scored_results = self._score_results(results, query)

            # Generate search summary
            search_summary = self._generate_search_summary(scored_results, query)

            processing_time = time.time() - start_time

            return {
                "search_results": scored_results,
                "search_summary": search_summary,
                "query_used": enhanced_query,
                "total_results": len(scored_results),
                "processing_time": processing_time,
                "search_quality": self._assess_search_quality(scored_results),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Enhanced search error: {str(e)}")
            return {
                "search_results": [],
                "search_summary": f"Search failed: {str(e)}",
                "query_used": query,
                "total_results": 0,
                "processing_time": time.time() - start_time,
                "error": str(e)
            }

    def _process_search_results(self, raw_results: Dict[str, Any],
                              original_query: str) -> List[Dict[str, Any]]:
        """Process and enhance raw search results"""
        if not raw_results or "results" not in raw_results:
            return []

        processed = []

        for i, result in enumerate(raw_results["results"]):
            processed_result = {
                "rank": i + 1,
                "title": result.get("title", "No title"),
                "content": self._clean_content(result.get("content", "")),
                "url": result.get("url", ""),
                "published_date": result.get("published_date", ""),
                "score": result.get("score", 0.0),
                "relevance_score": self._calculate_relevance(result, original_query),
                "content_quality": self._assess_content_quality(result.get("content", "")),
                "source_type": self._classify_source(result.get("url", "")),
                "snippet": self._create_snippet(result.get("content", ""), original_query)
            }
            processed.append(processed_result)

        # Sort by relevance score
        processed.sort(key=lambda x: x["relevance_score"], reverse=True)

        return processed

    def _enhance_search_query(self, query: str, context: str) -> str:
        """Enhance search query with context for better results"""
        if not context:
            return query

        # Extract key terms from context
        key_terms = self._extract_key_terms(context)

        # Combine with original query
        if key_terms:
            enhanced = f"{query} {' '.join(key_terms[:3])}"
            return enhanced

        return query

    def _calculate_relevance(self, result: Dict[str, Any], query: str) -> float:
        """Calculate relevance score for search result"""
        title = result.get("title", "").lower()
        content = result.get("content", "").lower()
        query_lower = query.lower()

        # Query terms in title (high weight)
        title_score = sum(1 for term in query_lower.split() if term in title) * 0.3

        # Query terms in content (medium weight)
        content_score = sum(1 for term in query_lower.split() if term in content) * 0.2

        # Original score from API
        api_score = result.get("score", 0.0) * 0.5

        # Combine scores
        total_score = min(title_score + content_score + api_score, 1.0)

        return total_score

    def _assess_content_quality(self, content: str) -> float:
        """Assess content quality based on various factors"""
        if not content:
            return 0.0

        # Length score (optimal around 200-800 chars)
        length = len(content)
        if 200 <= length <= 800:
            length_score = 1.0
        elif length < 200:
            length_score = length / 200
        else:
            length_score = max(0.5, 1000 / length)

        # Sentence structure score
        sentences = content.count('.') + content.count('!') + content.count('?')
        sentence_score = min(sentences / 5, 1.0)

        # Information density (avoid repetitive content)
        words = content.split()
        unique_words = len(set(words))
        density_score = unique_words / len(words) if words else 0

        # Combine scores
        quality = (length_score * 0.4 + sentence_score * 0.3 + density_score * 0.3)

        return min(quality, 1.0)

    def _classify_source(self, url: str) -> str:
        """Classify source type based on URL"""
        url_lower = url.lower()

        if any(domain in url_lower for domain in ["wikipedia.org", "britannica.com"]):
            return "encyclopedia"
        elif any(domain in url_lower for domain in ["arxiv.org", "pubmed", "scholar.google"]):
            return "academic"
        elif any(domain in url_lower for domain in ["news", "reuters", "bbc", "cnn"]):
            return "news"
        elif any(domain in url_lower for domain in ["github.com", "stackoverflow"]):
            return "technical"
        elif any(domain in url_lower for domain in [".gov", ".edu"]):
            return "official"
        else:
            return "general"

    def _create_snippet(self, content: str, query: str) -> str:
        """Create relevant snippet from content"""
        if not content or len(content) <= 150:
            return content

        # Find best matching section
        query_terms = query.lower().split()
        sentences = content.split('.')

        best_sentence = ""
        best_score = 0

        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = sum(1 for term in query_terms if term in sentence_lower)

            if score > best_score and len(sentence.strip()) > 50:
                best_score = score
                best_sentence = sentence.strip()

        if best_sentence:
            return best_sentence[:150] + "..." if len(best_sentence) > 150 else best_sentence

        # Fallback to first 150 chars
        return content[:150] + "..."

    def _score_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Add comprehensive scoring to results"""
        for result in results:
            # Combine multiple factors
            relevance = result.get("relevance_score", 0.0)
            quality = result.get("content_quality", 0.0)
            api_score = result.get("score", 0.0)

            # Source type bonus
            source_bonus = {
                "academic": 0.1,
                "official": 0.1,
                "encyclopedia": 0.05,
                "technical": 0.05,
                "news": 0.02,
                "general": 0.0
            }.get(result.get("source_type", "general"), 0.0)

            # Final composite score
            result["composite_score"] = min(
                relevance * 0.4 + quality * 0.3 + api_score * 0.2 + source_bonus, 1.0
            )

        # Sort by composite score
        results.sort(key=lambda x: x.get("composite_score", 0.0), reverse=True)

        return results

    def _generate_search_summary(self, results: List[Dict[str, Any]], query: str) -> str:
        """Generate summary of search results"""
        if not results:
            return f"No relevant results found for: {query}"

        total_results = len(results)
        avg_score = sum(r.get("composite_score", 0.0) for r in results) / total_results
        source_types = set(r.get("source_type", "general") for r in results)

        summary = f"""Search completed for: "{query}"

**Results Overview:**
- Found {total_results} relevant sources
- Average relevance: {avg_score:.1%}
- Source types: {', '.join(source_types)}
- Top result: {results[0].get('title', 'N/A')} ({results[0].get('composite_score', 0.0):.1%} relevance)"""

        return summary

    def _assess_search_quality(self, results: List[Dict[str, Any]]) -> str:
        """Assess overall search quality"""
        if not results:
            return "poor"

        avg_score = sum(r.get("composite_score", 0.0) for r in results) / len(results)

        if avg_score >= 0.8:
            return "excellent"
        elif avg_score >= 0.6:
            return "good"
        elif avg_score >= 0.4:
            return "fair"
        else:
            return "poor"

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text for query enhancement"""
        # Simple keyword extraction (can be enhanced with NLP)
        words = text.lower().split()

        # Filter out common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}

        key_terms = [word for word in words if len(word) > 3 and word not in stop_words]

        # Return most frequent terms
        from collections import Counter
        term_counts = Counter(key_terms)

        return [term for term, count in term_counts.most_common(5)]

    def _clean_content(self, content: str) -> str:
        """Clean and normalize content"""
        if not content:
            return ""

        # Remove extra whitespace
        cleaned = " ".join(content.split())

        # Remove HTML tags if present
        import re
        cleaned = re.sub(r'<[^>]+>', '', cleaned)

        # Normalize quotes
        cleaned = cleaned.replace('"', '"').replace('"', '"')

        return cleaned

    def _create_fallback_results(self, query: str, error: str) -> List[Dict[str, Any]]:
        """Create fallback results when search fails"""
        return [{
            "rank": 1,
            "title": f"Search Error: {query}",
            "content": f"Unable to perform web search due to: {error}. Please try again later or refine your query.",
            "url": "",
            "score": 0.0,
            "relevance_score": 0.0,
            "content_quality": 0.0,
            "source_type": "error",
            "snippet": f"Search temporarily unavailable for: {query}",
            "composite_score": 0.0
        }]

    async def _rate_limit(self):
        """Implement rate limiting for web searches"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            await asyncio.sleep(sleep_time)

        self.last_request_time = current_time

    def get_metrics(self) -> Dict[str, Any]:
        """Get search performance metrics"""
        return {
            **asdict(self.metrics),
            "max_results": self.max_results,
            "success_rate": f"{self.metrics.get_success_rate():.1f}%"
        }

# ==================== CLIENT FACTORY ====================

class LLMClientFactory:
    """Factory for creating and managing LLM clients"""

    @staticmethod
    def create_gemini_client() -> GeminiClient:
        return GeminiClient()

    @staticmethod
    def create_tavily_client() -> TavilyWebSearchClient:
        return TavilyWebSearchClient()

    @staticmethod
    def create_all_clients() -> Dict[str, Any]:
        return {
            "gemini": GeminiClient(),
            "tavily": TavilyWebSearchClient()
        }

    @staticmethod
    def test_client_connections() -> Dict[str, bool]:
        results = {}

        try:
            gemini = GeminiClient()
            results["gemini"] = True
            logger.info("✅ Gemini client connection successful")
        except Exception as e:
            results["gemini"] = False
            logger.error(f"❌ Gemini client failed: {str(e)}")

        try:
            tavily = TavilyWebSearchClient()
            results["tavily"] = True
            logger.info("✅ Tavily client connection successful")
        except Exception as e:
            results["tavily"] = False
            logger.error(f"❌ Tavily client failed: {str(e)}")

        return results

# ==================== MAIN TESTING ====================

async def main():
    """Test LLM clients functionality"""
    print("🧪 Testing LLM Clients")
    print(f"👤 User: TIRUMALAMANAV")
    print(f"📅 Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 60)

    # Test client creation
    print("🔧 Testing client initialization...")
    connections = LLMClientFactory.test_client_connections()

    for client_name, status in connections.items():
        status_emoji = "✅" if status else "❌"
        print(f"  {status_emoji} {client_name.title()} client: {'OK' if status else 'FAILED'}")

    if all(connections.values()):
        print("\n🚀 All clients initialized successfully!")
    else:
        print("\n⚠️ Some clients failed to initialize.")
        print("💡 Check your API keys and network connection.")

if __name__ == "__main__":
    asyncio.run(main())
