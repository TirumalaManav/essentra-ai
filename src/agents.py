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
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Union
from enum import Enum
import json
import traceback
from dataclasses import dataclass, field

# Pydantic for data validation
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== MCP PROTOCOL IMPLEMENTATION ====================

class MCPMessageType(Enum):
    """Model Context Protocol message types"""
    # Document processing
    DOCUMENT_UPLOAD_REQUEST = "DOCUMENT_UPLOAD_REQUEST"
    DOCUMENT_PROCESSING_STATUS = "DOCUMENT_PROCESSING_STATUS"
    DOCUMENT_PROCESSING_COMPLETE = "DOCUMENT_PROCESSING_COMPLETE"

    # Query processing
    QUERY_RECEIVED = "QUERY_RECEIVED"
    QUERY_ANALYSIS_REQUEST = "QUERY_ANALYSIS_REQUEST"
    QUERY_ANALYSIS_RESPONSE = "QUERY_ANALYSIS_RESPONSE"

    # Retrieval operations
    RETRIEVAL_REQUEST = "RETRIEVAL_REQUEST"
    RETRIEVAL_RESPONSE = "RETRIEVAL_RESPONSE"

    # Web search operations
    WEB_SEARCH_REQUEST = "WEB_SEARCH_REQUEST"
    WEB_SEARCH_RESPONSE = "WEB_SEARCH_RESPONSE"

    # LLM operations
    LLM_GENERATION_REQUEST = "LLM_GENERATION_REQUEST"
    LLM_GENERATION_RESPONSE = "LLM_GENERATION_RESPONSE"

    # Coordination
    AGENT_HANDOFF = "AGENT_HANDOFF"
    WORKFLOW_COMPLETE = "WORKFLOW_COMPLETE"
    ERROR_OCCURRED = "ERROR_OCCURRED"

class MCPMessage(BaseModel):
    """Standard MCP message format"""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MCPMessageType
    sender_agent: str
    receiver_agent: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    payload: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=1, ge=1, le=5)  # 1=highest, 5=lowest

    @validator('timestamp', pre=True)
    def parse_timestamp(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v

class MCPResponse(BaseModel):
    """Standard MCP response format"""
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_message_id: str
    trace_id: str
    success: bool
    data: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    processing_time: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class MCPBus:
    """Message bus for agent communication"""

    def __init__(self):
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.agent_registry: Dict[str, 'BaseAgent'] = {}
        self.message_history: List[MCPMessage] = []
        self.active_traces: Dict[str, List[MCPMessage]] = {}
        self._bus_running = False

    def register_agent(self, agent: 'BaseAgent'):
        """Register an agent with the bus"""
        self.agent_registry[agent.agent_name] = agent
        logger.info(f"🤖 Registered agent: {agent.agent_name}")

    async def send_message(self, message: MCPMessage) -> MCPResponse:
        """Send message through the bus"""
        # Store message in history
        self.message_history.append(message)

        # Track by trace_id
        if message.trace_id not in self.active_traces:
            self.active_traces[message.trace_id] = []
        self.active_traces[message.trace_id].append(message)

        logger.info(f"📨 MCP Message: {message.sender_agent} -> {message.receiver_agent} | Type: {message.message_type.value}")

        # Route to target agent
        if message.receiver_agent in self.agent_registry:
            target_agent = self.agent_registry[message.receiver_agent]
            return await target_agent.handle_mcp_message(message)
        else:
            error_response = MCPResponse(
                original_message_id=message.message_id,
                trace_id=message.trace_id,
                success=False,
                error_message=f"Agent {message.receiver_agent} not found"
            )
            logger.error(f"❌ Agent {message.receiver_agent} not registered")
            return error_response

    def get_trace_history(self, trace_id: str) -> List[MCPMessage]:
        """Get all messages for a trace"""
        return self.active_traces.get(trace_id, [])

    def get_agent_stats(self) -> Dict[str, Any]:
        """Get bus statistics"""
        return {
            "registered_agents": list(self.agent_registry.keys()),
            "total_messages": len(self.message_history),
            "active_traces": len(self.active_traces),
            "bus_uptime": datetime.utcnow().isoformat()
        }

# ==================== BASE AGENT ARCHITECTURE ====================

class AgentStatus(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"

@dataclass
class AgentMetrics:
    """Agent performance metrics"""
    messages_processed: int = 0
    total_processing_time: float = 0.0
    successful_operations: int = 0
    failed_operations: int = 0
    average_response_time: float = 0.0
    last_activity: Optional[datetime] = None

class BaseAgent(ABC):
    """Abstract base class for all agents"""

    def __init__(self, agent_name: str, mcp_bus: MCPBus):
        self.agent_name = agent_name
        self.mcp_bus = mcp_bus
        self.status = AgentStatus.IDLE
        self.metrics = AgentMetrics()
        self.capabilities: List[str] = []
        self.config: Dict[str, Any] = {}

        # Register with MCP bus
        self.mcp_bus.register_agent(self)

        logger.info(f"🤖 Initialized {self.agent_name}")

    async def handle_mcp_message(self, message: MCPMessage) -> MCPResponse:
        """Handle incoming MCP message"""
        start_time = asyncio.get_event_loop().time()
        self.status = AgentStatus.PROCESSING
        self.metrics.messages_processed += 1
        self.metrics.last_activity = datetime.utcnow()

        try:
            # Process the message
            result = await self.process_message(message)

            # Update metrics
            processing_time = asyncio.get_event_loop().time() - start_time
            self.metrics.total_processing_time += processing_time
            self.metrics.successful_operations += 1
            self._update_average_response_time()

            self.status = AgentStatus.IDLE

            return MCPResponse(
                original_message_id=message.message_id,
                trace_id=message.trace_id,
                success=True,
                data=result,
                processing_time=processing_time
            )

        except Exception as e:
            # Handle errors
            processing_time = asyncio.get_event_loop().time() - start_time
            self.metrics.failed_operations += 1
            self.status = AgentStatus.ERROR

            error_msg = f"Error in {self.agent_name}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.error(traceback.format_exc())

            return MCPResponse(
                original_message_id=message.message_id,
                trace_id=message.trace_id,
                success=False,
                error_message=error_msg,
                processing_time=processing_time
            )

    @abstractmethod
    async def process_message(self, message: MCPMessage) -> Dict[str, Any]:
        """Process specific message types - implemented by subclasses"""
        pass

    def _update_average_response_time(self):
        """Update average response time metric"""
        if self.metrics.successful_operations > 0:
            self.metrics.average_response_time = (
                self.metrics.total_processing_time / self.metrics.successful_operations
            )

    async def send_message(self, message_type: MCPMessageType, receiver: str,
                          payload: Dict[str, Any], trace_id: str = None) -> MCPResponse:
        """Send message to another agent"""
        message = MCPMessage(
            message_type=message_type,
            sender_agent=self.agent_name,
            receiver_agent=receiver,
            trace_id=trace_id or str(uuid.uuid4()),
            payload=payload
        )
        return await self.mcp_bus.send_message(message)

    def get_status(self) -> Dict[str, Any]:
        """Get agent status and metrics"""
        return {
            "agent_name": self.agent_name,
            "status": self.status.value,
            "capabilities": self.capabilities,
            "metrics": {
                "messages_processed": self.metrics.messages_processed,
                "successful_operations": self.metrics.successful_operations,
                "failed_operations": self.metrics.failed_operations,
                "average_response_time": round(self.metrics.average_response_time, 3),
                "total_processing_time": round(self.metrics.total_processing_time, 3),
                "last_activity": self.metrics.last_activity.isoformat() if self.metrics.last_activity else None
            }
        }

# ==================== SPECIALIZED AGENTS ====================

class CoordinatorAgent(BaseAgent):
    """Main orchestrator agent that routes requests and manages workflow"""

    def __init__(self, mcp_bus: MCPBus):
        super().__init__("CoordinatorAgent", mcp_bus)
        self.capabilities = [
            "workflow_orchestration",
            "request_routing",
            "decision_making",
            "error_handling"
        ]
        self.workflow_state: Dict[str, Any] = {}

    async def process_message(self, message: MCPMessage) -> Dict[str, Any]:
        """Route and coordinate between other agents"""

        if message.message_type == MCPMessageType.QUERY_RECEIVED:
            return await self._handle_query_routing(message)

        elif message.message_type == MCPMessageType.DOCUMENT_UPLOAD_REQUEST:
            return await self._handle_document_routing(message)

        elif message.message_type == MCPMessageType.AGENT_HANDOFF:
            return await self._handle_agent_handoff(message)

        else:
            # ✅ FIXED ERROR MESSAGE FORMAT TO MATCH TEST EXPECTATION
            error_msg = f"Unsupported message type: {message.message_type}"
            logger.warning(f"⚠️ {error_msg}")
            return {"error": error_msg}

    async def _handle_query_routing(self, message: MCPMessage) -> Dict[str, Any]:
        """Route user queries to appropriate agents"""
        query = message.payload.get("query", "")
        session_id = message.payload.get("session_id", "")

        logger.info(f"🎯 Routing query: '{query[:50]}...'")

        # Analyze query intent
        query_analysis = await self.send_message(
            MCPMessageType.QUERY_ANALYSIS_REQUEST,
            "RetrievalAgent",
            {
                "query": query,
                "session_id": session_id,
                "analysis_type": "intent_detection"
            },
            message.trace_id
        )

        if not query_analysis.success:
            return {"error": "Failed to analyze query intent"}

        # Determine next steps based on analysis
        intent = query_analysis.data.get("intent", "general")

        if intent == "web_search_needed":
            # Route to web search agent
            web_response = await self.send_message(
                MCPMessageType.WEB_SEARCH_REQUEST,
                "WebSearchAgent",
                {"query": query, "session_id": session_id},
                message.trace_id
            )
            return {"routing_decision": "web_search", "result": web_response.data}

        else:
            # Route to retrieval agent for document search
            retrieval_response = await self.send_message(
                MCPMessageType.RETRIEVAL_REQUEST,
                "RetrievalAgent",
                {"query": query, "session_id": session_id},
                message.trace_id
            )
            return {"routing_decision": "document_retrieval", "result": retrieval_response.data}

    async def _handle_document_routing(self, message: MCPMessage) -> Dict[str, Any]:
        """Route document processing requests"""
        file_path = message.payload.get("file_path", "")
        session_id = message.payload.get("session_id", "")

        logger.info(f"📄 Routing document processing: {file_path}")

        # Send to ingestion agent
        ingestion_response = await self.send_message(
            MCPMessageType.DOCUMENT_PROCESSING_STATUS,
            "IngestionAgent",
            {
                "file_path": file_path,
                "session_id": session_id,
                "processing_options": message.payload.get("options", {})
            },
            message.trace_id
        )

        return {"routing_decision": "document_processing", "result": ingestion_response.data}

    async def _handle_agent_handoff(self, message: MCPMessage) -> Dict[str, Any]:
        """Handle handoffs between agents"""
        target_agent = message.payload.get("target_agent")
        handoff_data = message.payload.get("data", {})

        logger.info(f"🔄 Agent handoff to: {target_agent}")

        # Forward to target agent
        response = await self.send_message(
            MCPMessageType.QUERY_ANALYSIS_REQUEST,  # Generic processing
            target_agent,
            handoff_data,
            message.trace_id
        )

        return {"handoff_result": response.data}

class IngestionAgent(BaseAgent):
    """Handles document upload and processing operations"""

    def __init__(self, mcp_bus: MCPBus):
        super().__init__("IngestionAgent", mcp_bus)
        self.capabilities = [
            "document_parsing",
            "multi_format_support",
            "chunk_creation",
            "vector_storage"
        ]
        self.document_processor = None  # Will be injected

    def set_document_processor(self, processor):
        """Inject document processor dependency"""
        self.document_processor = processor
        logger.info("📄 Document processor injected into IngestionAgent")

    async def process_message(self, message: MCPMessage) -> Dict[str, Any]:
        """Process document ingestion requests"""

        if not self.document_processor:
            return {"error": "Document processor not configured"}

        if message.message_type == MCPMessageType.DOCUMENT_PROCESSING_STATUS:
            return await self._process_document(message)

        else:
            return {"error": f"Unsupported message type: {message.message_type}"}

    async def _process_document(self, message: MCPMessage) -> Dict[str, Any]:
        """Process uploaded document"""
        file_path = message.payload.get("file_path")
        session_id = message.payload.get("session_id")
        options = message.payload.get("processing_options", {})

        logger.info(f"📄 Processing document: {file_path}")

        try:
            # Process document using injected processor
            result = await self.document_processor.process_document(
                file_path,
                session_id,
                chunk_size=options.get("chunk_size", 500),
                chunk_overlap=options.get("chunk_overlap", 50)
            )

            if result.success:
                # Notify coordinator of completion
                await self.send_message(
                    MCPMessageType.DOCUMENT_PROCESSING_COMPLETE,
                    "CoordinatorAgent",
                    {
                        "file_id": result.file_id,
                        "chunks_created": result.chunks_created,
                        "processing_time": result.processing_time,
                        "session_id": session_id
                    },
                    message.trace_id
                )

                return {
                    "success": True,
                    "file_id": result.file_id,
                    "chunks_created": result.chunks_created,
                    "processing_time": result.processing_time,
                    "metadata": result.metadata.dict()
                }
            else:
                return {
                    "success": False,
                    "error": result.error_message
                }

        except Exception as e:
            logger.error(f"❌ Document processing failed: {str(e)}")
            return {
                "success": False,
                "error": f"Processing failed: {str(e)}"
            }

class RetrievalAgent(BaseAgent):
    """Handles semantic search and document retrieval"""

    def __init__(self, mcp_bus: MCPBus):
        super().__init__("RetrievalAgent", mcp_bus)
        self.capabilities = [
            "semantic_search",
            "query_analysis",
            "context_ranking",
            "hybrid_retrieval"
        ]
        self.document_processor = None  # Will be injected

    def set_document_processor(self, processor):
        """Inject document processor dependency"""
        self.document_processor = processor
        logger.info("🔍 Document processor injected into RetrievalAgent")

    async def process_message(self, message: MCPMessage) -> Dict[str, Any]:
        """Process retrieval requests"""

        if message.message_type == MCPMessageType.RETRIEVAL_REQUEST:
            return await self._perform_retrieval(message)

        elif message.message_type == MCPMessageType.QUERY_ANALYSIS_REQUEST:
            return await self._analyze_query(message)

        else:
            return {"error": f"Unsupported message type: {message.message_type}"}

    async def _perform_retrieval(self, message: MCPMessage) -> Dict[str, Any]:
        """Perform semantic retrieval with enhanced error handling"""
        query = message.payload.get("query")
        session_id = message.payload.get("session_id")
        n_results = message.payload.get("n_results", 5)

        # ✅ ADD PROPER NULL CHECKS
        if not query:
            logger.warning("⚠️ Retrieval attempted without query")
            return {"error": "Query is required for retrieval"}

        if not session_id:
            logger.warning("⚠️ Retrieval attempted without session_id")
            return {"error": "Session ID is required for retrieval"}

        # ✅ SAFE LOGGING WITH NULL CHECK
        query_preview = query[:50] + "..." if len(query) > 50 else query
        logger.info(f"🔍 Performing retrieval for: '{query_preview}'")

        try:
            if not self.document_processor:
                return {"error": "Document processor not configured"}

            # Retrieve similar chunks
            results = await self.document_processor.retrieve_similar_chunks(
                query, session_id, n_results
            )

            # ✅ CHECK IF RESULTS ARE VALID
            if not results:
                logger.info(f"📭 No results found for query: '{query_preview}'")
                return {
                    "retrieval_results": [],
                    "llm_response": {
                        "response": "I couldn't find any relevant information in the uploaded documents for your query.",
                        "sources_used": [],
                        "context_count": 0,
                        "session_id": session_id
                    }
                }

            # Format results for LLM consumption
            formatted_contexts = []
            for i, result in enumerate(results):
                # ✅ SAFE ACCESS TO RESULT DATA
                try:
                    formatted_contexts.append({
                        "rank": i + 1,
                        "content": result.get("content", ""),
                        "source": result.get("metadata", {}).get("source_file", "unknown"),
                        "similarity_score": result.get("similarity_score", 0.0),
                        "chunk_index": result.get("metadata", {}).get("chunk_index", 0)
                    })
                except Exception as format_error:
                    logger.warning(f"⚠️ Error formatting result {i}: {str(format_error)}")
                    continue

            # ✅ CHECK IF WE HAVE VALID FORMATTED CONTEXTS
            if not formatted_contexts:
                logger.warning("⚠️ No valid contexts after formatting")
                return {
                    "retrieval_results": [],
                    "llm_response": {
                        "response": "I found some information but encountered issues processing it. Please try rephrasing your query.",
                        "sources_used": [],
                        "context_count": 0,
                        "session_id": session_id
                    }
                }

            # Send to LLM agent for response generation
            try:
                llm_response = await self.send_message(
                    MCPMessageType.LLM_GENERATION_REQUEST,
                    "LLMResponseAgent",
                    {
                        "query": query,
                        "contexts": formatted_contexts,
                        "session_id": session_id
                    },
                    message.trace_id
                )

                # ✅ CHECK LLM RESPONSE SUCCESS
                if llm_response.success:
                    return {
                        "retrieval_results": formatted_contexts,
                        "llm_response": llm_response.data
                    }
                else:
                    logger.warning(f"⚠️ LLM response failed: {llm_response.error_message}")
                    return {
                        "retrieval_results": formatted_contexts,
                        "llm_response": {
                            "response": "I found relevant information but encountered an issue generating the response. Here are the relevant sources I found.",
                            "sources_used": [ctx["source"] for ctx in formatted_contexts[:3]],
                            "context_count": len(formatted_contexts),
                            "session_id": session_id,
                            "error": "LLM response generation failed"
                        }
                    }

            except Exception as llm_error:
                logger.error(f"❌ LLM communication failed: {str(llm_error)}")
                return {
                    "retrieval_results": formatted_contexts,
                    "llm_response": {
                        "response": "I found relevant information in the documents but cannot generate a response right now. Please check the sources below.",
                        "sources_used": [ctx["source"] for ctx in formatted_contexts[:3]],
                        "context_count": len(formatted_contexts),
                        "session_id": session_id,
                        "error": f"LLM communication error: {str(llm_error)}"
                    }
                }

        except Exception as e:
            error_msg = f"Retrieval failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {
                "error": error_msg,
                "retrieval_results": [],
                "llm_response": {
                    "response": "I encountered an error while searching the documents. Please try again or rephrase your query.",
                    "sources_used": [],
                    "context_count": 0,
                    "session_id": session_id or "unknown",
                    "error": error_msg
                }
            }

    async def _analyze_query(self, message: MCPMessage) -> Dict[str, Any]:
        """Analyze query intent and characteristics"""
        query = message.payload.get("query", "")
        analysis_type = message.payload.get("analysis_type", "general")

        # Simple intent detection (can be enhanced with ML models)
        web_search_indicators = [
            "latest", "recent", "current", "news", "today",
            "what's happening", "breaking", "update"
        ]

        query_lower = query.lower()
        needs_web_search = any(indicator in query_lower for indicator in web_search_indicators)

        return {
            "intent": "web_search_needed" if needs_web_search else "document_search",
            "query_length": len(query),
            "query_type": analysis_type,
            "confidence": 0.8 if needs_web_search else 0.9
        }

class LLMResponseAgent(BaseAgent):
    """Handles LLM interactions for response generation"""

    def __init__(self, mcp_bus: MCPBus):
        super().__init__("LLMResponseAgent", mcp_bus)
        self.capabilities = [
            "response_generation",
            "context_synthesis",
            "prompt_engineering",
            "answer_formatting"
        ]
        self.llm_client = None  # Will be injected

    def set_llm_client(self, client):
        """Inject LLM client dependency"""
        self.llm_client = client
        logger.info("🧠 LLM client injected into LLMResponseAgent")

    async def process_message(self, message: MCPMessage) -> Dict[str, Any]:
        """Process LLM generation requests"""

        if message.message_type == MCPMessageType.LLM_GENERATION_REQUEST:
            return await self._generate_response(message)

        else:
            return {"error": f"Unsupported message type: {message.message_type}"}

    async def _generate_response(self, message: MCPMessage) -> Dict[str, Any]:
        """Generate response using LLM"""
        query = message.payload.get("query")
        contexts = message.payload.get("contexts", [])
        session_id = message.payload.get("session_id")

        logger.info(f"🧠 Generating LLM response for query: '{query[:50]}...'")

        try:
            if not self.llm_client:
                return {"error": "LLM client not configured"}

            # Format context for LLM
            context_text = "\n\n".join([
                f"Source {ctx['rank']} ({ctx['source']}):\n{ctx['content']}"
                for ctx in contexts[:3]  # Use top 3 results
            ])

            # Create prompt
            prompt = f"""Based on the following context information, please provide a comprehensive answer to the user's question.

Context:
{context_text}

Question: {query}

Please provide a detailed, accurate answer based on the context provided. If the context doesn't contain enough information to fully answer the question, please say so and provide what information is available.

Answer:"""

            # Generate response using LLM client
            response = await self.llm_client.generate_response(prompt)

            return {
                "response": response,
                "sources_used": [ctx["source"] for ctx in contexts[:3]],
                "context_count": len(contexts),
                "session_id": session_id
            }

        except Exception as e:
            logger.error(f"❌ LLM response generation failed: {str(e)}")
            return {
                "error": f"Response generation failed: {str(e)}",
                "fallback_response": "I apologize, but I'm unable to generate a response at the moment. Please try again."
            }

class WebSearchAgent(BaseAgent):
    """Handles web search for real-time information"""

    def __init__(self, mcp_bus: MCPBus):
        super().__init__("WebSearchAgent", mcp_bus)
        self.capabilities = [
            "web_search",
            "real_time_information",
            "search_result_processing",
            "external_knowledge"
        ]
        self.web_search_client = None  # Will be injected

    def set_web_search_client(self, client):
        """Inject web search client dependency"""
        self.web_search_client = client
        logger.info("🌐 Web search client injected into WebSearchAgent")

    async def process_message(self, message: MCPMessage) -> Dict[str, Any]:
        """Process web search requests"""

        if message.message_type == MCPMessageType.WEB_SEARCH_REQUEST:
            return await self._perform_web_search(message)

        else:
            return {"error": f"Unsupported message type: {message.message_type}"}

    async def _perform_web_search(self, message: MCPMessage) -> Dict[str, Any]:
        """Perform web search and process results"""
        query = message.payload.get("query")
        session_id = message.payload.get("session_id")
        max_results = message.payload.get("max_results", 5)

        logger.info(f"🌐 Performing web search for: '{query[:50]}...'")

        try:
            if not self.web_search_client:
                return {"error": "Web search client not configured"}

            # Perform web search
            search_results = await self.web_search_client.search(query, max_results)

            # Process and format results
            formatted_results = []
            for result in search_results:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "url": result.get("url", ""),
                    "relevance_score": result.get("score", 0.0)
                })

            # Send to LLM for response generation
            llm_response = await self.send_message(
                MCPMessageType.LLM_GENERATION_REQUEST,
                "LLMResponseAgent",
                {
                    "query": query,
                    "web_results": formatted_results,
                    "session_id": session_id
                },
                message.trace_id
            )

            return {
                "search_results": formatted_results,
                "llm_response": llm_response.data,
                "source_type": "web_search"
            }

        except Exception as e:
            logger.error(f"❌ Web search failed: {str(e)}")
            return {
                "error": f"Web search failed: {str(e)}",
                "fallback_results": []
            }

# ==================== AGENT MANAGER ====================

class AgentManager:
    """Manages all agents and their interactions"""

    def __init__(self):
        self.mcp_bus = MCPBus()
        self.agents: Dict[str, BaseAgent] = {}
        self.initialized = False

        logger.info("🎯 AgentManager initialized")

    def initialize_agents(self) -> Dict[str, BaseAgent]:
        """Initialize all agents"""
        if self.initialized:
            return self.agents

        # Create all agents
        self.agents = {
            "CoordinatorAgent": CoordinatorAgent(self.mcp_bus),
            "IngestionAgent": IngestionAgent(self.mcp_bus),
            "RetrievalAgent": RetrievalAgent(self.mcp_bus),
            "LLMResponseAgent": LLMResponseAgent(self.mcp_bus),
            "WebSearchAgent": WebSearchAgent(self.mcp_bus)
        }

        self.initialized = True
        logger.info("✅ All agents initialized successfully")

        return self.agents

    def inject_dependencies(self, document_processor=None, llm_client=None, web_search_client=None):
        """Inject external dependencies into agents"""
        if document_processor:
            if "IngestionAgent" in self.agents:
                self.agents["IngestionAgent"].set_document_processor(document_processor)
            if "RetrievalAgent" in self.agents:
                self.agents["RetrievalAgent"].set_document_processor(document_processor)

        if llm_client:
            if "LLMResponseAgent" in self.agents:
                self.agents["LLMResponseAgent"].set_llm_client(llm_client)

        if web_search_client:
            if "WebSearchAgent" in self.agents:
                self.agents["WebSearchAgent"].set_web_search_client(web_search_client)

        logger.info("🔧 Dependencies injected into agents")

    async def process_user_query(self, query: str, session_id: str) -> Dict[str, Any]:
        """Process user query through the agent system"""
        if not self.initialized:
            raise RuntimeError("Agents not initialized. Call initialize_agents() first.")

        trace_id = str(uuid.uuid4())

        # Send initial query to coordinator
        message = MCPMessage(
            message_type=MCPMessageType.QUERY_RECEIVED,
            sender_agent="UserInterface",
            receiver_agent="CoordinatorAgent",
            trace_id=trace_id,
            payload={
                "query": query,
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

        response = await self.mcp_bus.send_message(message)

        return {
            "response": response.data,
            "trace_id": trace_id,
            "processing_time": response.processing_time,
            "success": response.success
        }

    async def process_document_upload(self, file_path: str, session_id: str,
                                    options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process document upload through the agent system"""
        if not self.initialized:
            raise RuntimeError("Agents not initialized. Call initialize_agents() first.")

        trace_id = str(uuid.uuid4())

        # Send document upload request to coordinator
        message = MCPMessage(
            message_type=MCPMessageType.DOCUMENT_UPLOAD_REQUEST,
            sender_agent="UserInterface",
            receiver_agent="CoordinatorAgent",
            trace_id=trace_id,
            payload={
                "file_path": file_path,
                "session_id": session_id,
                "options": options or {},
                "timestamp": datetime.utcnow().isoformat()
            }
        )

        response = await self.mcp_bus.send_message(message)

        return {
            "response": response.data,
            "trace_id": trace_id,
            "processing_time": response.processing_time,
            "success": response.success
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        agent_statuses = {}
        for name, agent in self.agents.items():
            agent_statuses[name] = agent.get_status()

        return {
            "system_initialized": self.initialized,
            "total_agents": len(self.agents),
            "mcp_bus_stats": self.mcp_bus.get_agent_stats(),
            "agents": agent_statuses,
            "timestamp": datetime.utcnow().isoformat()
        }

# ==================== TESTING FUNCTIONALITY ====================

async def test_agent_system():
    """Test the complete agent system"""
    print("🧪 Testing Agent System")
    print("=" * 50)

    # Initialize manager
    manager = AgentManager()
    agents = manager.initialize_agents()

    print(f"✅ Initialized {len(agents)} agents")

    # Test MCP communication
    print("\n📨 Testing MCP Communication...")

    # Create test message
    test_message = MCPMessage(
        message_type=MCPMessageType.QUERY_ANALYSIS_REQUEST,
        sender_agent="TestSender",
        receiver_agent="RetrievalAgent",
        payload={"query": "test query", "analysis_type": "intent_detection"}
    )

    # Send through coordinator
    coordinator = agents["CoordinatorAgent"]
    response = await coordinator.handle_mcp_message(test_message)

    print(f"📨 MCP Response: {response.success}")
    print(f"⏱️ Processing time: {response.processing_time:.3f}s")

    # Test system status
    print("\n📊 System Status:")
    status = manager.get_system_status()
    for agent_name, agent_status in status["agents"].items():
        print(f"  🤖 {agent_name}: {agent_status['status']}")
        print(f"      Messages processed: {agent_status['metrics']['messages_processed']}")

    print("\n✅ Agent system test completed!")

if __name__ == "__main__":
    # Run agent system test
    asyncio.run(test_agent_system())
