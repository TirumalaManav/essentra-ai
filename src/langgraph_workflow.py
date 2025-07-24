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
from datetime import datetime
from typing import Dict, Any, List, Optional, TypedDict, Annotated, Sequence
from enum import Enum
import json
from dataclasses import dataclass, asdict

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Our imports
from agents import AgentManager, MCPMessageType
from memory import ConversationMemory, ConversationTurn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== STATE MANAGEMENT ====================

class WorkflowStage(Enum):
    """Workflow execution stages"""
    INITIALIZE = "initialize"
    INTENT_ANALYSIS = "intent_analysis"
    DOCUMENT_UPLOAD = "document_upload"
    DOCUMENT_RETRIEVAL = "document_retrieval"
    WEB_SEARCH = "web_search"
    LLM_GENERATION = "llm_generation"
    RESPONSE_FORMATTING = "response_formatting"
    MEMORY_UPDATE = "memory_update"
    COMPLETE = "complete"
    ERROR_HANDLING = "error_handling"

class WorkflowState(TypedDict):
    """LangGraph state definition with comprehensive tracking"""
    # Core request data
    messages: Annotated[Sequence[Dict[str, Any]], add_messages]
    user_query: str
    session_id: str

    # Workflow control
    current_stage: WorkflowStage
    trace_id: str
    workflow_start_time: datetime

    # Intent and routing
    detected_intent: str
    routing_decision: str
    confidence_score: float

    # Document processing
    uploaded_files: List[Dict[str, Any]]
    processed_documents: List[Dict[str, Any]]

    # Retrieval results
    retrieval_results: List[Dict[str, Any]]
    context_chunks: List[Dict[str, Any]]

    # Web search results
    web_search_results: List[Dict[str, Any]]
    external_sources: List[Dict[str, Any]]

    # LLM processing
    llm_prompt: str
    llm_response: str
    response_metadata: Dict[str, Any]

    # Memory and context
    conversation_history: List[Dict[str, Any]]
    previous_context: str
    memory_updated: bool

    # Performance metrics
    total_processing_time: float
    stage_timings: Dict[str, float]
    agent_interactions: List[Dict[str, Any]]

    # Error handling
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]

    # Final output
    final_response: str
    sources_used: List[str]
    confidence_rating: float

# ==================== WORKFLOW NODES ====================

class LangGraphWorkflow:
    """Advanced LangGraph workflow orchestrator"""

    def __init__(self, agent_manager: AgentManager,
                 document_processor=None, llm_client=None, web_search_client=None):
        self.agent_manager = agent_manager
        self.memory = ConversationMemory()

        # Initialize agents if not already done
        if not agent_manager.initialized:
            agent_manager.initialize_agents()

        # Inject dependencies
        agent_manager.inject_dependencies(
            document_processor=document_processor,
            llm_client=llm_client,
            web_search_client=web_search_client
        )

        # Create workflow graph
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile(checkpointer=MemorySaver())

        logger.info("🔄 LangGraph workflow initialized successfully")

    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow with all nodes and edges"""
        workflow = StateGraph(WorkflowState)

        # Add all workflow nodes
        workflow.add_node("initialize", self._initialize_node)
        workflow.add_node("intent_analysis", self._intent_analysis_node)
        workflow.add_node("document_upload", self._document_upload_node)
        workflow.add_node("document_retrieval", self._document_retrieval_node)
        workflow.add_node("web_search", self._web_search_node)
        workflow.add_node("llm_generation", self._llm_generation_node)
        workflow.add_node("response_formatting", self._response_formatting_node)
        workflow.add_node("memory_update", self._memory_update_node)
        workflow.add_node("error_handling", self._error_handling_node)

        # Define workflow edges and routing
        workflow.set_entry_point("initialize")

        # Initialize -> Intent Analysis
        workflow.add_edge("initialize", "intent_analysis")

        # Intent Analysis -> Conditional routing
        workflow.add_conditional_edges(
            "intent_analysis",
            self._route_after_intent_analysis,
            {
                "document_upload": "document_upload",
                "document_retrieval": "document_retrieval",
                "web_search": "web_search",
                "error": "error_handling"
            }
        )

        # Document Upload -> Document Retrieval
        workflow.add_edge("document_upload", "document_retrieval")

        # Document Retrieval -> LLM Generation
        workflow.add_edge("document_retrieval", "llm_generation")

        # Web Search -> LLM Generation
        workflow.add_edge("web_search", "llm_generation")

        # LLM Generation -> Response Formatting
        workflow.add_edge("llm_generation", "response_formatting")

        # Response Formatting -> Memory Update
        workflow.add_edge("response_formatting", "memory_update")

        # Memory Update -> END
        workflow.add_edge("memory_update", END)

        # Error Handling -> END
        workflow.add_edge("error_handling", END)

        return workflow

    async def _initialize_node(self, state: WorkflowState) -> WorkflowState:
        """Initialize workflow state and setup"""
        start_time = datetime.utcnow()

        logger.info(f"🚀 Initializing workflow for user: TIRUMALAMANAV")
        logger.info(f"📝 Query: '{state['user_query'][:50]}...'")

        # Get conversation history from memory
        previous_context = self.memory.get_recent_context(state["session_id"], turns=3)

        # Update state
        state.update({
            "current_stage": WorkflowStage.INITIALIZE,
            "trace_id": str(uuid.uuid4()),
            "workflow_start_time": start_time,
            "conversation_history": [],
            "previous_context": previous_context,
            "stage_timings": {"initialize": 0.0},
            "agent_interactions": [],
            "errors": [],
            "warnings": [],
            "uploaded_files": [],
            "processed_documents": [],
            "retrieval_results": [],
            "context_chunks": [],
            "web_search_results": [],
            "external_sources": [],
            "response_metadata": {},
            "memory_updated": False,
            "sources_used": [],
            "confidence_rating": 0.0
        })

        # Record timing
        end_time = datetime.utcnow()
        state["stage_timings"]["initialize"] = (end_time - start_time).total_seconds()

        logger.info(f"✅ Workflow initialized (Trace: {state['trace_id']})")
        return state

    async def _intent_analysis_node(self, state: WorkflowState) -> WorkflowState:
        """Analyze user intent and determine routing"""
        start_time = datetime.utcnow()
        state["current_stage"] = WorkflowStage.INTENT_ANALYSIS

        logger.info(f"🎯 Analyzing intent for query: '{state['user_query'][:50]}...'")

        try:
            # Use coordinator agent to analyze intent
            result = await self.agent_manager.process_user_query(
                state["user_query"],
                state["session_id"]
            )

            if result["success"]:
                response_data = result["response"]
                routing_decision = response_data.get("routing_decision", "document_retrieval")

                # Determine confidence based on routing decision
                if routing_decision == "web_search":
                    confidence = 0.85
                    intent = "web_search_needed"
                elif routing_decision == "document_retrieval":
                    confidence = 0.90
                    intent = "document_search"
                else:
                    confidence = 0.75
                    intent = "general_query"

                state.update({
                    "detected_intent": intent,
                    "routing_decision": routing_decision,
                    "confidence_score": confidence
                })

                # Record agent interaction
                state["agent_interactions"].append({
                    "agent": "CoordinatorAgent",
                    "action": "intent_analysis",
                    "timestamp": datetime.utcnow().isoformat(),
                    "success": True,
                    "trace_id": result["trace_id"]
                })

                logger.info(f"🎯 Intent detected: {intent} (confidence: {confidence:.2f})")
                logger.info(f"🔄 Routing decision: {routing_decision}")

            else:
                # Handle analysis failure
                state.update({
                    "detected_intent": "error",
                    "routing_decision": "error",
                    "confidence_score": 0.0
                })

                state["errors"].append({
                    "stage": "intent_analysis",
                    "error": "Failed to analyze user intent",
                    "timestamp": datetime.utcnow().isoformat()
                })

                logger.error(f"❌ Intent analysis failed")

        except Exception as e:
            state.update({
                "detected_intent": "error",
                "routing_decision": "error",
                "confidence_score": 0.0
            })

            state["errors"].append({
                "stage": "intent_analysis",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })

            logger.error(f"❌ Intent analysis error: {str(e)}")

        # Record timing
        end_time = datetime.utcnow()
        state["stage_timings"]["intent_analysis"] = (end_time - start_time).total_seconds()

        return state

    async def _document_upload_node(self, state: WorkflowState) -> WorkflowState:
        """Handle document upload and processing"""
        start_time = datetime.utcnow()
        state["current_stage"] = WorkflowStage.DOCUMENT_UPLOAD

        logger.info(f"📄 Processing document upload for session: {state['session_id']}")

        try:
            # Check if there are files to process (in real implementation)
            # For now, simulate document processing

            # This would be called when user uploads files
            # result = await self.agent_manager.process_document_upload(
            #     file_path, state["session_id"], options
            # )

            # Simulate successful document processing
            state["processed_documents"].append({
                "file_id": str(uuid.uuid4()),
                "filename": "user_document.pdf",
                "chunks_created": 25,
                "processing_time": 2.5,
                "status": "processed"
            })

            logger.info(f"✅ Document processing completed")

        except Exception as e:
            state["errors"].append({
                "stage": "document_upload",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })

            logger.error(f"❌ Document upload error: {str(e)}")

        # Record timing
        end_time = datetime.utcnow()
        state["stage_timings"]["document_upload"] = (end_time - start_time).total_seconds()

        return state

    async def _document_retrieval_node(self, state: WorkflowState) -> WorkflowState:
        """Perform document retrieval and context gathering"""
        start_time = datetime.utcnow()
        state["current_stage"] = WorkflowStage.DOCUMENT_RETRIEVAL

        logger.info(f"🔍 Performing document retrieval for: '{state['user_query'][:50]}...'")

        try:
            # Use retrieval agent through coordinator
            result = await self.agent_manager.process_user_query(
                state["user_query"],
                state["session_id"]
            )

            if result["success"]:
                response_data = result["response"]
                retrieval_data = response_data.get("result", {})

                # Extract retrieval results
                retrieval_results = retrieval_data.get("retrieval_results", [])
                llm_response_data = retrieval_data.get("llm_response", {})

                state.update({
                    "retrieval_results": retrieval_results,
                    "context_chunks": retrieval_results[:5],  # Top 5 chunks
                })

                # Store LLM response data for later use
                if llm_response_data:
                    state["response_metadata"].update(llm_response_data)

                # Record sources
                sources = [chunk.get("source", "unknown") for chunk in retrieval_results]
                state["sources_used"].extend(sources)

                logger.info(f"🔍 Retrieved {len(retrieval_results)} relevant chunks")
                logger.info(f"📚 Sources: {list(set(sources))}")

            else:
                state["warnings"].append({
                    "stage": "document_retrieval",
                    "warning": "No relevant documents found",
                    "timestamp": datetime.utcnow().isoformat()
                })

                logger.warning(f"⚠️ No relevant documents found")

        except Exception as e:
            state["errors"].append({
                "stage": "document_retrieval",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })

            logger.error(f"❌ Document retrieval error: {str(e)}")

        # Record timing
        end_time = datetime.utcnow()
        state["stage_timings"]["document_retrieval"] = (end_time - start_time).total_seconds()

        return state

    async def _web_search_node(self, state: WorkflowState) -> WorkflowState:
        """Perform web search for real-time information"""
        start_time = datetime.utcnow()
        state["current_stage"] = WorkflowStage.WEB_SEARCH

        logger.info(f"🌐 Performing web search for: '{state['user_query'][:50]}...'")

        try:
            # Get web search agent
            web_agent = self.agent_manager.agents.get("WebSearchAgent")

            if web_agent:
                # Create message for web search
                from agents import MCPMessage

                search_message = MCPMessage(
                    message_type=MCPMessageType.WEB_SEARCH_REQUEST,
                    sender_agent="WorkflowOrchestrator",
                    receiver_agent="WebSearchAgent",
                    trace_id=state["trace_id"],
                    payload={
                        "query": state["user_query"],
                        "session_id": state["session_id"],
                        "max_results": 5
                    }
                )

                response = await web_agent.handle_mcp_message(search_message)

                if response.success:
                    search_results = response.data.get("search_results", [])
                    llm_response_data = response.data.get("llm_response", {})

                    state.update({
                        "web_search_results": search_results,
                        "external_sources": [result.get("url", "") for result in search_results]
                    })

                    # Store LLM response data
                    if llm_response_data:
                        state["response_metadata"].update(llm_response_data)

                    # Record sources
                    sources = [result.get("title", "Web source") for result in search_results]
                    state["sources_used"].extend(sources)

                    logger.info(f"🌐 Found {len(search_results)} web results")

                else:
                    state["warnings"].append({
                        "stage": "web_search",
                        "warning": "Web search failed",
                        "timestamp": datetime.utcnow().isoformat()
                    })

                    logger.warning(f"⚠️ Web search failed")

            else:
                state["warnings"].append({
                    "stage": "web_search",
                    "warning": "Web search agent not available",
                    "timestamp": datetime.utcnow().isoformat()
                })

        except Exception as e:
            state["errors"].append({
                "stage": "web_search",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })

            logger.error(f"❌ Web search error: {str(e)}")

        # Record timing
        end_time = datetime.utcnow()
        state["stage_timings"]["web_search"] = (end_time - start_time).total_seconds()

        return state

    async def _llm_generation_node(self, state: WorkflowState) -> WorkflowState:
        """Generate final response using LLM"""
        start_time = datetime.utcnow()
        state["current_stage"] = WorkflowStage.LLM_GENERATION

        logger.info(f"🧠 Generating LLM response...")

        try:
            # Get existing LLM response from previous steps
            if "response" in state["response_metadata"]:
                llm_response = state["response_metadata"]["response"]

                state.update({
                    "llm_response": llm_response,
                    "confidence_rating": 0.85
                })

                logger.info(f"🧠 LLM response generated (length: {len(llm_response)})")

            else:
                # Fallback response
                fallback_response = f"""I understand you're asking about: "{state['user_query']}"

Based on the available information, I can provide some insights. However, for a more comprehensive answer, you might want to:

1. Upload relevant documents to get detailed information
2. Refine your query to be more specific
3. Check if your question relates to recent developments (which might require web search)

Please feel free to rephrase your question or provide additional context!"""

                state.update({
                    "llm_response": fallback_response,
                    "confidence_rating": 0.60
                })

                state["warnings"].append({
                    "stage": "llm_generation",
                    "warning": "Using fallback response",
                    "timestamp": datetime.utcnow().isoformat()
                })

                logger.warning(f"⚠️ Using fallback LLM response")

        except Exception as e:
            # Error fallback
            error_response = "I apologize, but I encountered an error while processing your request. Please try again or rephrase your question."

            state.update({
                "llm_response": error_response,
                "confidence_rating": 0.0
            })

            state["errors"].append({
                "stage": "llm_generation",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })

            logger.error(f"❌ LLM generation error: {str(e)}")

        # Record timing
        end_time = datetime.utcnow()
        state["stage_timings"]["llm_generation"] = (end_time - start_time).total_seconds()

        return state

    async def _response_formatting_node(self, state: WorkflowState) -> WorkflowState:
        """Format final response with metadata"""
        start_time = datetime.utcnow()
        state["current_stage"] = WorkflowStage.RESPONSE_FORMATTING

        logger.info(f"✨ Formatting final response...")

        try:
            # Create formatted response
            response_parts = []

            # Main response
            response_parts.append(state["llm_response"])

            # Add sources if available
            unique_sources = list(set(state["sources_used"]))
            if unique_sources:
                response_parts.append(f"\n\n📚 **Sources used:**")
                for i, source in enumerate(unique_sources[:5], 1):
                    response_parts.append(f"{i}. {source}")

            # Add confidence and metadata
            confidence = state["confidence_rating"]
            response_parts.append(f"\n\n🎯 **Confidence:** {confidence:.1%}")

            # Add processing info
            total_time = sum(state["stage_timings"].values())
            response_parts.append(f"⏱️ **Processing time:** {total_time:.2f}s")

            # Add session info
            response_parts.append(f"🆔 **Session:** {state['session_id']}")

            final_response = "\n".join(response_parts)

            state.update({
                "final_response": final_response,
                "total_processing_time": total_time
            })

            logger.info(f"✨ Response formatted (total time: {total_time:.2f}s)")

        except Exception as e:
            # Fallback to basic response
            state["final_response"] = state.get("llm_response", "Error generating response")

            state["errors"].append({
                "stage": "response_formatting",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })

            logger.error(f"❌ Response formatting error: {str(e)}")

        # Record timing
        end_time = datetime.utcnow()
        state["stage_timings"]["response_formatting"] = (end_time - start_time).total_seconds()

        return state

    async def _memory_update_node(self, state: WorkflowState) -> WorkflowState:
        """Update conversation memory"""
        start_time = datetime.utcnow()
        state["current_stage"] = WorkflowStage.MEMORY_UPDATE

        logger.info(f"🧠 Updating conversation memory...")

        try:
            # Create conversation turn
            turn = ConversationTurn(
                turn_id=str(uuid.uuid4()),
                user_message=state["user_query"],
                assistant_response=state["final_response"],
                timestamp=datetime.utcnow(),
                sources_used=state["sources_used"],
                retrieval_context=state["context_chunks"],
                session_id=state["session_id"]
            )

            # Add to memory
            self.memory.add_turn(state["session_id"], turn)

            state["memory_updated"] = True

            logger.info(f"🧠 Memory updated for session: {state['session_id']}")

        except Exception as e:
            state["errors"].append({
                "stage": "memory_update",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })

            logger.error(f"❌ Memory update error: {str(e)}")

        # Record final timing
        end_time = datetime.utcnow()
        state["stage_timings"]["memory_update"] = (end_time - start_time).total_seconds()

        # Calculate total processing time
        state["total_processing_time"] = sum(state["stage_timings"].values())

        logger.info(f"✅ Workflow completed successfully!")
        logger.info(f"📊 Total processing time: {state['total_processing_time']:.2f}s")

        return state

    async def _error_handling_node(self, state: WorkflowState) -> WorkflowState:
        """Handle workflow errors gracefully"""
        start_time = datetime.utcnow()
        state["current_stage"] = WorkflowStage.ERROR_HANDLING

        logger.info(f"⚠️ Handling workflow errors...")

        # Create error response
        error_count = len(state["errors"])
        warning_count = len(state["warnings"])

        if error_count > 0:
            error_response = f"""I apologize, but I encountered {error_count} error(s) while processing your request: "{state['user_query']}"

Please try:
1. Rephrasing your question
2. Checking if all required documents are uploaded
3. Ensuring your question is clear and specific

If the problem persists, please contact support with session ID: {state['session_id']}"""

        else:
            error_response = "Your request was processed with some warnings, but I've done my best to provide a helpful response."

        state.update({
            "final_response": error_response,
            "confidence_rating": 0.3,
            "total_processing_time": sum(state["stage_timings"].values())
        })

        # Record timing
        end_time = datetime.utcnow()
        state["stage_timings"]["error_handling"] = (end_time - start_time).total_seconds()

        logger.warning(f"⚠️ Error handling completed (errors: {error_count}, warnings: {warning_count})")

        return state

    def _route_after_intent_analysis(self, state: WorkflowState) -> str:
        """Route workflow based on intent analysis results"""
        routing_decision = state.get("routing_decision", "error")

        logger.info(f"🔄 Routing workflow: {routing_decision}")

        if routing_decision == "web_search":
            return "web_search"
        elif routing_decision == "document_retrieval":
            return "document_retrieval"
        elif routing_decision == "document_processing":
            return "document_upload"
        else:
            return "error"

    async def process_user_request(self, user_query: str, session_id: str,
                                 uploaded_files: List[str] = None) -> Dict[str, Any]:
        """Process user request through the complete workflow"""

        logger.info(f"🚀 Processing user request from TIRUMALAMANAV")
        logger.info(f"📝 Query: '{user_query[:100]}...'")
        logger.info(f"🆔 Session: {session_id}")

        # Create initial state
        initial_state = WorkflowState(
            messages=[],
            user_query=user_query,
            session_id=session_id,
            current_stage=WorkflowStage.INITIALIZE,
            trace_id="",
            workflow_start_time=datetime.utcnow(),
            detected_intent="",
            routing_decision="",
            confidence_score=0.0,
            uploaded_files=uploaded_files or [],
            processed_documents=[],
            retrieval_results=[],
            context_chunks=[],
            web_search_results=[],
            external_sources=[],
            llm_prompt="",
            llm_response="",
            response_metadata={},
            conversation_history=[],
            previous_context="",
            memory_updated=False,
            total_processing_time=0.0,
            stage_timings={},
            agent_interactions=[],
            errors=[],
            warnings=[],
            final_response="",
            sources_used=[],
            confidence_rating=0.0
        )

        try:
            # Run workflow
            config = {"configurable": {"thread_id": session_id}}
            final_state = await self.app.ainvoke(initial_state, config)

            # Extract results
            result = {
                "success": True,
                "response": final_state["final_response"],
                "confidence": final_state["confidence_rating"],
                "sources": final_state["sources_used"],
                "processing_time": final_state["total_processing_time"],
                "trace_id": final_state["trace_id"],
                "session_id": session_id,
                "stage_timings": final_state["stage_timings"],
                "errors": final_state["errors"],
                "warnings": final_state["warnings"],
                "workflow_metadata": {
                    "detected_intent": final_state["detected_intent"],
                    "routing_decision": final_state["routing_decision"],
                    "retrieval_results_count": len(final_state["retrieval_results"]),
                    "web_results_count": len(final_state["web_search_results"]),
                    "memory_updated": final_state["memory_updated"]
                }
            }

            logger.info(f"✅ Workflow completed successfully!")
            logger.info(f"📊 Processing time: {result['processing_time']:.2f}s")
            logger.info(f"🎯 Confidence: {result['confidence']:.1%}")

            return result

        except Exception as e:
            logger.error(f"❌ Workflow execution failed: {str(e)}")

            return {
                "success": False,
                "response": f"I apologize, but I encountered an error processing your request. Please try again. Error: {str(e)}",
                "confidence": 0.0,
                "sources": [],
                "processing_time": 0.0,
                "trace_id": "",
                "session_id": session_id,
                "stage_timings": {},
                "errors": [{"error": str(e), "timestamp": datetime.utcnow().isoformat()}],
                "warnings": [],
                "workflow_metadata": {}
            }

# ==================== WORKFLOW FACTORY ====================

class WorkflowFactory:
    """Factory for creating and managing workflow instances"""

    @staticmethod
    def create_production_workflow(document_processor=None, llm_client=None,
                                 web_search_client=None) -> LangGraphWorkflow:
        """Create production-ready workflow instance"""

        # Initialize agent manager
        agent_manager = AgentManager()

        # Create workflow
        workflow = LangGraphWorkflow(
            agent_manager=agent_manager,
            document_processor=document_processor,
            llm_client=llm_client,
            web_search_client=web_search_client
        )

        logger.info("🏭 Production workflow created successfully")

        return workflow

# ==================== TESTING FUNCTIONALITY ====================

async def test_langgraph_workflow():
    """Test the complete LangGraph workflow"""
    print("🧪 Testing LangGraph Workflow")
    print("🎯 Author: TIRUMALAMANAV")
    print("=" * 60)

    # Create workflow
    workflow = WorkflowFactory.create_production_workflow()

    # Test queries
    test_queries = [
        {
            "query": "What is artificial intelligence and how does machine learning work?",
            "session_id": "test_session_MANAV_1",
            "expected_intent": "document_search"
        },
        {
            "query": "What are the latest AI developments in 2025?",
            "session_id": "test_session_MANAV_2",
            "expected_intent": "web_search_needed"
        }
    ]

    for i, test_case in enumerate(test_queries, 1):
        print(f"\n🧪 Test Case {i}: {test_case['query'][:50]}...")

        start_time = datetime.utcnow()
        result = await workflow.process_user_request(
            test_case["query"],
            test_case["session_id"]
        )
        end_time = datetime.utcnow()

        print(f"✅ Success: {result['success']}")
        print(f"⏱️ Time: {(end_time - start_time).total_seconds():.2f}s")
        print(f"🎯 Confidence: {result['confidence']:.1%}")
        print(f"📚 Sources: {len(result['sources'])}")
        print(f"📝 Response length: {len(result['response'])}")

        if result['workflow_metadata']:
            metadata = result['workflow_metadata']
            print(f"🔍 Intent: {metadata.get('detected_intent', 'unknown')}")
            print(f"🔄 Routing: {metadata.get('routing_decision', 'unknown')}")

    print(f"\n🎉 LangGraph workflow testing completed!")

if __name__ == "__main__":
    # Run workflow test
    asyncio.run(test_langgraph_workflow())
