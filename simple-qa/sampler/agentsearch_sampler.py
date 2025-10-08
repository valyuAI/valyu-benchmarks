"""
AgentSearch Sampler for simple-qa benchmarking framework.

This sampler wraps the AgentSearchAPI and query.ts functionality to provide
a standardized interface for the simple-qa benchmarking system.
"""

import time
import logging
from typing import Any, Dict, Optional

from ..types import MessageList, SamplerBase, SamplerResponse
from .agentsearch import AgentSearchAPI

logger = logging.getLogger(__name__)


class AgentSearchSampler(SamplerBase):
    """
    Sampler that uses AgentSearch API (Node.js + Vertex AI Gemini + Search Tools) for Q&A.

    This sampler integrates the AgentSearchAPI and query.ts infrastructure
    with the simple-qa benchmarking framework.
    """

    def __init__(
        self,
        script_path: str = "query.ts",
        timeout: int = 300,
        max_retries: int = 3,
        source_dataset: str = "unknown",
        tool_type: str = "valyu",
        system_message: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize AgentSearch sampler.

        Args:
            script_path: Path to the query.ts script (relative or absolute)
            timeout: Timeout in seconds for query.ts subprocess calls
            max_retries: Maximum number of retry attempts
            tool_type: Search tool to use ("valyu", "google", "exa", "parallel")
            system_message: System message (for compatibility, not used by AgentSearch)
            **kwargs: Additional configuration options
        """
        self.script_path = script_path
        self.timeout = timeout
        self.max_retries = max_retries
        self.source_dataset = source_dataset
        self.tool_type = tool_type
        self.system_message = system_message

        # Store additional config for AgentSearchAPI
        self.config = {
            "script_path": script_path,
            "timeout": timeout,
            "max_retries": max_retries,
            "tool_type": tool_type,
            **kwargs
        }

        # Lazy initialization of AgentSearchAPI
        self._api = None
        self._initialized = False

    def _get_api(self) -> AgentSearchAPI:
        """Lazy initialization of AgentSearchAPI."""
        if self._api is None:
            self._api = AgentSearchAPI(config=self.config)
            try:
                self._api.initialize()
                self._initialized = True
                logger.info("AgentSearchAPI initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize AgentSearchAPI: {e}")
                self._initialized = False
                raise
        return self._api

    def _pack_message(self, role: str, content: Any) -> Dict[str, Any]:
        """
        Pack message in standard format required by evaluations.

        Args:
            role: Message role ("user", "assistant", "system")
            content: Message content

        Returns:
            Packed message dictionary
        """
        return {"role": str(role), "content": content}

    def _extract_question_from_messages(self, message_list: MessageList) -> str:
        """
        Extract the main question from the message list.

        Args:
            message_list: List of messages from the evaluation

        Returns:
            The question string to process
        """
        # Find the last user message as the main question
        for message in reversed(message_list):
            if message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, str) and content.strip():
                    return content.strip()
                elif isinstance(content, list):
                    # Handle multimodal content - extract text parts
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    if text_parts:
                        return " ".join(text_parts).strip()

        # Fallback: join all user messages
        user_messages = []
        for message in message_list:
            if message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, str):
                    user_messages.append(content)

        if user_messages:
            return " ".join(user_messages).strip()

        raise ValueError("No user message found in message list")

    def _detect_source_dataset(self, message_list: MessageList) -> str:
        """
        Detect source dataset from message metadata or content.

        Args:
            message_list: List of messages from the evaluation

        Returns:
            Source dataset type 
        """
        # Check message metadata for source dataset hints
        for message in message_list:
            metadata = message.get("metadata", {})
            if isinstance(metadata, dict):
                source = metadata.get("source_dataset")
                if source and isinstance(source, str):
                    return source.lower()

        # Check message content for dataset clues
        question = self._extract_question_from_messages(message_list)
        question_lower = question.lower()

        return self.source_dataset

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        """
        Process messages through AgentSearch API.

        Args:
            message_list: List of messages to process

        Returns:
            SamplerResponse with AgentSearch result
        """
        try:
            # Extract the question from messages
            question = self._extract_question_from_messages(message_list)
            logger.debug(f"Extracted question: {question[:100]}...")

            # Detect source dataset
            source_dataset = self._detect_source_dataset(message_list)
            logger.debug(f"Detected source dataset: {source_dataset}")

            # Get the API instance (lazy initialization)
            api = self._get_api()

            # Query the API with tool type
            start_time = time.time()
            response_text = api.query(question, self.tool_type)
            query_time = time.time() - start_time

            # Get tool outputs from the last query
            tool_outputs = api.get_last_tool_outputs()

            # Prepare response metadata
            response_metadata = {
                "query_time": query_time,
                "source_dataset": source_dataset,
                "tool_type": self.tool_type,
                "tool_outputs_count": len(tool_outputs),
                "tool_outputs": tool_outputs,
                "script_path": str(self.script_path),
                "api_metadata": api.get_metadata()
            }

            logger.info(f"AgentSearch query completed in {query_time:.2f}s with {len(tool_outputs)} tool outputs")

            return SamplerResponse(
                response_text=response_text,
                actual_queried_message_list=message_list,
                response_metadata=response_metadata,
            )

        except Exception as e:
            error_msg = f"AgentSearch API error: {str(e)}"
            logger.error(error_msg)

            # Return error response instead of raising
            return SamplerResponse(
                response_text=f"Error: {error_msg}",
                actual_queried_message_list=message_list,
                response_metadata={
                    "error": str(e),
                    "source_dataset": getattr(self, "source_dataset", "unknown"),
                    "tool_type": getattr(self, "tool_type", "valyu"),
                    "script_path": str(self.script_path),
                    "query_time": 0.0,
                    "tool_outputs_count": 0,
                    "tool_outputs": []
                },
            )

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._api is not None:
            self._api.cleanup()
            self._api = None
        self._initialized = False
        logger.debug("AgentSearchSampler cleanup completed")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors during cleanup