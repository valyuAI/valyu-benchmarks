"""
AgentSearch API implementation for benchmarking.

This module implements the AgentSearch API interface by calling the query.ts Node.js script
via subprocess for web search-augmented question answering.
"""

import logging
import subprocess
import json
import os
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from .base_api import BaseAPI

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class AgentSearchAPI(BaseAPI):
    """AgentSearch API implementation using Node.js subprocess calls."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize AgentSearch API with configuration."""
        super().__init__(config)

        # Default configuration
        self.max_retries = self.config.get("max_retries", 3)
        self.timeout = self.config.get("timeout", 300)  # 5 minutes timeout
        self.script_path = self.config.get("script_path", "query.ts")
        self.tool_type = self.config.get("tool_type", "valyu")  # Default to valyu

        # Determine the full path to query.ts
        if not os.path.isabs(self.script_path):
            # If relative path, assume it's relative to the project root
            project_root = Path(__file__).parent.parent
            self.script_path = project_root / self.script_path

        # Store tool outputs from last query
        self._last_tool_outputs = []

    def initialize(self) -> None:
        """Initialize AgentSearch API by checking if Node.js and dependencies are available."""
        try:
            # Check if Node.js is available
            result = subprocess.run(
                ['node', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                raise Exception("Node.js not found or not working")

            logger.info(f"Node.js version: {result.stdout.strip()}")

            # Check if query.ts exists
            if not os.path.exists(self.script_path):
                raise Exception(f"AgentSearch script not found at: {self.script_path}")

            logger.info(f"AgentSearch API initialized successfully, script path: {self.script_path}")

        except Exception as e:
            logger.error(f"Failed to initialize AgentSearch API: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def query(self, question: str, tool_type: str = None) -> str:
        """
        Query AgentSearch API with a question by calling the Node.js script.

        Args:
            question: The question to ask
            tool_type: The search tool to use ("valyu", "google", "exa", "parallel")

        Returns:
            str: AgentSearch response

        Raises:
            Exception: If the API call fails after retries
        """
        if not os.path.exists(self.script_path):
            self.initialize()

        # Use provided tool_type or fall back to instance default
        selected_tool = tool_type if tool_type is not None else self.tool_type

        try:
            logger.debug(f"Querying AgentSearch API: {question[:100]}...")

            # Prepare the command to call query.ts
            cmd = [
                'npx', 'tsx',
                str(self.script_path),
                '--benchmark',
                '--question', question,
                '--type', 'financial',
                '--tool', selected_tool
            ]

            # Pass through existing environment variables
            env = os.environ.copy()

            # Execute the Node.js script
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
                cwd=Path(self.script_path).parent  # Set working directory to script directory
            )

            # Check for errors
            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                logger.error(f"AgentSearch script failed with return code {result.returncode}: {error_msg}")
                raise Exception(f"AgentSearch script error: {error_msg}")

            # Display stderr output for debugging (this includes console.error from Node.js)
            if result.stderr:
                stderr_output = result.stderr.strip()
                if stderr_output:
                    # Print stderr directly so debug messages are visible
                    print(f"[AgentSearch Debug] {stderr_output}")

            # Get the response from stdout
            response = result.stdout.strip()

            if not response:
                error_msg = result.stderr.strip() if result.stderr else "No response received"
                logger.error(f"Empty response from AgentSearch: {error_msg}")
                raise Exception(f"Empty response from AgentSearch: {error_msg}")

            logger.debug(f"AgentSearch API raw response: {response[:100]}...")

            # Parse JSON response to extract text and tool outputs
            try:
                parsed_response = json.loads(response)
                text_response = parsed_response.get('text', '')
                tool_outputs = parsed_response.get('toolOutputs', [])
                message_count = parsed_response.get('messageCount', 0)

                logger.debug(f"AgentSearch parsed - Text: {len(text_response)} chars, Tools: {len(tool_outputs)}, Messages: {message_count}")

                # Store tool outputs for later retrieval
                self._last_tool_outputs = tool_outputs

                return text_response

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response from AgentSearch, treating as plain text: {e}")
                # Fallback to treating response as plain text
                self._last_tool_outputs = []
                return response

        except subprocess.TimeoutExpired:
            logger.error(f"AgentSearch script timed out after {self.timeout} seconds")
            raise Exception(f"AgentSearch script timeout ({self.timeout}s)")
        except Exception as e:
            logger.error(f"AgentSearch API query failed: {e}")
            raise Exception(f"AgentSearch API error: {str(e)}")

    def get_name(self) -> str:
        """Return the API name."""
        return "agentsearch"

    def get_metadata(self) -> Dict[str, Any]:
        """Return AgentSearch API metadata."""
        metadata = super().get_metadata()
        metadata.update({
            "provider": "AgentSearch + Vertex AI Gemini",
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "script_path": str(self.script_path),
            "tool_type": self.tool_type,
            "version": "1.0",
            "model": "vertex/gemini-2.5-pro",
            "tool_choice": "iterative_research_up_to_5_calls"
        })
        return metadata

    def validate_config(self) -> bool:
        """Validate AgentSearch API configuration."""
        try:
            # Check if required environment variables are set
            gemini_api_key = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")

            if not gemini_api_key:
                logger.error("GOOGLE_GENERATIVE_AI_API_KEY not found in environment")
                return False

            # Check if Node.js is available
            result = subprocess.run(
                ['node', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                logger.error("Node.js not available")
                return False

            # Check if query.ts exists
            if not os.path.exists(self.script_path):
                logger.error(f"AgentSearch script not found at: {self.script_path}")
                return False

            logger.info("AgentSearch API configuration validated successfully")
            return True

        except Exception as e:
            logger.error(f"AgentSearch API configuration validation failed: {e}")
            return False

    def get_last_tool_outputs(self) -> List[Dict[str, Any]]:
        """Get tool outputs from the last query."""
        return self._last_tool_outputs.copy() if self._last_tool_outputs else []

    def cleanup(self) -> None:
        """Clean up AgentSearch API resources."""
        # No specific cleanup needed for subprocess calls
        self._last_tool_outputs = []
        logger.debug("AgentSearch API cleanup completed")
