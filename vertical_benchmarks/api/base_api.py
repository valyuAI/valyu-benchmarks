"""
Base API interface for benchmarking system.

This module defines the abstract base class that all API implementations
must inherit from to ensure consistent interface across different APIs.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseAPI(ABC):
    """Abstract base class for all API implementations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the API with optional configuration.

        Args:
            config: Optional configuration dictionary specific to the API
        """
        self.config = config or {}

    @abstractmethod
    def query(self, question: str) -> str:
        """
        Query the API with a question and return the response.

        Args:
            question: The question to ask the API

        Returns:
            str: The API's response as a string

        Raises:
            Exception: If the API call fails
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Return the name of this API for identification.

        Returns:
            str: Unique identifier for this API
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """
        Return metadata about this API implementation.

        Returns:
            Dict containing API metadata like version, model, etc.
        """
        return {
            "name": self.get_name(),
            "config": self.config
        }

    def validate_config(self) -> bool:
        """
        Validate that the API configuration is correct.

        Returns:
            bool: True if configuration is valid
        """
        return True

    def initialize(self) -> None:
        """
        Initialize any required clients or resources.
        Called before the first query.
        """
        pass

    def cleanup(self) -> None:
        """
        Clean up any resources after benchmarking is complete.
        """
        pass