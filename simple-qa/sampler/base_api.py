"""
Base API class for simple-qa sampler APIs.

This is a minimal base class to support the existing QueryTSAPI implementation.
"""

from typing import Dict, Any


class BaseAPI:
    """Base class for API implementations."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with configuration."""
        self.config = config or {}

    def initialize(self) -> None:
        """Initialize the API. Override in subclasses."""
        pass

    def query(self, question: str, source_dataset: str = "unknown") -> str:
        """Query the API. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement query method")

    def get_name(self) -> str:
        """Return the API name. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement get_name method")

    def get_metadata(self) -> Dict[str, Any]:
        """Return API metadata."""
        return {
            "name": self.get_name(),
            "config": self.config,
        }

    def validate_config(self) -> bool:
        """Validate API configuration. Override in subclasses."""
        return True

    def cleanup(self) -> None:
        """Clean up API resources. Override in subclasses."""
        pass