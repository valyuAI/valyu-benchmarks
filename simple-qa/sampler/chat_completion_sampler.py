import time
from typing import Any, Optional, Dict

import openai
from openai import OpenAI

from ..types import MessageList, SamplerBase, SamplerResponse

OPENAI_SYSTEM_MESSAGE_API = "You are a helpful assistant."
OPENAI_SYSTEM_MESSAGE_CHATGPT = (
    "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."
    + "\nKnowledge cutoff: 2023-12\nCurrent date: 2024-04-01"
)


class ChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API.

    This sampler wraps OpenAI's chat completion API for evaluating model responses.
    Supports text and image inputs with configurable temperature and token limits.
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        system_message: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ) -> None:
        """
        Initialize the OpenAI chat completion sampler.

        Args:
            model: OpenAI model identifier (e.g., 'gpt-4.1-2025-04-14')
            system_message: Optional system message to prepend to conversations
            temperature: Sampling temperature (0.0-1.0). Lower = more deterministic
            max_tokens: Maximum tokens to generate in response
        """
        self.api_key_name = "OPENAI_API_KEY"
        self.client = OpenAI()
        # using api_key=os.environ.get("OPENAI_API_KEY")  # please set your API_KEY
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"

    def _handle_image(
        self,
        image: str,
        encoding: str = "base64",
        format: str = "png",
        fovea: int = 768,
    ) -> Dict[str, Any]:
        """
        Format image for OpenAI API.

        Args:
            image: Base64-encoded image string
            encoding: Encoding type (default: 'base64')
            format: Image format (default: 'png')
            fovea: Resolution parameter (unused, kept for compatibility)

        Returns:
            Dictionary formatted for OpenAI image_url content type
        """
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str) -> Dict[str, str]:
        """
        Format text for OpenAI API.

        Args:
            text: Text content

        Returns:
            Dictionary formatted for OpenAI text content type
        """
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any) -> Dict[str, Any]:
        """
        Pack a message in OpenAI API format.

        Args:
            role: Message role ('user', 'assistant', 'system')
            content: Message content (text or structured content)

        Returns:
            Dictionary with 'role' and 'content' keys
        """
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        """
        Generate a response using OpenAI's chat completion API.

        Args:
            message_list: List of message dictionaries with 'role' and 'content'

        Returns:
            SamplerResponse containing the generated text, metadata, and original messages

        Raises:
            ValueError: If API returns empty response
            openai.BadRequestError: If request is malformed (caught and returns error response)
            Exception: On other API errors (triggers exponential backoff retry)
        """
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list
        trial = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("OpenAI API returned empty response; retrying")
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=message_list,
                )
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return SamplerResponse(
                    response_text="No response (bad request).",
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
