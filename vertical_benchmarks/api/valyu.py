"""
Valyu API implementation for benchmarking.

This module implements the Valyu API interface using the Anthropic Claude + Valyu provider
integration as specified in the user's requirements.
"""

import logging
from datetime import datetime
from typing import Dict, Any

from anthropic import Anthropic
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from valyu import AnthropicProvider

from .base_api import BaseAPI

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class ValyuAPI(BaseAPI):
    """Valyu API implementation using Anthropic Claude + Valyu provider."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Valyu API with configuration."""
        super().__init__(config)

        # Default configuration
        self.model = self.config.get("model", "claude-sonnet-4-20250514")
        self.max_tokens = self.config.get("max_tokens", 10000)
        self.max_retries = self.config.get("max_retries", 3)
        self.max_tool_calls = self.config.get("max_tool_calls", 5)  # Limit tool calls per query

        # Initialize clients (lazy initialization)
        self.anthropic_client = None
        self.valyu_provider = None
        self.valyu_tools = None

    def initialize(self) -> None:
        """Initialize Anthropic client and Valyu provider."""
        try:
            # Initialize Anthropic client
            self.anthropic_client = Anthropic()

            # Initialize Valyu provider
            self.valyu_provider = AnthropicProvider()
            self.valyu_tools = self.valyu_provider.get_tools()

            logger.info("Valyu API with Anthropic Claude initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Valyu API: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def query(self, question: str, source_dataset: str = "unknown") -> str:
        """
        Query Valyu API with a financial question using Anthropic Claude.
        Allows Claude to intelligently use up to max_tool_calls (default: 5) when needed for validation or if unsure.

        Args:
            question: The financial question to ask

        Returns:
            str: Valyu's response

        Raises:
            Exception: If the API call fails after retries
        """
        if not self.anthropic_client:
            self.initialize()

        try:
            # Initialize tool call counter
            total_tool_calls = 0
            
            # Compose input messages for Anthropic format with strict iterative tool usage instructions
            messages = [
                {
                    "role": "user",
                    "content": f"""You are a financial research assistant with access to powerful financial data tools. Today's date is {datetime.now().strftime('%Y-%m-%d')}.

VALYU DATASET ROUTING - CRITICAL:
When making tool calls to Valyu, you MUST analyze the financial question and explicitly specify which data sources are relevant. Valyu has the following datasets available:
- balance_sheet: Balance sheet data and financial position information
- cashflow: Cash flow statements and liquidity analysis
- crypto: Cryptocurrency data, prices, and market information
- dividends: Dividend payments, yields, and distribution data
- earnings: Earnings reports, EPS, and profitability metrics
- forex: Foreign exchange rates and currency data
- income_statements: Income statement data, revenue, and expense information
- insider_transactions: Insider trading activity and executive transactions
- market_movers: Stock price movements, volume, and market activity
- stocks: General stock data, prices, and market information
- sec_filings: SEC filings, 10-K, 10-Q, proxy statements, and regulatory documents

This question is of type {question_type}.

DATASET SELECTION STRATEGY:
1. Analyze the question to identify what type of financial information is needed
2. Map the question to one or more relevant datasets from the list above
3. When making tool calls, explicitly mention the relevant datasets in your query
4. Examples of dataset mapping:
   - "What is Apple's revenue?" → income_statements, earnings
   - "Show me Tesla's cash position" → balance_sheet, cashflow
   - "Recent insider trading at Microsoft" → insider_transactions, sec_filings
   - "Bitcoin price analysis" → crypto, market_movers
   - "Dividend yield of Coca-Cola" → dividends, stocks
   - "Netflix's quarterly earnings" → earnings, income_statements, sec_filings

CRITICAL INSTRUCTIONS FOR TOOL USAGE:
You MUST use tools iteratively up to {self.max_tool_calls} times to thoroughly research this financial question. DO NOT provide a final answer until you have either:
1. Used all {self.max_tool_calls} available tool calls, OR
2. Obtained completely comprehensive and verified information that fully answers the question

ITERATIVE RESEARCH PROCESS:
- Tool Call 1: Start with broad research on the main topic/company/concept, specifying relevant datasets
- Tool Call 2: If initial results are incomplete, narrow down or search for specific missing information with targeted datasets
- Tool Call 3: If still missing details, search for additional context, recent news, or validation data from complementary datasets
- Tool Call 4: Cross-reference findings or search for any remaining gaps using alternative or additional datasets
- Tool Call 5: Final verification or search for the most current/specific information needed

WHEN TO CONTINUE TO NEXT TOOL CALL:
✓ If you only have partial information about the question
✓ If you need to verify or cross-check your findings
✓ If you're missing specific numbers, dates, or recent updates
✓ If the question has multiple parts and you've only addressed some
✓ If you found conflicting information that needs resolution
✓ If you need more context to provide a complete answer

WHEN TO STOP (ONLY):
✓ You have completely comprehensive information that fully answers ALL aspects of the question
✓ You have used all {self.max_tool_calls} tool calls

EXAMPLES OF INCOMPLETE ANSWERS REQUIRING MORE TOOL CALLS:
- "The company's revenue is approximately..." → CONTINUE: Get exact figures from income_statements
- "Based on recent reports..." → CONTINUE: Get the most current data from sec_filings
- "The stock has been performing..." → CONTINUE: Get specific performance metrics from stocks, market_movers
- "Several factors affect..." → CONTINUE: Get comprehensive list with details from multiple relevant datasets

DO NOT SETTLE FOR PARTIAL ANSWERS. Each tool call should build upon the previous one to create a comprehensive response.

Financial Question: {question}

Begin your research now with Tool Call 1. Remember to specify the relevant datasets (from the list above) in your query to ensure proper routing."""
                }
            ]

            logger.debug(f"Querying Valyu API with Anthropic Claude (enforcing iterative research with up to {self.max_tool_calls} tool calls): {question[:100]}...")

            # Tool use conversation loop - encourage iterative tool usage up to max limit
            while total_tool_calls < self.max_tool_calls:
                # Encourage tool usage throughout the research process
                if total_tool_calls == 0:
                    # Force first tool call to get Claude started with research
                    response = self.anthropic_client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        tools=self.valyu_tools,
                        tool_choice={"type": "any"},  # Force first tool usage
                        messages=messages
                    )
                else:
                    # Continue encouraging tool usage for thorough research
                    # Add a system reminder about the iterative process
                    enhanced_messages = messages + [
                        {
                            "role": "user", 
                            "content": f"Continue your research. You have used {total_tool_calls}/{self.max_tool_calls} tool calls so far. Based on your previous findings, determine if you need additional information to provide a completely comprehensive answer. If there are any gaps, uncertainties, or missing details, make another tool call. Only stop if you have thoroughly comprehensive information that fully addresses the question."
                        }
                    ]
                    
                    response = self.anthropic_client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        tools=self.valyu_tools,
                        tool_choice={"type": "any"},  # Continue encouraging tool usage
                        messages=enhanced_messages
                    )

                # Check if Claude wants to use tools
                current_tool_calls = [c for c in response.content if getattr(c, 'type', None) == 'tool_use']
                current_tool_count = len(current_tool_calls)
                
                # If Claude doesn't want to use tools, check if we should encourage one more attempt
                if current_tool_count == 0:
                    # If we haven't used many tool calls yet, encourage one more attempt
                    if total_tool_calls < 3:
                        logger.debug(f"Claude stopped tool usage early after {total_tool_calls} calls. Encouraging continuation...")
                        # Try one more time with stronger encouragement
                        encourage_messages = messages + [
                            {
                                "role": "user",
                                "content": f"You've only used {total_tool_calls}/{self.max_tool_calls} tool calls. Please make sure you have truly comprehensive information. Review your current findings and identify ANY gaps, missing details, or areas that could benefit from additional research. Make another tool call to gather more specific, detailed, or recent information. This is critical for providing the most accurate and complete financial analysis."
                            }
                        ]
                        
                        response = self.anthropic_client.messages.create(
                            model=self.model,
                            max_tokens=self.max_tokens,
                            tools=self.valyu_tools,
                            tool_choice={"type": "any"},
                            messages=encourage_messages
                        )
                        
                        # Check again for tool calls
                        current_tool_calls = [c for c in response.content if getattr(c, 'type', None) == 'tool_use']
                        current_tool_count = len(current_tool_calls)
                        
                        # If still no tool calls, proceed to final answer
                        if current_tool_count == 0:
                            logger.debug(f"Claude definitively finished research after {total_tool_calls} tool calls despite encouragement.")
                            # Extract final response
                            result_parts = []
                            for content in response.content:
                                if hasattr(content, "text"):
                                    result_parts.append(content.text)
                                elif isinstance(content, dict) and content.get("type") == "text":
                                    result_parts.append(content.get("text", ""))
                            
                            result = "\n".join(result_parts).strip()
                            logger.debug(f"Valyu API response (after {total_tool_calls} tool calls): {result[:100]}...")
                            return result
                    else:
                        logger.debug(f"Claude finished research after {total_tool_calls} tool calls. Providing final answer.")
                        # Extract final response
                        result_parts = []
                        for content in response.content:
                            if hasattr(content, "text"):
                                result_parts.append(content.text)
                            elif isinstance(content, dict) and content.get("type") == "text":
                                result_parts.append(content.get("text", ""))
                        
                        result = "\n".join(result_parts).strip()
                        logger.debug(f"Valyu API response (after {total_tool_calls} tool calls): {result[:100]}...")
                        return result

                # Check if this would exceed our limit
                remaining_calls = self.max_tool_calls - total_tool_calls
                if current_tool_count > remaining_calls:
                    # Limit tool calls to not exceed max
                    current_tool_calls = current_tool_calls[:remaining_calls]
                    current_tool_count = remaining_calls
                    logger.debug(f"Limiting to {current_tool_count} tool calls to stay within max limit of {self.max_tool_calls}")

                total_tool_calls += current_tool_count
                logger.debug(f"Processing {current_tool_count} tool calls (total: {total_tool_calls}/{self.max_tool_calls})")

                # Execute tool calls using Valyu provider
                if current_tool_count < len([c for c in response.content if getattr(c, 'type', None) == 'tool_use']):
                    # Create a modified response with only the allowed tool calls
                    limited_content = []
                    tool_calls_added = 0
                    for content in response.content:
                        if getattr(content, 'type', None) == 'tool_use' and tool_calls_added < current_tool_count:
                            limited_content.append(content)
                            tool_calls_added += 1
                        elif getattr(content, 'type', None) != 'tool_use':
                            limited_content.append(content)
                    
                    # Create a modified response object
                    class LimitedResponse:
                        def __init__(self, original_response, limited_content):
                            self.content = limited_content
                            # Copy other attributes from original response
                            for attr in dir(original_response):
                                if not attr.startswith('_') and attr != 'content':
                                    setattr(self, attr, getattr(original_response, attr))
                    
                    limited_response = LimitedResponse(response, limited_content)
                    tool_results = self.valyu_provider.handle_tool_calls(response=limited_response)
                else:
                    tool_results = self.valyu_provider.handle_tool_calls(response=response)

                logger.debug(f"Executed {len(tool_results)} tool calls")

                # Build updated conversation with tool results
                messages = self.valyu_provider.build_conversation(
                    messages, response, tool_results
                )

            # If we've reached the tool call limit, get final response without tools
            logger.debug(f"Reached maximum tool calls limit ({self.max_tool_calls}). Getting final response.")
            final_response = self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=2000,  # Increased for final response
                messages=messages
            )
            
            # Extract text content from final response
            result_parts = []
            for content in final_response.content:
                if hasattr(content, "text"):
                    result_parts.append(content.text)
                elif isinstance(content, dict) and content.get("type") == "text":
                    result_parts.append(content.get("text", ""))

            result = "\n".join(result_parts).strip()
            logger.debug(f"Valyu API response (after {total_tool_calls} tool calls): {result[:100]}...")
            return result

        except Exception as e:
            logger.error(f"Valyu API query failed: {e}")
            raise Exception(f"Valyu API error: {str(e)}")

    def get_name(self) -> str:
        """Return the API name."""
        return "valyu"

    def get_metadata(self) -> Dict[str, Any]:
        """Return Valyu API metadata."""
        metadata = super().get_metadata()
        metadata.update({
            "model": self.model,
            "provider": "Anthropic Claude + Valyu",
            "max_tokens": self.max_tokens,
            "max_tool_calls": self.max_tool_calls,
            "version": "3.0",
            "tool_choice": "strict_iterative_research_up_to_max_limit"
        })
        return metadata

    def validate_config(self) -> bool:
        """Validate Valyu API configuration."""
        try:
            # Check if required environment variables are set
            import os

            valyu_key = os.getenv("VALYU_API_KEY")
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")

            if not valyu_key:
                logger.error("VALYU_API_KEY not found in environment")
                return False

            if not anthropic_key:
                logger.error("ANTHROPIC_API_KEY not found in environment")
                return False

            logger.info("Valyu API with Anthropic Claude configuration validated successfully")
            return True

        except Exception as e:
            logger.error(f"Valyu API configuration validation failed: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up Valyu API resources."""
        # No specific cleanup needed for Anthropic/Valyu clients
        logger.debug("Valyu API with Anthropic Claude cleanup completed")