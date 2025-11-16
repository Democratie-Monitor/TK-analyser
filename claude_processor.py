"""
Claude API Processor for TK-Analyser

Robust processor for delegitimation analysis using Claude API with:
- Automatic retries with exponential backoff
- Comprehensive error handling
- Token usage tracking
- Multiple output modes (concise/elaborate)
"""

import anthropic
import json
import logging
import time
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class OutputMode(Enum):
    """Output verbosity modes for analysis."""
    CONCISE = "concise"      # Minimal tokens, just findings
    STANDARD = "standard"    # Balanced output
    ELABORATE = "elaborate"  # Detailed reasoning and context


@dataclass
class TokenUsage:
    """Track token usage for cost monitoring."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

    def add(self, other: 'TokenUsage') -> None:
        """Add token counts from another usage object."""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_creation_tokens += other.cache_creation_tokens
        self.cache_read_tokens += other.cache_read_tokens

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AnalysisResult:
    """Result of a single analysis."""
    success: bool
    data: Dict[str, Any]
    token_usage: TokenUsage
    error: Optional[str] = None
    retries: int = 0
    response_time: float = 0.0


class ClaudeProcessor:
    """Process text using Claude API with robust error handling."""

    # Pricing per 1M tokens (as of 2025)
    PRICING = {
        "claude-sonnet-4-20250514": {
            "input": 3.0,
            "output": 15.0,
            "cache_write": 3.75,
            "cache_read": 0.30
        }
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        timeout: float = 120.0
    ):
        """
        Initialize the Claude processor.

        Args:
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
            model: Claude model to use
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay between retries (seconds)
            timeout: Request timeout in seconds
        """
        try:
            self.client = anthropic.Anthropic(api_key=api_key)
            logger.info(f"Initialized Anthropic client for model: {model}")
        except anthropic.AuthenticationError as e:
            logger.error(f"Authentication failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            raise

        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout

        # Validate model pricing
        if model not in self.PRICING:
            logger.warning(f"Model {model} not in pricing table, using default pricing")

        # Track cumulative token usage
        self.total_usage = TokenUsage()
        self.request_count = 0
        self.error_count = 0

    def process_text(
        self,
        text: str,
        system_prompt: str,
        user_prompt: str,
        output_mode: OutputMode = OutputMode.STANDARD,
        max_tokens: int = 4096,
        temperature: float = 0.1
    ) -> AnalysisResult:
        """
        Process text using Claude API with automatic retries.

        Args:
            text: Speech text to analyze
            system_prompt: System prompt for Claude
            user_prompt: User prompt template (should contain {text} placeholder)
            output_mode: Verbosity level for output
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Returns:
            AnalysisResult with parsed data or error information
        """
        # Adjust max_tokens based on output mode
        if output_mode == OutputMode.CONCISE:
            max_tokens = min(max_tokens, 1024)
        elif output_mode == OutputMode.ELABORATE:
            max_tokens = max(max_tokens, 8192)

        # Validate and format the user prompt with the text
        if '{text}' not in user_prompt:
            logger.error("User prompt missing {text} placeholder")
            raise ValueError("User prompt must contain {text} placeholder")

        # Log analysis parameters
        logger.debug(f"Processing text: {len(text)} chars, mode={output_mode.value}")
        logger.debug(f"API config: model={self.model}, temp={temperature}, max_tokens={max_tokens}")

        try:
            formatted_prompt = user_prompt.format(text=text)
        except KeyError as e:
            logger.error(f"Prompt formatting failed, missing key: {e}")
            raise ValueError(f"Prompt template missing required placeholder: {e}")

        logger.debug(f"Formatted prompt length: {len(formatted_prompt)} chars")

        last_error = None
        token_usage = TokenUsage()
        total_retries = 0

        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": formatted_prompt}
                    ],
                    timeout=self.timeout
                )

                response_time = time.time() - start_time

                # Extract token usage
                usage = response.usage
                token_usage = TokenUsage(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_creation_tokens=getattr(usage, 'cache_creation_input_tokens', 0),
                    cache_read_tokens=getattr(usage, 'cache_read_input_tokens', 0)
                )

                # Update cumulative tracking
                self.total_usage.add(token_usage)
                self.request_count += 1

                # Parse the response
                parsed_data = self._parse_response(response)

                logger.info(
                    f"Successfully processed text (attempt {attempt + 1}): "
                    f"{token_usage.input_tokens} input, {token_usage.output_tokens} output tokens"
                )

                return AnalysisResult(
                    success=True,
                    data=parsed_data,
                    token_usage=token_usage,
                    retries=total_retries,
                    response_time=response_time
                )

            except anthropic.RateLimitError as e:
                last_error = f"Rate limit exceeded: {str(e)}"
                logger.warning(f"Rate limit hit on attempt {attempt + 1}")
                total_retries += 1
                self._wait_with_backoff(attempt)

            except anthropic.APIConnectionError as e:
                last_error = f"Connection error: {str(e)}"
                logger.warning(f"Connection error on attempt {attempt + 1}: {e}")
                total_retries += 1
                self._wait_with_backoff(attempt)

            except anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    # Server error, retry
                    last_error = f"Server error ({e.status_code}): {str(e)}"
                    logger.warning(f"Server error on attempt {attempt + 1}: {e}")
                    total_retries += 1
                    self._wait_with_backoff(attempt)
                else:
                    # Client error, don't retry
                    last_error = f"API error ({e.status_code}): {str(e)}"
                    logger.error(f"Client error: {e}")
                    self.error_count += 1
                    break

            except anthropic.APITimeoutError as e:
                last_error = f"Request timed out: {str(e)}"
                logger.warning(f"Timeout on attempt {attempt + 1}")
                total_retries += 1
                self._wait_with_backoff(attempt)

            except json.JSONDecodeError as e:
                last_error = f"Failed to parse JSON response: {str(e)}"
                logger.error(f"JSON parsing error: {e}")
                total_retries += 1
                self._wait_with_backoff(attempt)

            except Exception as e:
                last_error = f"Unexpected error: {type(e).__name__}: {str(e)}"
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                self.error_count += 1
                break

        # All retries exhausted
        self.error_count += 1
        logger.error(f"Failed after {total_retries} retries: {last_error}")

        return AnalysisResult(
            success=False,
            data=self._get_empty_result(),
            token_usage=token_usage,
            error=last_error,
            retries=total_retries
        )

    def _parse_response(self, response) -> Dict[str, Any]:
        """Parse Claude's response and extract JSON data."""
        if not response.content:
            logger.error("Empty response.content from Claude API")
            raise ValueError("Empty response from Claude")

        if len(response.content) == 0:
            logger.error("Response content list is empty")
            raise ValueError("Empty response content list from Claude")

        # Log response metadata
        logger.debug(f"Response stop_reason: {response.stop_reason}")
        logger.debug(f"Response content blocks: {len(response.content)}")

        text_content = response.content[0].text.strip()
        logger.debug(f"Raw response length: {len(text_content)} chars")
        logger.debug(f"Response preview: {text_content[:200]}...")

        # Try direct JSON parsing first
        try:
            parsed = json.loads(text_content)
            logger.debug("Successfully parsed response as direct JSON")
            return parsed
        except json.JSONDecodeError as e:
            logger.debug(f"Direct JSON parse failed: {e}")
            pass

        # Try to extract JSON from markdown code blocks
        if "```json" in text_content:
            start = text_content.find("```json") + 7
            end = text_content.find("```", start)
            if end > start:
                json_str = text_content[start:end].strip()
                try:
                    parsed = json.loads(json_str)
                    logger.debug("Successfully parsed JSON from ```json block")
                    return parsed
                except json.JSONDecodeError as e:
                    logger.debug(f"JSON parse from ```json block failed: {e}")
                    pass

        # Try to extract JSON from plain code blocks
        if "```" in text_content:
            start = text_content.find("```") + 3
            end = text_content.find("```", start)
            if end > start:
                json_str = text_content[start:end].strip()
                try:
                    parsed = json.loads(json_str)
                    logger.debug("Successfully parsed JSON from ``` block")
                    return parsed
                except json.JSONDecodeError as e:
                    logger.debug(f"JSON parse from ``` block failed: {e}")
                    pass

        # Try to find JSON object directly
        logger.debug("Attempting to extract JSON object from text")
        json_data = self._extract_json_object(text_content)
        if json_data:
            logger.debug("Successfully extracted JSON object from text")
            return json_data

        logger.error(f"No valid JSON found in response. Content: {text_content[:500]}...")
        raise json.JSONDecodeError("No valid JSON found in response", text_content, 0)

    def _extract_json_object(self, text: str) -> Optional[Dict]:
        """Extract first complete JSON object from text."""
        start_idx = text.find('{')
        if start_idx == -1:
            return None

        brace_count = 0
        in_string = False
        escape_next = False

        for i in range(start_idx, len(text)):
            char = text[i]

            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = text[start_idx:i+1]
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            # Continue searching for another JSON object
                            start_idx = text.find('{', i+1)
                            if start_idx == -1:
                                return None
                            brace_count = 0

        return None

    def _wait_with_backoff(self, attempt: int) -> None:
        """Wait with exponential backoff."""
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        # Add jitter to avoid thundering herd
        import random
        jitter = random.uniform(0, delay * 0.1)
        total_delay = delay + jitter

        logger.info(f"Waiting {total_delay:.2f} seconds before retry...")
        time.sleep(total_delay)

    def _get_empty_result(self) -> Dict[str, Any]:
        """Return empty result structure for failed analyses."""
        return {
            "gevonden_delegitimatie": [],
            "samenvatting": {
                "aantal_gevallen": 0,
                "meest_voorkomende_type": "error",
                "meest_getroffen_doelgroep": "error",
                "ernst_score": 0,
                "gemiddelde_confidence": 0.0,
                "hoogste_confidence": 0.0,
                "laagste_confidence": 0.0
            }
        }

    def get_cost_estimate(self) -> Dict[str, float]:
        """Calculate estimated cost based on token usage."""
        pricing = self.PRICING.get(self.model, self.PRICING["claude-sonnet-4-20250514"])

        input_cost = (self.total_usage.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.total_usage.output_tokens / 1_000_000) * pricing["output"]
        cache_write_cost = (self.total_usage.cache_creation_tokens / 1_000_000) * pricing["cache_write"]
        cache_read_cost = (self.total_usage.cache_read_tokens / 1_000_000) * pricing["cache_read"]

        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "cache_write_cost": cache_write_cost,
            "cache_read_cost": cache_read_cost,
            "total_cost": input_cost + output_cost + cache_write_cost + cache_read_cost
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "token_usage": self.total_usage.to_dict(),
            "estimated_cost": self.get_cost_estimate()
        }

    def reset_statistics(self) -> None:
        """Reset cumulative statistics."""
        self.total_usage = TokenUsage()
        self.request_count = 0
        self.error_count = 0
