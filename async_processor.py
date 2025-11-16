"""
Async Claude API Processor for TK-Analyser

High-performance async processor for parallel API calls with:
- Concurrent request processing
- Rate limiting and semaphores
- Batch processing with configurable concurrency
- Memory-efficient streaming
"""

import anthropic
import asyncio
import json
import logging
import time
from typing import Dict, Optional, List, Any
from dataclasses import dataclass

from claude_processor import (
    ClaudeProcessor,
    OutputMode,
    TokenUsage,
    AnalysisResult
)

logger = logging.getLogger(__name__)


@dataclass
class AsyncBatchResult:
    """Results from async batch processing."""
    results: List[AnalysisResult]
    total_time: float
    successful: int
    failed: int
    total_tokens: int
    total_cost: float


class AsyncClaudeProcessor(ClaudeProcessor):
    """
    Async version of ClaudeProcessor for parallel API calls.

    Inherits all functionality from ClaudeProcessor and adds async methods.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        timeout: float = 120.0,
        max_concurrent: int = 5,
        rate_limit_delay: float = 0.1
    ):
        """
        Initialize async processor.

        Args:
            max_concurrent: Maximum concurrent API requests
            rate_limit_delay: Minimum delay between requests (seconds)
        """
        super().__init__(
            api_key=api_key,
            model=model,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            timeout=timeout
        )

        self.max_concurrent = max_concurrent
        self.rate_limit_delay = rate_limit_delay
        self._semaphore = None
        self._last_request_time = 0

        # Initialize async client
        try:
            self.async_client = anthropic.AsyncAnthropic(api_key=api_key)
            logger.info(f"Initialized async Anthropic client with max_concurrent={max_concurrent}")
        except Exception as e:
            logger.error(f"Failed to initialize async client: {e}")
            raise

    async def _wait_for_rate_limit(self) -> None:
        """Ensure minimum delay between requests."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time

        if time_since_last < self.rate_limit_delay:
            wait_time = self.rate_limit_delay - time_since_last
            logger.debug(f"Rate limit delay: waiting {wait_time:.3f}s")
            await asyncio.sleep(wait_time)

        self._last_request_time = time.time()

    async def process_text_async(
        self,
        text: str,
        system_prompt: str,
        user_prompt: str,
        output_mode: OutputMode = OutputMode.STANDARD,
        max_tokens: int = 4096,
        temperature: float = 0.1
    ) -> AnalysisResult:
        """
        Process text asynchronously using Claude API.

        Args:
            text: Speech text to analyze
            system_prompt: System prompt for Claude
            user_prompt: User prompt template
            output_mode: Verbosity level
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Returns:
            AnalysisResult
        """
        # Adjust max_tokens based on output mode
        if output_mode == OutputMode.CONCISE:
            max_tokens = min(max_tokens, 1024)
        elif output_mode == OutputMode.ELABORATE:
            max_tokens = max(max_tokens, 8192)

        # Validate prompt
        if '{text}' not in user_prompt:
            logger.error("User prompt missing {text} placeholder")
            raise ValueError("User prompt must contain {text} placeholder")

        try:
            formatted_prompt = user_prompt.format(text=text)
        except KeyError as e:
            logger.error(f"Prompt formatting failed: {e}")
            raise ValueError(f"Prompt template missing placeholder: {e}")

        logger.debug(f"Async processing text: {len(text)} chars")

        last_error = None
        token_usage = TokenUsage()
        total_retries = 0

        for attempt in range(self.max_retries + 1):
            try:
                # Rate limiting
                await self._wait_for_rate_limit()

                start_time = time.time()

                response = await self.async_client.messages.create(
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

                # Update cumulative tracking (thread-safe needed)
                self.total_usage.add(token_usage)
                self.request_count += 1

                # Parse response
                parsed_data = self._parse_response(response)

                logger.debug(
                    f"Async request successful (attempt {attempt + 1}): "
                    f"{token_usage.total_tokens} tokens, {response_time:.2f}s"
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
                await self._async_backoff(attempt)

            except anthropic.APIConnectionError as e:
                last_error = f"Connection error: {str(e)}"
                logger.warning(f"Connection error on attempt {attempt + 1}")
                total_retries += 1
                await self._async_backoff(attempt)

            except anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    last_error = f"Server error ({e.status_code}): {str(e)}"
                    logger.warning(f"Server error on attempt {attempt + 1}")
                    total_retries += 1
                    await self._async_backoff(attempt)
                else:
                    last_error = f"API error ({e.status_code}): {str(e)}"
                    logger.error(f"Client error: {e}")
                    self.error_count += 1
                    break

            except anthropic.APITimeoutError as e:
                last_error = f"Request timed out: {str(e)}"
                logger.warning(f"Timeout on attempt {attempt + 1}")
                total_retries += 1
                await self._async_backoff(attempt)

            except json.JSONDecodeError as e:
                last_error = f"Failed to parse JSON: {str(e)}"
                logger.error(f"JSON parsing error: {e}")
                total_retries += 1
                await self._async_backoff(attempt)

            except Exception as e:
                last_error = f"Unexpected error: {type(e).__name__}: {str(e)}"
                logger.error(f"Unexpected error: {e}")
                self.error_count += 1
                break

        # All retries exhausted
        self.error_count += 1
        logger.error(f"Async request failed after {total_retries} retries: {last_error}")

        return AnalysisResult(
            success=False,
            data=self._get_empty_result(),
            token_usage=token_usage,
            error=last_error,
            retries=total_retries
        )

    async def _async_backoff(self, attempt: int) -> None:
        """Async wait with exponential backoff."""
        import random
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        jitter = random.uniform(0, delay * 0.1)
        total_delay = delay + jitter

        logger.debug(f"Async backoff: waiting {total_delay:.2f}s")
        await asyncio.sleep(total_delay)

    async def process_batch_async(
        self,
        texts: List[str],
        system_prompt: str,
        user_prompt: str,
        output_mode: OutputMode = OutputMode.STANDARD,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        progress_callback: Optional[callable] = None
    ) -> AsyncBatchResult:
        """
        Process multiple texts concurrently.

        Args:
            texts: List of texts to analyze
            system_prompt: System prompt
            user_prompt: User prompt template
            output_mode: Output verbosity
            max_tokens: Max tokens per response
            temperature: Sampling temperature
            progress_callback: Optional callback(completed, total) for progress

        Returns:
            AsyncBatchResult with all results
        """
        if not texts:
            logger.warning("Empty text list for batch processing")
            return AsyncBatchResult(
                results=[],
                total_time=0.0,
                successful=0,
                failed=0,
                total_tokens=0,
                total_cost=0.0
            )

        logger.info(f"Starting async batch processing: {len(texts)} texts, max_concurrent={self.max_concurrent}")
        start_time = time.time()

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)
        completed = 0
        results = []

        async def process_with_semaphore(idx: int, text: str) -> AnalysisResult:
            nonlocal completed
            async with semaphore:
                result = await self.process_text_async(
                    text=text,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    output_mode=output_mode,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(texts))
                logger.debug(f"Batch progress: {completed}/{len(texts)}")
                return result

        # Process all texts concurrently
        tasks = [
            process_with_semaphore(i, text)
            for i, text in enumerate(texts)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions that weren't caught
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} raised exception: {result}")
                processed_results.append(AnalysisResult(
                    success=False,
                    data=self._get_empty_result(),
                    token_usage=TokenUsage(),
                    error=str(result),
                    retries=0
                ))
            else:
                processed_results.append(result)

        total_time = time.time() - start_time
        successful = sum(1 for r in processed_results if r.success)
        failed = len(processed_results) - successful
        total_tokens = sum(r.token_usage.total_tokens for r in processed_results)

        # Calculate cost
        cost_estimate = self.get_cost_estimate()
        total_cost = cost_estimate['total_cost']

        logger.info(
            f"Async batch complete: {successful}/{len(texts)} successful, "
            f"{total_time:.2f}s total, {total_tokens} tokens"
        )

        return AsyncBatchResult(
            results=processed_results,
            total_time=total_time,
            successful=successful,
            failed=failed,
            total_tokens=total_tokens,
            total_cost=total_cost
        )

    def process_batch_sync(
        self,
        texts: List[str],
        system_prompt: str,
        user_prompt: str,
        output_mode: OutputMode = OutputMode.STANDARD,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        progress_callback: Optional[callable] = None
    ) -> AsyncBatchResult:
        """
        Synchronous wrapper for async batch processing.

        Use this when calling from non-async code.
        """
        return asyncio.run(
            self.process_batch_async(
                texts=texts,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_mode=output_mode,
                max_tokens=max_tokens,
                temperature=temperature,
                progress_callback=progress_callback
            )
        )

    async def close(self) -> None:
        """Close async client connections."""
        if hasattr(self, 'async_client'):
            await self.async_client.close()
            logger.info("Closed async Anthropic client")


# Example usage
async def example_async_usage():
    """Example of async batch processing."""
    processor = AsyncClaudeProcessor(max_concurrent=3)

    texts = [
        "Dit is een positieve tekst over de regering.",
        "De oppositie heeft gelijk met hun kritiek.",
        "Het parlement doet uitstekend werk."
    ]

    system_prompt = "You are a sentiment analyzer. Return JSON with sentiment label."
    user_prompt = "Analyze: {text}\n\nReturn: {{\"sentiment\": \"positive/negative/neutral\"}}"

    def progress(completed, total):
        print(f"Progress: {completed}/{total}")

    result = await processor.process_batch_async(
        texts=texts,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        progress_callback=progress
    )

    print(f"Completed: {result.successful}/{len(texts)} successful")
    print(f"Total time: {result.total_time:.2f}s")
    print(f"Total tokens: {result.total_tokens}")

    await processor.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_async_usage())
