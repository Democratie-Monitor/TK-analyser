"""
Memory Management and Resource Limits for TK-Analyser

Provides:
- Text length limits and validation
- Memory usage monitoring
- GPU memory management and cleanup
- Resource constraints for safe processing
"""

import logging
import gc
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MemoryLimits:
    """Configuration for memory and text limits."""
    max_text_length: int = 50000  # Max characters per text
    max_text_tokens: int = 16000  # Approximate max tokens (chars/4)
    max_batch_size: int = 100  # Max texts in single batch
    max_total_chars: int = 5_000_000  # Max total chars in memory
    warn_text_length: int = 10000  # Warn above this length
    truncate_to: int = 50000  # Truncate to this if exceeding max


class TextValidator:
    """Validate and sanitize text inputs."""

    def __init__(self, limits: Optional[MemoryLimits] = None):
        self.limits = limits or MemoryLimits()
        self._total_chars_processed = 0

    def validate_text(self, text: str) -> Tuple[str, dict]:
        """
        Validate and optionally truncate text.

        Args:
            text: Input text

        Returns:
            Tuple of (processed_text, validation_info)
        """
        info = {
            "original_length": len(text),
            "truncated": False,
            "warnings": []
        }

        if not text:
            logger.warning("Empty text provided")
            info["warnings"].append("empty_text")
            return "", info

        # Check for very long text
        if len(text) > self.limits.max_text_length:
            logger.warning(
                f"Text exceeds max length ({len(text)} > {self.limits.max_text_length}), truncating"
            )
            text = text[:self.limits.truncate_to]
            info["truncated"] = True
            info["truncated_length"] = len(text)
            info["warnings"].append("text_truncated")

        elif len(text) > self.limits.warn_text_length:
            logger.debug(f"Long text detected: {len(text)} chars")
            info["warnings"].append("long_text")

        # Check for problematic characters
        if '\x00' in text:
            logger.warning("Null bytes found in text, removing")
            text = text.replace('\x00', '')
            info["warnings"].append("null_bytes_removed")

        # Track total memory usage
        self._total_chars_processed += len(text)
        if self._total_chars_processed > self.limits.max_total_chars:
            logger.warning(
                f"Total chars processed ({self._total_chars_processed}) exceeds limit "
                f"({self.limits.max_total_chars}). Consider clearing cache."
            )
            info["warnings"].append("total_chars_high")

        info["final_length"] = len(text)
        info["estimated_tokens"] = len(text) // 4  # Rough estimate

        return text, info

    def validate_batch(self, texts: list) -> Tuple[list, dict]:
        """
        Validate a batch of texts.

        Args:
            texts: List of texts

        Returns:
            Tuple of (processed_texts, batch_info)
        """
        if len(texts) > self.limits.max_batch_size:
            logger.warning(
                f"Batch size ({len(texts)}) exceeds max ({self.limits.max_batch_size})"
            )

        batch_info = {
            "original_count": len(texts),
            "total_original_chars": sum(len(t) for t in texts),
            "truncated_count": 0,
            "warnings": []
        }

        processed_texts = []
        for i, text in enumerate(texts):
            processed, info = self.validate_text(text)
            processed_texts.append(processed)

            if info["truncated"]:
                batch_info["truncated_count"] += 1

        batch_info["total_final_chars"] = sum(len(t) for t in processed_texts)
        batch_info["estimated_total_tokens"] = batch_info["total_final_chars"] // 4

        if batch_info["truncated_count"] > 0:
            batch_info["warnings"].append(f"{batch_info['truncated_count']}_texts_truncated")

        logger.info(
            f"Batch validated: {len(texts)} texts, "
            f"{batch_info['total_final_chars']:,} chars, "
            f"~{batch_info['estimated_total_tokens']:,} tokens"
        )

        return processed_texts, batch_info

    def reset_counter(self) -> None:
        """Reset total chars processed counter."""
        self._total_chars_processed = 0
        logger.debug("Reset text validator counter")


class GPUMemoryManager:
    """Manage GPU memory for transformer models."""

    def __init__(self):
        self._torch_available = False
        self._cuda_available = False
        self._check_gpu()

    def _check_gpu(self) -> None:
        """Check GPU availability."""
        try:
            import torch
            self._torch_available = True
            self._cuda_available = torch.cuda.is_available()
            if self._cuda_available:
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                self._log_gpu_memory()
            else:
                logger.info("CUDA not available, using CPU")
        except ImportError:
            logger.debug("PyTorch not installed")
            self._torch_available = False

    def _log_gpu_memory(self) -> None:
        """Log current GPU memory usage."""
        if not self._cuda_available:
            return

        try:
            import torch
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            cached = torch.cuda.memory_reserved() / 1024 / 1024
            max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

            logger.debug(
                f"GPU Memory: {allocated:.1f}MB allocated, "
                f"{cached:.1f}MB cached, {max_memory:.1f}MB peak"
            )
        except Exception as e:
            logger.debug(f"Could not get GPU memory info: {e}")

    def clear_gpu_cache(self) -> None:
        """Clear GPU memory cache."""
        if not self._cuda_available:
            return

        try:
            import torch
            before = torch.cuda.memory_allocated() / 1024 / 1024
            torch.cuda.empty_cache()
            gc.collect()
            after = torch.cuda.memory_allocated() / 1024 / 1024

            logger.info(f"GPU cache cleared: {before:.1f}MB -> {after:.1f}MB")
        except Exception as e:
            logger.error(f"Failed to clear GPU cache: {e}")

    def get_memory_stats(self) -> dict:
        """Get current memory statistics."""
        stats = {
            "torch_available": self._torch_available,
            "cuda_available": self._cuda_available,
            "gpu_memory_mb": 0,
            "gpu_cached_mb": 0,
            "cpu_memory_mb": 0
        }

        # GPU memory
        if self._cuda_available:
            try:
                import torch
                stats["gpu_memory_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
                stats["gpu_cached_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
            except Exception:
                pass

        # CPU memory
        try:
            import psutil
            process = psutil.Process()
            stats["cpu_memory_mb"] = process.memory_info().rss / 1024 / 1024
        except ImportError:
            pass

        return stats


class ModelUnloader:
    """Safely unload transformer models from memory."""

    @staticmethod
    def unload_pipeline(pipeline) -> None:
        """
        Unload a HuggingFace pipeline from memory.

        Args:
            pipeline: HuggingFace pipeline object
        """
        if pipeline is None:
            return

        try:
            # Delete model and tokenizer
            if hasattr(pipeline, 'model'):
                del pipeline.model
            if hasattr(pipeline, 'tokenizer'):
                del pipeline.tokenizer

            del pipeline

            # Force garbage collection
            gc.collect()

            # Clear GPU cache
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("Model pipeline unloaded from memory")

        except Exception as e:
            logger.error(f"Error unloading model: {e}")

    @staticmethod
    def unload_transformers_cache() -> None:
        """Clear HuggingFace transformers cache from memory."""
        try:
            # Clear any cached models
            gc.collect()

            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            logger.info("Transformers cache cleared")

        except ImportError:
            pass
        except Exception as e:
            logger.error(f"Error clearing transformers cache: {e}")


class ResourceMonitor:
    """Monitor system resources during processing."""

    def __init__(self):
        self.gpu_manager = GPUMemoryManager()
        self.text_validator = TextValidator()

    def check_resources(self) -> dict:
        """Check current resource usage."""
        stats = self.gpu_manager.get_memory_stats()

        # Add system memory if available
        try:
            import psutil
            mem = psutil.virtual_memory()
            stats["system_memory_percent"] = mem.percent
            stats["system_memory_available_mb"] = mem.available / 1024 / 1024
        except ImportError:
            pass

        # Check if resources are constrained
        stats["gpu_memory_warning"] = stats.get("gpu_memory_mb", 0) > 8000  # >8GB
        stats["cpu_memory_warning"] = stats.get("cpu_memory_mb", 0) > 4000  # >4GB

        if stats.get("gpu_memory_warning"):
            logger.warning(f"High GPU memory usage: {stats['gpu_memory_mb']:.1f}MB")

        if stats.get("cpu_memory_warning"):
            logger.warning(f"High CPU memory usage: {stats['cpu_memory_mb']:.1f}MB")

        return stats

    def cleanup(self) -> None:
        """Perform full memory cleanup."""
        logger.info("Starting resource cleanup")

        # Clear Python garbage
        gc.collect()

        # Clear GPU cache
        self.gpu_manager.clear_gpu_cache()

        # Reset text validator counter
        self.text_validator.reset_counter()

        # Log final stats
        final_stats = self.check_resources()
        logger.info(f"Cleanup complete. Memory: {final_stats}")


# Convenience function
def enforce_text_limits(
    text: str,
    max_length: int = 50000,
    truncate: bool = True
) -> str:
    """
    Quick function to enforce text length limits.

    Args:
        text: Input text
        max_length: Maximum allowed length
        truncate: If True, truncate; if False, raise error

    Returns:
        Validated text
    """
    if len(text) <= max_length:
        return text

    if truncate:
        logger.warning(f"Truncating text from {len(text)} to {max_length} chars")
        return text[:max_length]
    else:
        raise ValueError(f"Text length {len(text)} exceeds maximum {max_length}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Test resource monitoring
    monitor = ResourceMonitor()
    stats = monitor.check_resources()
    print("Resource Stats:", stats)

    # Test text validation
    validator = TextValidator()
    long_text = "x" * 60000
    processed, info = validator.validate_text(long_text)
    print(f"Validation info: {info}")

    # Test cleanup
    monitor.cleanup()
