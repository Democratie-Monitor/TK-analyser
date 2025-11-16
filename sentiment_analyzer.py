"""
Dutch Sentiment Analysis Module for TK-Analyser

Supports multiple sentiment analysis backends:
- RobBERT v2 Dutch Sentiment (transformer-based, highest accuracy ~93%)
- BERTje (Dutch BERT model)
- Multilingual BERT sentiment (supports 6 languages including Dutch)
- Pattern.nl (rule-based, fast, lightweight)
- TextBlob-NL (rule-based)

Each backend can be evaluated and compared for performance on Dutch parliamentary speeches.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import time
import warnings

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Result from sentiment analysis."""
    label: str  # positive, negative, neutral
    score: float  # confidence/probability (0-1)
    raw_scores: Dict[str, float]  # all class probabilities
    processing_time: float
    backend: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DetailedSentiment:
    """Detailed sentiment analysis with multiple metrics."""
    polarity: float  # -1 to 1
    subjectivity: float  # 0 to 1
    label: str  # positive/negative/neutral
    confidence: float  # 0 to 1
    intensity: str  # weak/moderate/strong
    processing_time: float
    backend: str
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if result['metadata'] is None:
            result['metadata'] = {}
        return result


class SentimentBackend(ABC):
    """Abstract base class for sentiment analysis backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the backend."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of the backend."""
        pass

    @abstractmethod
    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of text."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend dependencies are installed."""
        pass

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze multiple texts. Override for batch optimization."""
        return [self.analyze(text) for text in texts]


class RobBERTSentiment(SentimentBackend):
    """
    RobBERT v2 Dutch Sentiment - State-of-the-art Dutch sentiment model.
    ~93% accuracy on Dutch Book Reviews Dataset.
    """

    def __init__(self, model_name: str = "DTAI-KULeuven/robbert-v2-dutch-sentiment"):
        self.model_name = model_name
        self._pipeline = None
        self._load_model()

    @property
    def name(self) -> str:
        return "robbert-v2-dutch-sentiment"

    @property
    def description(self) -> str:
        return "RobBERT v2 Dutch Sentiment - Transformer model with ~93% accuracy"

    def _load_model(self) -> None:
        """Load the model pipeline."""
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name
            )
            logger.info(f"Loaded {self.model_name}")
        except ImportError:
            logger.warning("transformers library not installed")
            self._pipeline = None
        except Exception as e:
            logger.error(f"Failed to load RobBERT model: {e}")
            self._pipeline = None

    def is_available(self) -> bool:
        return self._pipeline is not None

    def analyze(self, text: str) -> SentimentResult:
        if not self.is_available():
            return SentimentResult(
                label="error",
                score=0.0,
                raw_scores={},
                processing_time=0.0,
                backend=self.name,
                error="Model not available"
            )

        if not text or not text.strip():
            logger.warning(f"{self.name}: Empty text provided")
            return SentimentResult(
                label="neutral",
                score=0.5,
                raw_scores={},
                processing_time=0.0,
                backend=self.name,
                error="Empty text"
            )

        start_time = time.time()
        try:
            # Truncate text if too long (BERT has 512 token limit)
            original_length = len(text)
            truncated_text = text[:4000]  # Rough char limit

            if original_length > 4000:
                logger.debug(f"{self.name}: Truncated text from {original_length} to 4000 chars")

            logger.debug(f"{self.name}: Analyzing text ({len(truncated_text)} chars)")
            result = self._pipeline(truncated_text)[0]
            processing_time = time.time() - start_time

            # Map labels to standard format
            original_label = result['label']
            label = original_label.lower()
            if label in ['positive', 'pos', '1']:
                label = 'positive'
            elif label in ['negative', 'neg', '0']:
                label = 'negative'
            else:
                label = 'neutral'
                logger.debug(f"{self.name}: Mapped unknown label '{original_label}' to 'neutral'")

            logger.debug(f"{self.name}: Result={label}, score={result['score']:.3f}, time={processing_time:.3f}s")

            return SentimentResult(
                label=label,
                score=result['score'],
                raw_scores={result['label']: result['score']},
                processing_time=processing_time,
                backend=self.name
            )

        except Exception as e:
            logger.error(f"RobBERT analysis error: {e}")
            return SentimentResult(
                label="error",
                score=0.0,
                raw_scores={},
                processing_time=time.time() - start_time,
                backend=self.name,
                error=str(e)
            )

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Optimized batch processing."""
        if not self.is_available():
            return [self.analyze("") for _ in texts]

        if not texts:
            logger.warning("Empty text list provided for batch analysis")
            return []

        start_time = time.time()
        try:
            # Truncate texts
            truncated = [t[:4000] for t in texts]
            logger.debug(f"Batch processing {len(texts)} texts")
            results = self._pipeline(truncated)
            total_time = time.time() - start_time
            avg_time = total_time / len(texts)
            logger.info(f"Batch analysis completed: {len(texts)} texts in {total_time:.2f}s ({avg_time:.3f}s/text)")

            sentiment_results = []
            for result in results:
                label = result['label'].lower()
                if label in ['positive', 'pos', '1']:
                    label = 'positive'
                elif label in ['negative', 'neg', '0']:
                    label = 'negative'
                else:
                    label = 'neutral'

                sentiment_results.append(SentimentResult(
                    label=label,
                    score=result['score'],
                    raw_scores={result['label']: result['score']},
                    processing_time=avg_time,
                    backend=self.name
                ))

            return sentiment_results

        except Exception as e:
            logger.error(f"Batch analysis error: {e}")
            return [self.analyze(text) for text in texts]


class BERTjeSentiment(SentimentBackend):
    """
    BERTje - Dutch BERT model fine-tuned for sentiment.
    Good performance on Dutch NLP tasks.
    """

    def __init__(self, model_name: str = "wietsedv/bert-base-dutch-cased-finetuned-sentiment"):
        self.model_name = model_name
        self._pipeline = None
        self._load_model()

    @property
    def name(self) -> str:
        return "bertje-dutch-sentiment"

    @property
    def description(self) -> str:
        return "BERTje - Dutch BERT model fine-tuned for sentiment analysis"

    def _load_model(self) -> None:
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name
            )
            logger.info(f"Loaded {self.model_name}")
        except ImportError:
            logger.warning("transformers library not installed")
        except Exception as e:
            logger.warning(f"BERTje model not available: {e}")
            self._pipeline = None

    def is_available(self) -> bool:
        return self._pipeline is not None

    def analyze(self, text: str) -> SentimentResult:
        if not self.is_available():
            return SentimentResult(
                label="error",
                score=0.0,
                raw_scores={},
                processing_time=0.0,
                backend=self.name,
                error="Model not available"
            )

        start_time = time.time()
        try:
            truncated_text = text[:4000]
            result = self._pipeline(truncated_text)[0]
            processing_time = time.time() - start_time

            label = result['label'].lower()
            if 'pos' in label or label == '1':
                label = 'positive'
            elif 'neg' in label or label == '0':
                label = 'negative'
            else:
                label = 'neutral'

            return SentimentResult(
                label=label,
                score=result['score'],
                raw_scores={result['label']: result['score']},
                processing_time=processing_time,
                backend=self.name
            )

        except Exception as e:
            logger.error(f"BERTje analysis error: {e}")
            return SentimentResult(
                label="error",
                score=0.0,
                raw_scores={},
                processing_time=time.time() - start_time,
                backend=self.name,
                error=str(e)
            )


class MultilingualBERTSentiment(SentimentBackend):
    """
    Multilingual BERT sentiment - Supports Dutch + 5 other languages.
    Returns star rating (1-5) which is converted to sentiment.
    """

    def __init__(self, model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment"):
        self.model_name = model_name
        self._pipeline = None
        self._load_model()

    @property
    def name(self) -> str:
        return "multilingual-bert-sentiment"

    @property
    def description(self) -> str:
        return "Multilingual BERT - 5-star sentiment for Dutch/English/German/French/Spanish/Italian"

    def _load_model(self) -> None:
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name
            )
            logger.info(f"Loaded {self.model_name}")
        except ImportError:
            logger.warning("transformers library not installed")
        except Exception as e:
            logger.warning(f"Multilingual BERT not available: {e}")
            self._pipeline = None

    def is_available(self) -> bool:
        return self._pipeline is not None

    def analyze(self, text: str) -> SentimentResult:
        if not self.is_available():
            return SentimentResult(
                label="error",
                score=0.0,
                raw_scores={},
                processing_time=0.0,
                backend=self.name,
                error="Model not available"
            )

        start_time = time.time()
        try:
            truncated_text = text[:4000]
            result = self._pipeline(truncated_text)[0]
            processing_time = time.time() - start_time

            # Convert star rating to sentiment
            # Label format: "1 star", "2 stars", etc.
            stars = int(result['label'].split()[0])

            if stars >= 4:
                label = 'positive'
            elif stars <= 2:
                label = 'negative'
            else:
                label = 'neutral'

            return SentimentResult(
                label=label,
                score=result['score'],
                raw_scores={result['label']: result['score'], 'stars': stars},
                processing_time=processing_time,
                backend=self.name
            )

        except Exception as e:
            logger.error(f"Multilingual BERT analysis error: {e}")
            return SentimentResult(
                label="error",
                score=0.0,
                raw_scores={},
                processing_time=time.time() - start_time,
                backend=self.name,
                error=str(e)
            )


class PatternSentiment(SentimentBackend):
    """
    Pattern.nl - Rule-based sentiment analysis for Dutch.
    Fast and lightweight, uses lexicon of ~4000 Dutch lemmas.
    Lower accuracy than transformer models but much faster.
    """

    def __init__(self):
        self._pattern_available = False
        self._load_pattern()

    @property
    def name(self) -> str:
        return "pattern-nl"

    @property
    def description(self) -> str:
        return "Pattern.nl - Rule-based Dutch sentiment with 4000-word lexicon"

    def _load_pattern(self) -> None:
        try:
            # Pattern can be finicky, suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from pattern.nl import sentiment
                self._sentiment_func = sentiment
                self._pattern_available = True
            logger.info("Loaded Pattern.nl")
        except ImportError:
            logger.warning("Pattern library not installed (pip install pattern)")
            self._pattern_available = False
        except Exception as e:
            logger.warning(f"Pattern.nl not available: {e}")
            self._pattern_available = False

    def is_available(self) -> bool:
        return self._pattern_available

    def analyze(self, text: str) -> SentimentResult:
        if not self.is_available():
            return SentimentResult(
                label="error",
                score=0.0,
                raw_scores={},
                processing_time=0.0,
                backend=self.name,
                error="Pattern library not installed"
            )

        start_time = time.time()
        try:
            # Pattern returns (polarity, subjectivity)
            # Polarity: -1 to 1
            # Subjectivity: 0 to 1
            polarity, subjectivity = self._sentiment_func(text)
            processing_time = time.time() - start_time

            # Convert polarity to label
            if polarity > 0.1:
                label = 'positive'
            elif polarity < -0.1:
                label = 'negative'
            else:
                label = 'neutral'

            # Convert polarity to 0-1 score
            score = (polarity + 1) / 2  # Map -1..1 to 0..1

            return SentimentResult(
                label=label,
                score=score,
                raw_scores={
                    'polarity': polarity,
                    'subjectivity': subjectivity
                },
                processing_time=processing_time,
                backend=self.name
            )

        except Exception as e:
            logger.error(f"Pattern.nl analysis error: {e}")
            return SentimentResult(
                label="error",
                score=0.0,
                raw_scores={},
                processing_time=time.time() - start_time,
                backend=self.name,
                error=str(e)
            )


class TextBlobNLSentiment(SentimentBackend):
    """
    TextBlob-NL - TextBlob with Dutch Pattern analyzer.
    Uses Pattern under the hood but with TextBlob interface.
    """

    def __init__(self):
        self._available = False
        self._load_textblob()

    @property
    def name(self) -> str:
        return "textblob-nl"

    @property
    def description(self) -> str:
        return "TextBlob-NL - TextBlob with Dutch Pattern analyzer"

    def _load_textblob(self) -> None:
        try:
            from textblob import TextBlob
            from textblob_nl import PatternTagger, PatternAnalyzer
            self._TextBlob = TextBlob
            self._PatternTagger = PatternTagger
            self._PatternAnalyzer = PatternAnalyzer
            self._available = True
            logger.info("Loaded TextBlob-NL")
        except ImportError:
            logger.warning("TextBlob-NL not installed (pip install textblob-nl)")
            self._available = False
        except Exception as e:
            logger.warning(f"TextBlob-NL not available: {e}")
            self._available = False

    def is_available(self) -> bool:
        return self._available

    def analyze(self, text: str) -> SentimentResult:
        if not self.is_available():
            return SentimentResult(
                label="error",
                score=0.0,
                raw_scores={},
                processing_time=0.0,
                backend=self.name,
                error="TextBlob-NL not installed"
            )

        start_time = time.time()
        try:
            blob = self._TextBlob(
                text,
                pos_tagger=self._PatternTagger(),
                analyzer=self._PatternAnalyzer()
            )
            sentiment = blob.sentiment
            processing_time = time.time() - start_time

            polarity = sentiment.polarity
            subjectivity = sentiment.subjectivity

            if polarity > 0.1:
                label = 'positive'
            elif polarity < -0.1:
                label = 'negative'
            else:
                label = 'neutral'

            score = (polarity + 1) / 2

            return SentimentResult(
                label=label,
                score=score,
                raw_scores={
                    'polarity': polarity,
                    'subjectivity': subjectivity
                },
                processing_time=processing_time,
                backend=self.name
            )

        except Exception as e:
            logger.error(f"TextBlob-NL analysis error: {e}")
            return SentimentResult(
                label="error",
                score=0.0,
                raw_scores={},
                processing_time=time.time() - start_time,
                backend=self.name,
                error=str(e)
            )


class VaderSentiment(SentimentBackend):
    """
    VADER - Valence Aware Dictionary and sEntiment Reasoner.
    Not specifically for Dutch but can handle some Dutch text.
    Very fast, good for comparison baseline.
    """

    def __init__(self):
        self._analyzer = None
        self._load_vader()

    @property
    def name(self) -> str:
        return "vader"

    @property
    def description(self) -> str:
        return "VADER - Rule-based English sentiment (baseline comparison)"

    def _load_vader(self) -> None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._analyzer = SentimentIntensityAnalyzer()
            logger.info("Loaded VADER")
        except ImportError:
            logger.warning("VADER not installed (pip install vaderSentiment)")
        except Exception as e:
            logger.warning(f"VADER not available: {e}")

    def is_available(self) -> bool:
        return self._analyzer is not None

    def analyze(self, text: str) -> SentimentResult:
        if not self.is_available():
            return SentimentResult(
                label="error",
                score=0.0,
                raw_scores={},
                processing_time=0.0,
                backend=self.name,
                error="VADER not installed"
            )

        start_time = time.time()
        try:
            scores = self._analyzer.polarity_scores(text)
            processing_time = time.time() - start_time

            compound = scores['compound']
            if compound >= 0.05:
                label = 'positive'
            elif compound <= -0.05:
                label = 'negative'
            else:
                label = 'neutral'

            # Normalize compound to 0-1
            score = (compound + 1) / 2

            return SentimentResult(
                label=label,
                score=score,
                raw_scores=scores,
                processing_time=processing_time,
                backend=self.name
            )

        except Exception as e:
            logger.error(f"VADER analysis error: {e}")
            return SentimentResult(
                label="error",
                score=0.0,
                raw_scores={},
                processing_time=time.time() - start_time,
                backend=self.name,
                error=str(e)
            )


class SentimentAnalyzerFactory:
    """Factory for creating sentiment analysis backends."""

    _backends = {
        'robbert': RobBERTSentiment,
        'bertje': BERTjeSentiment,
        'multilingual': MultilingualBERTSentiment,
        'pattern': PatternSentiment,
        'textblob': TextBlobNLSentiment,
        'vader': VaderSentiment
    }

    @classmethod
    def create(cls, backend_name: str, **kwargs) -> SentimentBackend:
        """Create a sentiment backend by name."""
        if backend_name not in cls._backends:
            raise ValueError(f"Unknown backend: {backend_name}. Available: {list(cls._backends.keys())}")
        return cls._backends[backend_name](**kwargs)

    @classmethod
    def available_backends(cls) -> Dict[str, str]:
        """Get list of available backends with their status."""
        status = {}
        for name, backend_class in cls._backends.items():
            try:
                backend = backend_class()
                status[name] = {
                    'available': backend.is_available(),
                    'description': backend.description
                }
            except Exception as e:
                status[name] = {
                    'available': False,
                    'description': str(e)
                }
        return status

    @classmethod
    def get_available(cls) -> List[str]:
        """Get list of actually available backends."""
        available = []
        for name in cls._backends.keys():
            try:
                backend = cls.create(name)
                if backend.is_available():
                    available.append(name)
            except Exception:
                pass
        return available


class DutchSentimentAnalyzer:
    """High-level sentiment analyzer with multiple backend support."""

    def __init__(self, backends: Optional[List[str]] = None):
        """
        Initialize with specified backends.

        Args:
            backends: List of backend names to use. If None, uses all available.
        """
        if backends is None:
            backends = SentimentAnalyzerFactory.get_available()

        self.backends: Dict[str, SentimentBackend] = {}
        for name in backends:
            try:
                backend = SentimentAnalyzerFactory.create(name)
                if backend.is_available():
                    self.backends[name] = backend
                    logger.info(f"Initialized backend: {name}")
                else:
                    logger.warning(f"Backend {name} not available")
            except Exception as e:
                logger.error(f"Failed to initialize {name}: {e}")

        if not self.backends:
            logger.warning("No sentiment backends available")

    def analyze(self, text: str, backend: Optional[str] = None) -> SentimentResult:
        """
        Analyze text with specified or default backend.

        Args:
            text: Text to analyze
            backend: Specific backend to use (None = first available)

        Returns:
            SentimentResult
        """
        if not self.backends:
            return SentimentResult(
                label="error",
                score=0.0,
                raw_scores={},
                processing_time=0.0,
                backend="none",
                error="No backends available"
            )

        if backend:
            if backend not in self.backends:
                return SentimentResult(
                    label="error",
                    score=0.0,
                    raw_scores={},
                    processing_time=0.0,
                    backend=backend,
                    error=f"Backend {backend} not available"
                )
            return self.backends[backend].analyze(text)

        # Use first available
        backend_name = list(self.backends.keys())[0]
        return self.backends[backend_name].analyze(text)

    def analyze_with_all(self, text: str) -> Dict[str, SentimentResult]:
        """
        Analyze text with all available backends.

        Args:
            text: Text to analyze

        Returns:
            Dict mapping backend name to result
        """
        results = {}
        for name, backend in self.backends.items():
            results[name] = backend.analyze(text)
        return results

    def analyze_batch(
        self,
        texts: List[str],
        backend: Optional[str] = None
    ) -> List[SentimentResult]:
        """
        Analyze multiple texts with specified backend.

        Args:
            texts: List of texts
            backend: Backend to use

        Returns:
            List of results
        """
        if not self.backends:
            return [self.analyze(t) for t in texts]

        if backend and backend in self.backends:
            return self.backends[backend].analyze_batch(texts)

        backend_name = list(self.backends.keys())[0]
        return self.backends[backend_name].analyze_batch(texts)

    def get_consensus(self, text: str) -> DetailedSentiment:
        """
        Get consensus sentiment from all backends.

        Args:
            text: Text to analyze

        Returns:
            DetailedSentiment with aggregated results
        """
        start_time = time.time()
        all_results = self.analyze_with_all(text)

        if not all_results:
            return DetailedSentiment(
                polarity=0.0,
                subjectivity=0.0,
                label="neutral",
                confidence=0.0,
                intensity="none",
                processing_time=0.0,
                backend="consensus"
            )

        # Count labels
        label_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        scores = []
        polarities = []

        for name, result in all_results.items():
            if result.error is None:
                label_counts[result.label] = label_counts.get(result.label, 0) + 1
                scores.append(result.score)

                # Extract polarity if available
                if 'polarity' in result.raw_scores:
                    polarities.append(result.raw_scores['polarity'])
                else:
                    # Convert score to polarity
                    polarities.append(result.score * 2 - 1)

        # Determine consensus label
        consensus_label = max(label_counts, key=label_counts.get)
        confidence = label_counts[consensus_label] / len(all_results) if all_results else 0

        # Average polarity
        avg_polarity = sum(polarities) / len(polarities) if polarities else 0

        # Determine intensity
        abs_polarity = abs(avg_polarity)
        if abs_polarity > 0.6:
            intensity = "strong"
        elif abs_polarity > 0.3:
            intensity = "moderate"
        else:
            intensity = "weak"

        processing_time = time.time() - start_time

        return DetailedSentiment(
            polarity=avg_polarity,
            subjectivity=0.5,  # Default, as not all backends provide this
            label=consensus_label,
            confidence=confidence,
            intensity=intensity,
            processing_time=processing_time,
            backend="consensus",
            metadata={
                'backends_used': list(all_results.keys()),
                'individual_results': {k: v.to_dict() for k, v in all_results.items()},
                'label_agreement': label_counts
            }
        )


def check_available_backends() -> None:
    """Print available sentiment analysis backends."""
    print("Dutch Sentiment Analysis Backends:")
    print("=" * 60)

    status = SentimentAnalyzerFactory.available_backends()

    for name, info in status.items():
        available = "✓" if info['available'] else "✗"
        print(f"{available} {name}: {info['description']}")

    print("=" * 60)

    available = SentimentAnalyzerFactory.get_available()
    if available:
        print(f"\nAvailable backends: {', '.join(available)}")
    else:
        print("\nNo backends available. Install required packages:")
        print("  Transformer models: pip install transformers torch")
        print("  Pattern.nl: pip install pattern")
        print("  TextBlob-NL: pip install textblob-nl")
        print("  VADER: pip install vaderSentiment")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    check_available_backends()

    # Test if any backend works
    available = SentimentAnalyzerFactory.get_available()
    if available:
        print("\nTesting with sample text...")
        analyzer = DutchSentimentAnalyzer()
        test_text = "Dit is een geweldige dag! Ik ben heel blij."
        results = analyzer.analyze_with_all(test_text)

        for name, result in results.items():
            print(f"{name}: {result.label} (score: {result.score:.3f}, time: {result.processing_time:.3f}s)")
