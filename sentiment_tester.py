"""
Sentiment Library Comparison and Testing System

Evaluate and compare different sentiment analysis libraries on Dutch parliamentary speeches.
Provides metrics for:
- Inter-library agreement
- Processing speed
- Consistency across speeches
- Distribution of sentiment scores
"""

import pandas as pd
import json
import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import asdict
from collections import defaultdict
import hashlib

from sentiment_analyzer import (
    DutchSentimentAnalyzer,
    SentimentAnalyzerFactory,
    SentimentResult,
    DetailedSentiment
)

logger = logging.getLogger(__name__)


class SpeechSample:
    """A speech sample for testing."""

    def __init__(
        self,
        id: str,
        speaker_name: str,
        speaker_party: str,
        date: str,
        text: str
    ):
        self.id = id
        self.speaker_name = speaker_name
        self.speaker_party = speaker_party
        self.date = date
        self.text = text
        self.text_hash = hashlib.md5(text.encode()).hexdigest()[:12]

    @classmethod
    def from_row(cls, row: pd.Series, idx: int) -> 'SpeechSample':
        return cls(
            id=f"speech_{idx}",
            speaker_name=row['speaker_name'],
            speaker_party=row['speaker_party'],
            date=row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
            text=row['speech_text']
        )


class SentimentTestResult:
    """Result of testing one backend on one speech."""

    def __init__(
        self,
        backend_name: str,
        speech_id: str,
        speaker: str,
        party: str,
        label: str,
        score: float,
        raw_scores: Dict[str, float],
        processing_time: float,
        error: Optional[str] = None
    ):
        self.backend_name = backend_name
        self.speech_id = speech_id
        self.speaker = speaker
        self.party = party
        self.label = label
        self.score = score
        self.raw_scores = raw_scores
        self.processing_time = processing_time
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            'backend_name': self.backend_name,
            'speech_id': self.speech_id,
            'speaker': self.speaker,
            'party': self.party,
            'label': self.label,
            'score': self.score,
            'raw_scores': self.raw_scores,
            'processing_time': self.processing_time,
            'error': self.error
        }


class SentimentComparison:
    """Compare sentiment results across different backends."""

    def __init__(self, results: Dict[str, List[SentimentTestResult]]):
        """
        Args:
            results: Dict mapping backend_name to list of test results
        """
        self.results = results
        self.backend_names = list(results.keys())

    def calculate_agreement(self) -> Dict[str, Any]:
        """Calculate inter-backend agreement metrics."""
        if len(self.backend_names) < 2:
            return {"error": "Need at least 2 backends to compare"}

        # Get all speech IDs
        speech_ids = set()
        for backend_results in self.results.values():
            for result in backend_results:
                speech_ids.add(result.speech_id)

        if not speech_ids:
            return {"total_speeches": 0}

        # Calculate agreement
        full_agreement = 0
        pairwise_agreements = defaultdict(int)
        label_distributions = {name: defaultdict(int) for name in self.backend_names}

        for speech_id in speech_ids:
            # Get labels for this speech from all backends
            speech_labels = {}
            for backend_name in self.backend_names:
                for result in self.results[backend_name]:
                    if result.speech_id == speech_id and result.error is None:
                        speech_labels[backend_name] = result.label
                        label_distributions[backend_name][result.label] += 1
                        break

            if len(speech_labels) < 2:
                continue

            # Check full agreement
            labels = list(speech_labels.values())
            if len(set(labels)) == 1:
                full_agreement += 1

            # Pairwise agreement
            backends = list(speech_labels.keys())
            for i in range(len(backends)):
                for j in range(i + 1, len(backends)):
                    pair = tuple(sorted([backends[i], backends[j]]))
                    if speech_labels[backends[i]] == speech_labels[backends[j]]:
                        pairwise_agreements[pair] += 1

        n_speeches = len(speech_ids)

        # Calculate pairwise agreement percentages
        pairwise_pct = {}
        for pair, count in pairwise_agreements.items():
            pairwise_pct[f"{pair[0]}_vs_{pair[1]}"] = count / n_speeches

        return {
            "total_speeches": n_speeches,
            "full_agreement_rate": full_agreement / n_speeches,
            "pairwise_agreement": pairwise_pct,
            "label_distributions": {
                name: dict(dist) for name, dist in label_distributions.items()
            }
        }

    def calculate_performance(self) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics for each backend."""
        performance = {}

        for backend_name, results in self.results.items():
            if not results:
                continue

            successful = [r for r in results if r.error is None]
            if not successful:
                continue

            total_time = sum(r.processing_time for r in successful)
            scores = [r.score for r in successful]

            # Calculate score statistics
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)

            # Score variance
            variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
            std_dev = variance ** 0.5

            performance[backend_name] = {
                "total_analyses": len(results),
                "successful": len(successful),
                "failed": len(results) - len(successful),
                "success_rate": len(successful) / len(results),
                "total_time": total_time,
                "avg_time_per_analysis": total_time / len(successful),
                "avg_score": avg_score,
                "min_score": min_score,
                "max_score": max_score,
                "score_std_dev": std_dev,
                "positive_rate": sum(1 for r in successful if r.label == 'positive') / len(successful),
                "negative_rate": sum(1 for r in successful if r.label == 'negative') / len(successful),
                "neutral_rate": sum(1 for r in successful if r.label == 'neutral') / len(successful)
            }

        return performance

    def analyze_by_party(self) -> Dict[str, Dict[str, Any]]:
        """Analyze sentiment patterns by political party."""
        party_analysis = defaultdict(lambda: defaultdict(list))

        for backend_name, results in self.results.items():
            for result in results:
                if result.error is None:
                    party_analysis[result.party][backend_name].append({
                        'label': result.label,
                        'score': result.score
                    })

        # Aggregate party statistics
        party_stats = {}
        for party, backend_data in party_analysis.items():
            party_stats[party] = {}
            for backend_name, analyses in backend_data.items():
                scores = [a['score'] for a in analyses]
                labels = [a['label'] for a in analyses]

                party_stats[party][backend_name] = {
                    'count': len(analyses),
                    'avg_score': sum(scores) / len(scores) if scores else 0,
                    'positive_pct': labels.count('positive') / len(labels) if labels else 0,
                    'negative_pct': labels.count('negative') / len(labels) if labels else 0,
                    'neutral_pct': labels.count('neutral') / len(labels) if labels else 0
                }

        return party_stats

    def get_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive comparison report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "backends_compared": self.backend_names,
            "agreement_metrics": self.calculate_agreement(),
            "performance_metrics": self.calculate_performance(),
            "party_analysis": self.analyze_by_party()
        }


class SentimentTester:
    """Test and compare sentiment analysis backends."""

    def __init__(self, backends: Optional[List[str]] = None):
        """
        Args:
            backends: List of backend names to test. None = all available.
        """
        if backends is None:
            backends = SentimentAnalyzerFactory.get_available()

        self.analyzer = DutchSentimentAnalyzer(backends)
        self.test_results: Dict[str, List[SentimentTestResult]] = defaultdict(list)

        if not self.analyzer.backends:
            logger.warning("No sentiment backends available for testing")

    def load_test_samples(
        self,
        csv_path: str,
        n_samples: int = 10,
        min_length: int = 100,
        random_state: int = 42
    ) -> List[SpeechSample]:
        """Load speech samples for testing."""
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Filter by length
        valid_df = df[df['speech_text'].str.len() >= min_length]

        if len(valid_df) < n_samples:
            logger.warning(f"Only {len(valid_df)} speeches available")
            n_samples = len(valid_df)

        sample_df = valid_df.sample(n=n_samples, random_state=random_state)

        samples = []
        for idx, row in sample_df.iterrows():
            samples.append(SpeechSample.from_row(row, idx))

        logger.info(f"Loaded {len(samples)} test samples")
        return samples

    def test_backend(
        self,
        backend_name: str,
        samples: List[SpeechSample]
    ) -> List[SentimentTestResult]:
        """Test a single backend on all samples."""
        if backend_name not in self.analyzer.backends:
            logger.error(f"Backend {backend_name} not available")
            return []

        results = []
        logger.info(f"Testing {backend_name} on {len(samples)} samples")

        for sample in samples:
            sentiment_result = self.analyzer.analyze(sample.text, backend_name)

            test_result = SentimentTestResult(
                backend_name=backend_name,
                speech_id=sample.id,
                speaker=sample.speaker_name,
                party=sample.speaker_party,
                label=sentiment_result.label,
                score=sentiment_result.score,
                raw_scores=sentiment_result.raw_scores,
                processing_time=sentiment_result.processing_time,
                error=sentiment_result.error
            )

            results.append(test_result)
            self.test_results[backend_name].append(test_result)

        return results

    def run_comparison_test(
        self,
        samples: List[SpeechSample],
        backends: Optional[List[str]] = None
    ) -> SentimentComparison:
        """
        Test multiple backends on the same samples.

        Args:
            samples: Speech samples to test
            backends: Specific backends to test (None = all available)

        Returns:
            SentimentComparison with results
        """
        if backends is None:
            backends = list(self.analyzer.backends.keys())

        results = {}
        for backend_name in backends:
            if backend_name in self.analyzer.backends:
                results[backend_name] = self.test_backend(backend_name, samples)

        return SentimentComparison(results)

    def save_results(self, output_path: str) -> None:
        """Save all test results to JSON."""
        output = {
            "timestamp": datetime.now().isoformat(),
            "backends_tested": list(self.test_results.keys()),
            "results": {
                name: [r.to_dict() for r in results]
                for name, results in self.test_results.items()
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.info(f"Results saved to {output_path}")

    def generate_report(self, comparison: SentimentComparison) -> str:
        """Generate human-readable comparison report."""
        report = comparison.get_summary_report()

        lines = [
            "=" * 70,
            "SENTIMENT ANALYSIS LIBRARY COMPARISON REPORT",
            "=" * 70,
            f"Timestamp: {report['timestamp']}",
            f"Backends compared: {', '.join(report['backends_compared'])}",
            "",
            "AGREEMENT METRICS:",
            f"  Total speeches analyzed: {report['agreement_metrics']['total_speeches']}",
            f"  Full agreement rate: {report['agreement_metrics']['full_agreement_rate']:.2%}",
            "",
            "  Pairwise Agreement:"
        ]

        for pair, rate in report['agreement_metrics'].get('pairwise_agreement', {}).items():
            lines.append(f"    {pair}: {rate:.2%}")

        lines.extend([
            "",
            "LABEL DISTRIBUTIONS:"
        ])

        for backend, dist in report['agreement_metrics'].get('label_distributions', {}).items():
            total = sum(dist.values())
            lines.append(f"\n  {backend}:")
            for label, count in sorted(dist.items()):
                pct = count / total if total > 0 else 0
                lines.append(f"    {label}: {count} ({pct:.1%})")

        lines.extend([
            "",
            "PERFORMANCE METRICS:"
        ])

        for backend, metrics in report['performance_metrics'].items():
            lines.extend([
                f"\n  {backend}:",
                f"    Success rate: {metrics['success_rate']:.2%}",
                f"    Avg time/analysis: {metrics['avg_time_per_analysis']:.4f}s",
                f"    Avg sentiment score: {metrics['avg_score']:.3f}",
                f"    Score std dev: {metrics['score_std_dev']:.3f}",
                f"    Positive rate: {metrics['positive_rate']:.1%}",
                f"    Negative rate: {metrics['negative_rate']:.1%}",
                f"    Neutral rate: {metrics['neutral_rate']:.1%}"
            ])

        # Top parties by average sentiment
        lines.extend([
            "",
            "SENTIMENT BY PARTY (sample):"
        ])

        party_data = report.get('party_analysis', {})
        if party_data:
            # Get first backend for comparison
            first_backend = report['backends_compared'][0] if report['backends_compared'] else None
            if first_backend:
                party_scores = []
                for party, backend_data in party_data.items():
                    if first_backend in backend_data:
                        party_scores.append((
                            party,
                            backend_data[first_backend]['avg_score'],
                            backend_data[first_backend]['count']
                        ))

                party_scores.sort(key=lambda x: -x[1])

                for party, score, count in party_scores[:10]:
                    lines.append(f"    {party}: {score:.3f} (n={count})")

        lines.extend(["", "=" * 70])

        return "\n".join(lines)


def main():
    """Run sentiment library comparison."""
    import argparse

    parser = argparse.ArgumentParser(description='Compare sentiment analysis libraries')
    parser.add_argument('--csv', type=str, default='data/apb/apb_speeches.csv',
                        help='Path to speech data')
    parser.add_argument('--samples', type=int, default=10,
                        help='Number of samples to test')
    parser.add_argument('--min-length', type=int, default=200,
                        help='Minimum speech length')
    parser.add_argument('--backends', type=str, nargs='+', default=None,
                        help='Specific backends to test')
    parser.add_argument('--output', type=str, default='sentiment_test_results.json',
                        help='Output file for results')
    parser.add_argument('--list-backends', action='store_true',
                        help='List available backends and exit')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if args.list_backends:
        from sentiment_analyzer import check_available_backends
        check_available_backends()
        return

    # Initialize tester
    tester = SentimentTester(args.backends)

    if not tester.analyzer.backends:
        print("No sentiment backends available. Install required packages:")
        print("  pip install transformers torch  # For RobBERT, BERTje, Multilingual BERT")
        print("  pip install pattern              # For Pattern.nl")
        print("  pip install textblob-nl          # For TextBlob-NL")
        print("  pip install vaderSentiment       # For VADER (baseline)")
        return

    # Load samples
    samples = tester.load_test_samples(
        args.csv,
        n_samples=args.samples,
        min_length=args.min_length
    )

    # Run comparison
    print(f"\nTesting {len(tester.analyzer.backends)} backends on {len(samples)} samples...")
    comparison = tester.run_comparison_test(samples)

    # Generate and print report
    report = tester.generate_report(comparison)
    print(report)

    # Save results
    tester.save_results(args.output)

    # Save report
    report_path = args.output.replace('.json', '_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    # Save detailed JSON report
    summary = comparison.get_summary_report()
    summary_path = args.output.replace('.json', '_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
