"""
Sentiment Analysis for Dutch Parliamentary Speeches

Combines multiple sentiment analysis approaches:
1. Library-based backends (RobBERT, BERTje, Pattern.nl, etc.)
2. Claude API-based analysis with customizable prompts
3. Comparison and evaluation tools
"""

import pandas as pd
import json
import os
import logging
import argparse
import yaml
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import glob

from claude_processor import ClaudeProcessor, OutputMode
from sentiment_analyzer import (
    DutchSentimentAnalyzer,
    SentimentAnalyzerFactory,
    SentimentResult
)
from prompt_tester import PromptConfig

logger = logging.getLogger(__name__)


class SentimentAnalysisRunner:
    """Run sentiment analysis using multiple backends and/or Claude."""

    def __init__(
        self,
        csv_path: str = "data/apb/apb_speeches.csv",
        output_dir: str = "sentiment_output",
        use_libraries: bool = True,
        use_claude: bool = False,
        library_backends: Optional[List[str]] = None,
        claude_prompt_path: Optional[str] = None,
        min_length: int = 50
    ):
        """
        Initialize sentiment analysis runner.

        Args:
            csv_path: Path to speech data
            output_dir: Output directory
            use_libraries: Use library-based sentiment analyzers
            use_claude: Use Claude API for sentiment analysis
            library_backends: Specific library backends to use
            claude_prompt_path: Path to Claude sentiment prompt
            min_length: Minimum speech length
        """
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.min_length = min_length
        self.df: Optional[pd.DataFrame] = None
        self.results: List[Dict[str, Any]] = []

        # Initialize library analyzers
        self.use_libraries = use_libraries
        self.library_analyzer = None
        if use_libraries:
            if library_backends:
                self.library_analyzer = DutchSentimentAnalyzer(library_backends)
            else:
                self.library_analyzer = DutchSentimentAnalyzer()
            logger.info(f"Initialized library backends: {list(self.library_analyzer.backends.keys())}")

        # Initialize Claude analyzer
        self.use_claude = use_claude
        self.claude_processor = None
        self.claude_prompt = None
        if use_claude:
            self.claude_processor = ClaudeProcessor()
            if claude_prompt_path:
                self.claude_prompt = PromptConfig.from_yaml(claude_prompt_path)
            else:
                # Load default sentiment prompt
                default_path = "prompts/sentiment/standard.yaml"
                if os.path.exists(default_path):
                    self.claude_prompt = PromptConfig.from_yaml(default_path)
            logger.info("Initialized Claude sentiment analyzer")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def load_data(self) -> None:
        """Load speech data."""
        logger.info(f"Loading data from {self.csv_path}")

        if os.path.isfile(self.csv_path):
            speech_files = [self.csv_path]
        else:
            speech_files = glob.glob(os.path.join(self.csv_path, '*.csv'))

        if not speech_files:
            raise FileNotFoundError(f"No CSV files found at {self.csv_path}")

        all_dfs = []
        for file_path in speech_files:
            df = pd.read_csv(file_path)
            df['source_file'] = os.path.basename(file_path)
            all_dfs.append(df)

        self.df = pd.concat(all_dfs, ignore_index=True)

        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')

        # Filter by length
        original_count = len(self.df)
        self.df = self.df[self.df['speech_text'].str.len() >= self.min_length]

        logger.info(f"Loaded {len(self.df)} speeches (filtered {original_count - len(self.df)} short)")

    def analyze_single_speech(self, speech_row: pd.Series) -> Dict[str, Any]:
        """Analyze sentiment of a single speech."""
        speech_text = speech_row['speech_text']

        result = {
            "speech_metadata": {
                "speaker_name": speech_row.get('speaker_name', 'Unknown'),
                "speaker_party": speech_row.get('speaker_party', 'Unknown'),
                "date": speech_row['date'].strftime('%Y-%m-%d') if pd.notna(speech_row.get('date')) else 'Unknown',
                "text_length": len(speech_text),
                "source_file": speech_row.get('source_file', 'Unknown')
            },
            "library_results": {},
            "claude_result": None,
            "consensus": None
        }

        # Library-based analysis
        if self.use_libraries and self.library_analyzer:
            library_results = self.library_analyzer.analyze_with_all(speech_text)
            for backend_name, sentiment_result in library_results.items():
                result["library_results"][backend_name] = sentiment_result.to_dict()

            # Get consensus if multiple backends
            if len(library_results) > 1:
                consensus = self.library_analyzer.get_consensus(speech_text)
                result["consensus"] = consensus.to_dict()

        # Claude-based analysis
        if self.use_claude and self.claude_processor and self.claude_prompt:
            claude_result = self.claude_processor.process_text(
                text=speech_text,
                system_prompt=self.claude_prompt.system_prompt,
                user_prompt=self.claude_prompt.user_prompt,
                output_mode=OutputMode.STANDARD
            )

            result["claude_result"] = {
                "success": claude_result.success,
                "data": claude_result.data,
                "token_usage": claude_result.token_usage.to_dict(),
                "response_time": claude_result.response_time,
                "error": claude_result.error
            }

        return result

    def run_sample_analysis(
        self,
        n_samples: int = 10,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Run sentiment analysis on random sample."""
        if self.df is None:
            self.load_data()

        if len(self.df) < n_samples:
            n_samples = len(self.df)

        sample_df = self.df.sample(n=n_samples, random_state=random_state)
        logger.info(f"Analyzing {n_samples} speeches")

        self.results = []
        for _, row in tqdm(sample_df.iterrows(), total=n_samples, desc="Sentiment analysis"):
            try:
                result = self.analyze_single_speech(row)
                self.results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing speech: {e}")

        return self._compile_results()

    def run_full_analysis(self) -> Dict[str, Any]:
        """Run sentiment analysis on all speeches."""
        if self.df is None:
            self.load_data()

        logger.info(f"Analyzing all {len(self.df)} speeches")

        self.results = []
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Full sentiment analysis"):
            try:
                result = self.analyze_single_speech(row)
                self.results.append(result)
            except Exception as e:
                logger.error(f"Error: {e}")

        return self._compile_results()

    def run_party_analysis(self, parties: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze sentiment by political party."""
        if self.df is None:
            self.load_data()

        if parties is None:
            parties = self.df['speaker_party'].unique().tolist()

        party_results = {}
        for party in parties:
            party_df = self.df[self.df['speaker_party'] == party]
            if len(party_df) == 0:
                continue

            party_analyses = []
            for _, row in tqdm(party_df.iterrows(), total=len(party_df), desc=f"Party: {party}"):
                try:
                    result = self.analyze_single_speech(row)
                    party_analyses.append(result)
                except Exception as e:
                    logger.error(f"Error: {e}")

            party_results[party] = {
                "total_speeches": len(party_df),
                "analyzed": len(party_analyses),
                "analyses": party_analyses
            }

        return {
            "timestamp": datetime.now().isoformat(),
            "party_results": party_results,
            "summary": self._summarize_party_results(party_results)
        }

    def _compile_results(self) -> Dict[str, Any]:
        """Compile analysis results with statistics."""
        # Calculate sentiment distribution across backends
        backend_stats = {}

        # Library backends
        if self.use_libraries:
            for result in self.results:
                for backend_name, backend_result in result.get("library_results", {}).items():
                    if backend_name not in backend_stats:
                        backend_stats[backend_name] = {
                            "labels": [],
                            "scores": [],
                            "times": []
                        }

                    if backend_result.get("error") is None:
                        backend_stats[backend_name]["labels"].append(backend_result["label"])
                        backend_stats[backend_name]["scores"].append(backend_result["score"])
                        backend_stats[backend_name]["times"].append(backend_result["processing_time"])

        # Calculate statistics
        summary = {
            "total_speeches": len(self.results),
            "backends_used": list(backend_stats.keys()),
            "backend_statistics": {}
        }

        for backend_name, stats in backend_stats.items():
            if stats["labels"]:
                n = len(stats["labels"])
                summary["backend_statistics"][backend_name] = {
                    "analyzed": n,
                    "positive_rate": stats["labels"].count("positive") / n,
                    "negative_rate": stats["labels"].count("negative") / n,
                    "neutral_rate": stats["labels"].count("neutral") / n,
                    "avg_score": sum(stats["scores"]) / n,
                    "avg_processing_time": sum(stats["times"]) / n
                }

        # Claude statistics
        if self.use_claude:
            claude_successes = sum(
                1 for r in self.results
                if r.get("claude_result") and r["claude_result"].get("success")
            )
            summary["claude_statistics"] = {
                "analyzed": claude_successes,
                "success_rate": claude_successes / len(self.results) if self.results else 0
            }

            if self.claude_processor:
                summary["claude_token_usage"] = self.claude_processor.get_statistics()

        # Party distribution
        party_sentiments = {}
        for result in self.results:
            party = result["speech_metadata"]["speaker_party"]
            if party not in party_sentiments:
                party_sentiments[party] = {"positive": 0, "negative": 0, "neutral": 0, "count": 0}

            # Use first available backend for party stats
            if result.get("library_results"):
                first_backend = list(result["library_results"].keys())[0]
                label = result["library_results"][first_backend].get("label", "neutral")
                party_sentiments[party][label] = party_sentiments[party].get(label, 0) + 1
                party_sentiments[party]["count"] += 1

        summary["party_sentiment_distribution"] = party_sentiments

        return {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "use_libraries": self.use_libraries,
                "use_claude": self.use_claude,
                "min_length": self.min_length
            },
            "summary": summary,
            "analyses": self.results
        }

    def _summarize_party_results(self, party_results: Dict) -> Dict[str, Any]:
        """Summarize party-level sentiment results."""
        summary = {}
        for party, data in party_results.items():
            if not data["analyses"]:
                continue

            # Aggregate sentiment from first available backend
            labels = []
            scores = []

            for analysis in data["analyses"]:
                if analysis.get("library_results"):
                    first_backend = list(analysis["library_results"].keys())[0]
                    result = analysis["library_results"][first_backend]
                    if result.get("error") is None:
                        labels.append(result["label"])
                        scores.append(result["score"])

            if labels:
                summary[party] = {
                    "count": len(labels),
                    "positive_rate": labels.count("positive") / len(labels),
                    "negative_rate": labels.count("negative") / len(labels),
                    "neutral_rate": labels.count("neutral") / len(labels),
                    "avg_sentiment_score": sum(scores) / len(scores)
                }

        return summary

    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save results to file."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"sentiment_analysis_{timestamp}.json"

        output_path = os.path.join(self.output_dir, filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Results saved to {output_path}")

        # Save summary report
        self._save_summary_report(results, output_path)

        return output_path

    def _save_summary_report(self, results: Dict[str, Any], json_path: str) -> None:
        """Generate and save human-readable summary."""
        report_path = json_path.replace('.json', '_summary.txt')

        lines = [
            "=" * 70,
            "SENTIMENT ANALYSIS RESULTS SUMMARY",
            "=" * 70,
            f"Timestamp: {results['timestamp']}",
            f"Total speeches analyzed: {results['summary']['total_speeches']}",
            f"Backends used: {', '.join(results['summary']['backends_used'])}",
            "",
            "BACKEND STATISTICS:"
        ]

        for backend, stats in results['summary'].get('backend_statistics', {}).items():
            lines.extend([
                f"\n  {backend}:",
                f"    Analyzed: {stats['analyzed']}",
                f"    Positive: {stats['positive_rate']:.1%}",
                f"    Negative: {stats['negative_rate']:.1%}",
                f"    Neutral: {stats['neutral_rate']:.1%}",
                f"    Avg Score: {stats['avg_score']:.3f}",
                f"    Avg Time: {stats['avg_processing_time']:.4f}s"
            ])

        if 'claude_statistics' in results['summary']:
            lines.extend([
                "",
                "CLAUDE STATISTICS:",
                f"  Analyzed: {results['summary']['claude_statistics']['analyzed']}",
                f"  Success Rate: {results['summary']['claude_statistics']['success_rate']:.1%}"
            ])

        lines.extend([
            "",
            "SENTIMENT BY PARTY:"
        ])

        party_dist = results['summary'].get('party_sentiment_distribution', {})
        # Sort by average sentiment (if calculable)
        party_items = []
        for party, data in party_dist.items():
            if data['count'] > 0:
                pos_rate = data['positive'] / data['count']
                party_items.append((party, pos_rate, data['count']))

        party_items.sort(key=lambda x: -x[1])

        for party, pos_rate, count in party_items:
            data = party_dist[party]
            lines.append(
                f"  {party}: Pos {data['positive']}/{count} "
                f"({pos_rate:.1%}), Neg {data['negative']}/{count}, "
                f"Neut {data['neutral']}/{count}"
            )

        lines.extend(["", "=" * 70])

        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))

        logger.info(f"Summary report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Sentiment analysis for Dutch parliamentary speeches')

    # Data options
    parser.add_argument('--csv', type=str, default='data/apb/apb_speeches.csv',
                        help='Path to speech data')
    parser.add_argument('--output-dir', type=str, default='sentiment_output',
                        help='Output directory')

    # Analysis options
    parser.add_argument('--sample', type=int, default=0,
                        help='Number of random samples (0 = all)')
    parser.add_argument('--min-length', type=int, default=50,
                        help='Minimum speech length')
    parser.add_argument('--parties', type=str, nargs='+', default=None,
                        help='Specific parties to analyze')

    # Backend options
    parser.add_argument('--libraries', action='store_true', default=True,
                        help='Use library-based sentiment analyzers')
    parser.add_argument('--no-libraries', action='store_false', dest='libraries',
                        help='Disable library-based analyzers')
    parser.add_argument('--backends', type=str, nargs='+', default=None,
                        help='Specific library backends to use')
    parser.add_argument('--claude', action='store_true',
                        help='Use Claude API for sentiment analysis')
    parser.add_argument('--claude-prompt', type=str, default=None,
                        help='Path to Claude sentiment prompt YAML')

    # Utility options
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

    # Initialize runner
    runner = SentimentAnalysisRunner(
        csv_path=args.csv,
        output_dir=args.output_dir,
        use_libraries=args.libraries,
        use_claude=args.claude,
        library_backends=args.backends,
        claude_prompt_path=args.claude_prompt,
        min_length=args.min_length
    )

    try:
        runner.load_data()

        # Run analysis
        if args.parties:
            logger.info(f"Running party-specific analysis for: {args.parties}")
            results = runner.run_party_analysis(args.parties)
        elif args.sample > 0:
            logger.info(f"Running sample analysis ({args.sample} speeches)")
            results = runner.run_sample_analysis(args.sample)
        else:
            logger.info("Running full analysis")
            results = runner.run_full_analysis()

        # Save results
        output_path = runner.save_results(results)

        # Print summary
        print("\n" + "=" * 50)
        print("SENTIMENT ANALYSIS COMPLETE")
        print("=" * 50)
        print(f"Total speeches: {results['summary']['total_speeches']}")
        print(f"Backends: {', '.join(results['summary']['backends_used'])}")

        if results['summary'].get('backend_statistics'):
            first_backend = list(results['summary']['backend_statistics'].keys())[0]
            stats = results['summary']['backend_statistics'][first_backend]
            print(f"\nUsing {first_backend}:")
            print(f"  Positive: {stats['positive_rate']:.1%}")
            print(f"  Negative: {stats['negative_rate']:.1%}")
            print(f"  Neutral: {stats['neutral_rate']:.1%}")

        print(f"\nResults: {output_path}")
        print("=" * 50)

    except KeyboardInterrupt:
        logger.warning("Analysis interrupted")
        if runner.results:
            results = runner._compile_results()
            runner.save_results(results, "interrupted_sentiment.json")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
