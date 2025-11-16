"""
TK-Analyser v2 - Claude API Edition

Robust delegitimation analysis using Claude API with:
- Multiple output modes (concise/standard/elaborate)
- Comprehensive error handling with retries
- Token usage tracking and cost estimation
- Batch processing with progress tracking
- Detailed statistics and reporting
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

from claude_processor import ClaudeProcessor, OutputMode, AnalysisResult
from prompt_tester import PromptConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnalysisConfig:
    """Configuration for analysis run."""

    def __init__(
        self,
        csv_path: str = "data/apb/apb_speeches.csv",
        prompt_path: str = "prompts/variants/standard.yaml",
        output_dir: str = "analysis_output",
        output_mode: str = "standard",
        min_length: int = 50,
        sample_size: int = 0,
        random_state: int = 42,
        max_retries: int = 3,
        temperature: float = 0.1,
        save_intermediate: bool = True,
        batch_size: int = 10
    ):
        """Initialize configuration."""
        self.csv_path = csv_path
        self.prompt_path = prompt_path
        self.output_dir = output_dir
        self.output_mode = OutputMode(output_mode)
        self.min_length = min_length
        self.sample_size = sample_size
        self.random_state = random_state
        self.max_retries = max_retries
        self.temperature = temperature
        self.save_intermediate = save_intermediate
        self.batch_size = batch_size

    @classmethod
    def from_yaml(cls, path: str) -> 'AnalysisConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "csv_path": self.csv_path,
            "prompt_path": self.prompt_path,
            "output_dir": self.output_dir,
            "output_mode": self.output_mode.value,
            "min_length": self.min_length,
            "sample_size": self.sample_size,
            "random_state": self.random_state,
            "max_retries": self.max_retries,
            "temperature": self.temperature,
            "save_intermediate": self.save_intermediate,
            "batch_size": self.batch_size
        }


class TKAnalyzer:
    """Main analyzer for Dutch parliamentary speeches."""

    def __init__(self, config: AnalysisConfig):
        """
        Initialize the analyzer.

        Args:
            config: Analysis configuration
        """
        self.config = config
        self.processor = ClaudeProcessor(max_retries=config.max_retries)
        self.prompt_config = PromptConfig.from_yaml(config.prompt_path)
        self.df: Optional[pd.DataFrame] = None
        self.results: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def load_data(self) -> None:
        """Load speech data from CSV file(s)."""
        logger.info(f"Loading data from {self.config.csv_path}")

        # Handle single file or directory of files
        if os.path.isfile(self.config.csv_path):
            speech_files = [self.config.csv_path]
        else:
            speech_files = glob.glob(os.path.join(self.config.csv_path, 'speeches_*.csv'))
            if not speech_files:
                speech_files = glob.glob(os.path.join(self.config.csv_path, '*.csv'))

        if not speech_files:
            raise FileNotFoundError(f"No CSV files found at {self.config.csv_path}")

        # Load and combine all CSVs
        all_dfs = []
        for file_path in speech_files:
            logger.info(f"Loading {file_path}")
            df = pd.read_csv(file_path)
            df['source_file'] = os.path.basename(file_path)
            all_dfs.append(df)

        self.df = pd.concat(all_dfs, ignore_index=True)

        # Convert date column
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')

        # Filter by minimum length
        original_count = len(self.df)
        self.df = self.df[self.df['speech_text'].str.len() >= self.config.min_length]

        logger.info(
            f"Loaded {len(self.df)} speeches "
            f"(filtered {original_count - len(self.df)} short speeches)"
        )

    def analyze_single_speech(self, speech_row: pd.Series) -> Dict[str, Any]:
        """
        Analyze a single speech.

        Args:
            speech_row: DataFrame row containing speech data

        Returns:
            Analysis result dictionary
        """
        speech_text = speech_row['speech_text']

        result = self.processor.process_text(
            text=speech_text,
            system_prompt=self.prompt_config.system_prompt,
            user_prompt=self.prompt_config.user_prompt,
            output_mode=self.config.output_mode,
            temperature=self.config.temperature
        )

        # Build result structure
        analysis_entry = {
            "speech_metadata": {
                "speaker_name": speech_row.get('speaker_name', 'Unknown'),
                "speaker_party": speech_row.get('speaker_party', 'Unknown'),
                "date": speech_row['date'].strftime('%Y-%m-%d') if pd.notna(speech_row.get('date')) else 'Unknown',
                "text_length": len(speech_text),
                "source_file": speech_row.get('source_file', 'Unknown')
            },
            "analysis_metadata": {
                "success": result.success,
                "retries": result.retries,
                "response_time": result.response_time,
                "token_usage": result.token_usage.to_dict(),
                "error": result.error
            },
            "findings": result.data
        }

        return analysis_entry

    def run_sample_analysis(self) -> Dict[str, Any]:
        """
        Run analysis on a random sample of speeches.

        Returns:
            Complete analysis results
        """
        if self.df is None:
            self.load_data()

        n_samples = self.config.sample_size
        if n_samples <= 0:
            logger.warning("Sample size must be > 0, using 10")
            n_samples = 10

        if len(self.df) < n_samples:
            logger.warning(f"Only {len(self.df)} speeches available, reducing sample size")
            n_samples = len(self.df)

        # Sample speeches
        sample_df = self.df.sample(n=n_samples, random_state=self.config.random_state)
        logger.info(f"Analyzing {n_samples} random speeches")

        # Analyze each speech
        self.results = []
        self.errors = []

        for idx, (_, row) in enumerate(tqdm(sample_df.iterrows(), total=n_samples, desc="Analyzing speeches")):
            try:
                result = self.analyze_single_speech(row)
                self.results.append(result)

                if not result['analysis_metadata']['success']:
                    self.errors.append({
                        "index": idx,
                        "speaker": row.get('speaker_name', 'Unknown'),
                        "error": result['analysis_metadata']['error']
                    })

                # Save intermediate results periodically
                if self.config.save_intermediate and (idx + 1) % self.config.batch_size == 0:
                    self._save_intermediate_results()

            except Exception as e:
                logger.error(f"Failed to analyze speech {idx}: {e}")
                self.errors.append({
                    "index": idx,
                    "speaker": row.get('speaker_name', 'Unknown'),
                    "error": str(e)
                })

        return self._compile_results()

    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run analysis on all speeches.

        Returns:
            Complete analysis results
        """
        if self.df is None:
            self.load_data()

        logger.info(f"Analyzing all {len(self.df)} speeches")

        self.results = []
        self.errors = []

        for idx, (_, row) in enumerate(tqdm(self.df.iterrows(), total=len(self.df), desc="Full analysis")):
            try:
                result = self.analyze_single_speech(row)
                self.results.append(result)

                if not result['analysis_metadata']['success']:
                    self.errors.append({
                        "index": idx,
                        "speaker": row.get('speaker_name', 'Unknown'),
                        "error": result['analysis_metadata']['error']
                    })

                # Save intermediate results
                if self.config.save_intermediate and (idx + 1) % self.config.batch_size == 0:
                    self._save_intermediate_results()

            except Exception as e:
                logger.error(f"Failed to analyze speech {idx}: {e}")
                self.errors.append({
                    "index": idx,
                    "speaker": row.get('speaker_name', 'Unknown'),
                    "error": str(e)
                })

        return self._compile_results()

    def run_party_analysis(self, parties: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run analysis grouped by political party.

        Args:
            parties: List of party names to analyze (None = all parties)

        Returns:
            Results grouped by party
        """
        if self.df is None:
            self.load_data()

        if parties is None:
            parties = self.df['speaker_party'].unique().tolist()

        logger.info(f"Analyzing speeches from {len(parties)} parties")

        party_results = {}

        for party in parties:
            logger.info(f"Processing party: {party}")
            party_df = self.df[self.df['speaker_party'] == party]

            if len(party_df) == 0:
                logger.warning(f"No speeches found for party: {party}")
                continue

            party_analyses = []
            for _, row in tqdm(party_df.iterrows(), total=len(party_df), desc=f"{party}"):
                try:
                    result = self.analyze_single_speech(row)
                    party_analyses.append(result)
                except Exception as e:
                    logger.error(f"Error analyzing {party} speech: {e}")

            party_results[party] = {
                "total_speeches": len(party_df),
                "analyzed": len(party_analyses),
                "analyses": party_analyses
            }

        return {
            "timestamp": datetime.now().isoformat(),
            "config": self.config.to_dict(),
            "prompt_used": self.prompt_config.name,
            "party_results": party_results,
            "processor_stats": self.processor.get_statistics()
        }

    def _compile_results(self) -> Dict[str, Any]:
        """Compile all results into final output structure."""
        # Calculate summary statistics
        successful = [r for r in self.results if r['analysis_metadata']['success']]
        total_cases = 0
        all_confidences = []
        type_counts = {}
        target_counts = {}

        for result in successful:
            findings = result['findings']
            cases = findings.get('gevonden_delegitimatie', findings.get('cases', []))
            total_cases += len(cases)

            for case in cases:
                # Track confidence
                if 'confidence' in case:
                    all_confidences.append(case['confidence'])

                # Track types
                case_type = case.get('type', 'unknown')
                type_counts[case_type] = type_counts.get(case_type, 0) + 1

                # Track targets
                target = case.get('doelgroep', case.get('target', 'unknown'))
                target_counts[target] = target_counts.get(target, 0) + 1

        summary = {
            "total_speeches_analyzed": len(self.results),
            "successful_analyses": len(successful),
            "failed_analyses": len(self.errors),
            "success_rate": len(successful) / max(len(self.results), 1),
            "total_cases_found": total_cases,
            "avg_cases_per_speech": total_cases / max(len(successful), 1),
            "avg_confidence": sum(all_confidences) / max(len(all_confidences), 1) if all_confidences else 0,
            "type_distribution": type_counts,
            "target_distribution": target_counts
        }

        return {
            "timestamp": datetime.now().isoformat(),
            "config": self.config.to_dict(),
            "prompt_used": {
                "name": self.prompt_config.name,
                "description": self.prompt_config.description,
                "version": self.prompt_config.version
            },
            "summary": summary,
            "analyses": self.results,
            "errors": self.errors,
            "processor_statistics": self.processor.get_statistics()
        }

    def _save_intermediate_results(self) -> None:
        """Save intermediate results to file."""
        intermediate_path = os.path.join(
            self.config.output_dir,
            f"intermediate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(intermediate_path, 'w', encoding='utf-8') as f:
            json.dump({
                "partial_results": len(self.results),
                "analyses": self.results[-self.config.batch_size:],
                "errors": self.errors
            }, f, ensure_ascii=False, indent=2, default=str)

        logger.debug(f"Saved intermediate results to {intermediate_path}")

    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save final results to file.

        Args:
            results: Results dictionary
            filename: Optional filename (auto-generated if not provided)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            mode = self.config.output_mode.value
            filename = f"analysis_{mode}_{timestamp}.json"

        output_path = os.path.join(self.config.output_dir, filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Results saved to {output_path}")

        # Also save a summary report
        self._save_summary_report(results, output_path)

        return output_path

    def _save_summary_report(self, results: Dict[str, Any], json_path: str) -> None:
        """Generate and save a human-readable summary report."""
        report_path = json_path.replace('.json', '_summary.txt')

        lines = [
            "=" * 70,
            "TK-ANALYSER RESULTS SUMMARY",
            "=" * 70,
            f"Timestamp: {results['timestamp']}",
            f"Prompt: {results['prompt_used']['name']} (v{results['prompt_used']['version']})",
            f"Output Mode: {results['config']['output_mode']}",
            "",
            "ANALYSIS SUMMARY:",
            f"  Total speeches analyzed: {results['summary']['total_speeches_analyzed']}",
            f"  Successful: {results['summary']['successful_analyses']}",
            f"  Failed: {results['summary']['failed_analyses']}",
            f"  Success rate: {results['summary']['success_rate']:.2%}",
            "",
            "FINDINGS:",
            f"  Total delegitimation cases: {results['summary']['total_cases_found']}",
            f"  Average cases per speech: {results['summary']['avg_cases_per_speech']:.2f}",
            f"  Average confidence: {results['summary']['avg_confidence']:.3f}",
            "",
            "TYPE DISTRIBUTION:",
        ]

        for type_name, count in sorted(results['summary']['type_distribution'].items(), key=lambda x: -x[1]):
            lines.append(f"  {type_name}: {count}")

        lines.extend([
            "",
            "TARGET DISTRIBUTION:",
        ])

        for target, count in sorted(results['summary']['target_distribution'].items(), key=lambda x: -x[1]):
            lines.append(f"  {target}: {count}")

        stats = results['processor_statistics']
        lines.extend([
            "",
            "RESOURCE USAGE:",
            f"  Total requests: {stats['request_count']}",
            f"  Total errors: {stats['error_count']}",
            f"  Error rate: {stats['error_rate']:.2%}",
            "",
            "TOKEN USAGE:",
            f"  Input tokens: {stats['token_usage']['input_tokens']:,}",
            f"  Output tokens: {stats['token_usage']['output_tokens']:,}",
            f"  Total tokens: {stats['token_usage']['input_tokens'] + stats['token_usage']['output_tokens']:,}",
            "",
            "ESTIMATED COST:",
            f"  Input: ${stats['estimated_cost']['input_cost']:.4f}",
            f"  Output: ${stats['estimated_cost']['output_cost']:.4f}",
            f"  Total: ${stats['estimated_cost']['total_cost']:.4f}",
            "",
            "=" * 70
        ])

        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))

        logger.info(f"Summary report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Dutch parliamentary speeches for delegitimation patterns using Claude API'
    )

    # Data options
    parser.add_argument('--csv', type=str, default='data/apb/apb_speeches.csv',
                        help='Path to speech data CSV')
    parser.add_argument('--prompt', type=str, default='prompts/variants/standard.yaml',
                        help='Path to prompt YAML file')
    parser.add_argument('--output-dir', type=str, default='analysis_output',
                        help='Output directory')

    # Analysis options
    parser.add_argument('--mode', type=str, choices=['concise', 'standard', 'elaborate'],
                        default='standard', help='Output verbosity mode')
    parser.add_argument('--sample', type=int, default=0,
                        help='Number of random speeches to analyze (0 = all)')
    parser.add_argument('--min-length', type=int, default=50,
                        help='Minimum speech text length')
    parser.add_argument('--parties', type=str, nargs='+', default=None,
                        help='Specific parties to analyze')

    # Processing options
    parser.add_argument('--max-retries', type=int, default=3,
                        help='Maximum API retry attempts')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Model temperature (lower = more deterministic)')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Save intermediate results every N analyses')
    parser.add_argument('--no-intermediate', action='store_true',
                        help='Disable intermediate result saving')

    # Configuration file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML configuration file (overrides other args)')

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = AnalysisConfig.from_yaml(args.config)
    else:
        config = AnalysisConfig(
            csv_path=args.csv,
            prompt_path=args.prompt,
            output_dir=args.output_dir,
            output_mode=args.mode,
            min_length=args.min_length,
            sample_size=args.sample,
            max_retries=args.max_retries,
            temperature=args.temperature,
            save_intermediate=not args.no_intermediate,
            batch_size=args.batch_size
        )

    # Initialize analyzer
    analyzer = TKAnalyzer(config)

    try:
        # Load data
        analyzer.load_data()

        # Run appropriate analysis mode
        if args.parties:
            logger.info(f"Running party-specific analysis for: {args.parties}")
            results = analyzer.run_party_analysis(args.parties)
        elif config.sample_size > 0:
            logger.info(f"Running sample analysis ({config.sample_size} speeches)")
            results = analyzer.run_sample_analysis()
        else:
            logger.info("Running full analysis")
            results = analyzer.run_full_analysis()

        # Save results
        output_path = analyzer.save_results(results)
        logger.info(f"Analysis complete. Results saved to {output_path}")

        # Print summary to console
        print("\n" + "=" * 50)
        print("ANALYSIS COMPLETE")
        print("=" * 50)
        print(f"Total speeches: {results['summary']['total_speeches_analyzed']}")
        print(f"Cases found: {results['summary']['total_cases_found']}")
        print(f"Success rate: {results['summary']['success_rate']:.2%}")
        print(f"Total cost: ${results['processor_statistics']['estimated_cost']['total_cost']:.4f}")
        print(f"Results: {output_path}")
        print("=" * 50)

    except KeyboardInterrupt:
        logger.warning("Analysis interrupted by user")
        if analyzer.results:
            logger.info("Saving partial results...")
            partial_results = analyzer._compile_results()
            analyzer.save_results(partial_results, "interrupted_analysis.json")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
