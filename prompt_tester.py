"""
Prompt Comparison and Testing System for TK-Analyser

Test different prompts on the same speech samples and compare:
- Consistency of findings
- Token usage efficiency
- Cost per analysis
- Inter-prompt agreement
"""

import pandas as pd
import json
import yaml
import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib

from claude_processor import ClaudeProcessor, OutputMode, AnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class PromptConfig:
    """Configuration for a single prompt."""
    name: str
    description: str
    system_prompt: str
    user_prompt: str
    version: str = "1.0"

    @classmethod
    def from_yaml(cls, path: str) -> 'PromptConfig':
        """Load prompt configuration from YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(
            name=data.get('name', os.path.basename(path)),
            description=data.get('description', ''),
            system_prompt=data['system_prompt'],
            user_prompt=data['user_prompt'],
            version=data.get('version', '1.0')
        )


@dataclass
class SpeechSample:
    """A speech sample for testing."""
    id: str
    speaker_name: str
    speaker_party: str
    date: str
    text: str
    text_hash: str  # For verifying same text across tests

    @classmethod
    def from_row(cls, row: pd.Series, idx: int) -> 'SpeechSample':
        """Create from DataFrame row."""
        text = row['speech_text']
        return cls(
            id=f"speech_{idx}",
            speaker_name=row['speaker_name'],
            speaker_party=row['speaker_party'],
            date=row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
            text=text,
            text_hash=hashlib.md5(text.encode()).hexdigest()[:12]
        )


@dataclass
class PromptTestResult:
    """Results from testing one prompt on one speech."""
    prompt_name: str
    speech_id: str
    success: bool
    cases_found: int
    confidence_scores: List[float]
    types_found: List[str]
    targets_found: List[str]
    token_usage: Dict[str, int]
    response_time: float
    cost_estimate: float
    raw_result: Dict[str, Any]
    error: Optional[str] = None


class PromptComparison:
    """Compare results across different prompts."""

    def __init__(self, results: Dict[str, List[PromptTestResult]]):
        """
        Initialize with results from multiple prompts.

        Args:
            results: Dict mapping prompt_name to list of test results
        """
        self.results = results
        self.prompt_names = list(results.keys())

    def calculate_agreement(self) -> Dict[str, float]:
        """
        Calculate inter-prompt agreement on findings.

        Returns percentage of speeches where prompts agree on:
        - Presence/absence of delegitimation
        - Number of cases (within tolerance)
        - Types found
        """
        if len(self.prompt_names) < 2:
            return {"error": "Need at least 2 prompts to compare"}

        # Get all speech IDs
        speech_ids = set()
        for prompt_results in self.results.values():
            for result in prompt_results:
                speech_ids.add(result.speech_id)

        agreements = {
            "presence_agreement": 0.0,
            "count_agreement": 0.0,
            "type_agreement": 0.0,
            "total_speeches": len(speech_ids)
        }

        if not speech_ids:
            return agreements

        presence_agree = 0
        count_agree = 0
        type_agree = 0

        for speech_id in speech_ids:
            # Get results for this speech from all prompts
            speech_results = {}
            for prompt_name in self.prompt_names:
                for result in self.results[prompt_name]:
                    if result.speech_id == speech_id:
                        speech_results[prompt_name] = result
                        break

            if len(speech_results) < 2:
                continue

            # Check presence agreement (all agree on found/not found)
            presence_flags = [r.cases_found > 0 for r in speech_results.values()]
            if all(presence_flags) or not any(presence_flags):
                presence_agree += 1

            # Check count agreement (within 1 case)
            counts = [r.cases_found for r in speech_results.values()]
            if max(counts) - min(counts) <= 1:
                count_agree += 1

            # Check type agreement (same types found)
            type_sets = [set(r.types_found) for r in speech_results.values()]
            if type_sets:
                common_types = type_sets[0]
                for ts in type_sets[1:]:
                    common_types = common_types.intersection(ts)
                # Agreement if overlap > 50%
                all_types = set()
                for ts in type_sets:
                    all_types.update(ts)
                if all_types:
                    overlap = len(common_types) / len(all_types)
                    if overlap >= 0.5:
                        type_agree += 1
                else:
                    type_agree += 1  # No types found by any = agreement

        n = len(speech_ids)
        agreements["presence_agreement"] = presence_agree / n
        agreements["count_agreement"] = count_agree / n
        agreements["type_agreement"] = type_agree / n

        return agreements

    def calculate_efficiency(self) -> Dict[str, Dict[str, float]]:
        """Calculate token efficiency metrics for each prompt."""
        efficiency = {}

        for prompt_name, results in self.results.items():
            if not results:
                continue

            total_input_tokens = sum(r.token_usage.get('input_tokens', 0) for r in results)
            total_output_tokens = sum(r.token_usage.get('output_tokens', 0) for r in results)
            total_tokens = total_input_tokens + total_output_tokens
            total_cases = sum(r.cases_found for r in results)
            total_time = sum(r.response_time for r in results)
            total_cost = sum(r.cost_estimate for r in results)
            success_rate = sum(1 for r in results if r.success) / len(results)

            efficiency[prompt_name] = {
                "total_tokens": total_tokens,
                "avg_tokens_per_analysis": total_tokens / len(results),
                "tokens_per_case": total_tokens / max(total_cases, 1),
                "avg_response_time": total_time / len(results),
                "total_cost": total_cost,
                "avg_cost_per_analysis": total_cost / len(results),
                "cost_per_case": total_cost / max(total_cases, 1),
                "success_rate": success_rate,
                "total_cases_found": total_cases
            }

        return efficiency

    def get_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive comparison report."""
        return {
            "agreement_metrics": self.calculate_agreement(),
            "efficiency_metrics": self.calculate_efficiency(),
            "prompt_summary": {
                name: {
                    "total_analyses": len(results),
                    "total_cases": sum(r.cases_found for r in results),
                    "avg_confidence": sum(
                        sum(r.confidence_scores) / len(r.confidence_scores)
                        for r in results if r.confidence_scores
                    ) / max(sum(1 for r in results if r.confidence_scores), 1),
                    "errors": sum(1 for r in results if r.error)
                }
                for name, results in self.results.items()
            }
        }


class PromptTester:
    """Test and compare different prompts on speech samples."""

    def __init__(
        self,
        processor: ClaudeProcessor,
        prompts_dir: str = "prompts/variants"
    ):
        """
        Initialize the prompt tester.

        Args:
            processor: Claude processor instance
            prompts_dir: Directory containing prompt YAML files
        """
        self.processor = processor
        self.prompts_dir = prompts_dir
        self.prompts: Dict[str, PromptConfig] = {}
        self.test_results: Dict[str, List[PromptTestResult]] = defaultdict(list)
        self._load_prompts()

    def _load_prompts(self) -> None:
        """Load all prompt configurations from the prompts directory."""
        if not os.path.exists(self.prompts_dir):
            logger.warning(f"Prompts directory not found: {self.prompts_dir}")
            return

        for filename in os.listdir(self.prompts_dir):
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                path = os.path.join(self.prompts_dir, filename)
                try:
                    config = PromptConfig.from_yaml(path)
                    self.prompts[config.name] = config
                    logger.info(f"Loaded prompt: {config.name}")
                except Exception as e:
                    logger.error(f"Failed to load prompt from {path}: {e}")

    def load_test_samples(
        self,
        csv_path: str,
        n_samples: int = 10,
        min_length: int = 100,
        random_state: int = 42
    ) -> List[SpeechSample]:
        """
        Load speech samples for testing.

        Args:
            csv_path: Path to CSV file
            n_samples: Number of samples to load
            min_length: Minimum text length
            random_state: Random seed for reproducibility

        Returns:
            List of SpeechSample objects
        """
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])

        # Filter by length
        valid_df = df[df['speech_text'].str.len() >= min_length]

        if len(valid_df) < n_samples:
            logger.warning(f"Only {len(valid_df)} speeches available, reducing sample size")
            n_samples = len(valid_df)

        # Sample
        sample_df = valid_df.sample(n=n_samples, random_state=random_state)

        samples = []
        for idx, row in sample_df.iterrows():
            samples.append(SpeechSample.from_row(row, idx))

        logger.info(f"Loaded {len(samples)} test samples")
        return samples

    def test_prompt(
        self,
        prompt_name: str,
        samples: List[SpeechSample],
        output_mode: OutputMode = OutputMode.STANDARD
    ) -> List[PromptTestResult]:
        """
        Test a single prompt on all samples.

        Args:
            prompt_name: Name of prompt to test
            samples: List of speech samples
            output_mode: Output verbosity mode

        Returns:
            List of test results
        """
        if prompt_name not in self.prompts:
            raise ValueError(f"Unknown prompt: {prompt_name}")

        prompt_config = self.prompts[prompt_name]
        results = []

        for sample in samples:
            logger.info(f"Testing {prompt_name} on {sample.id}")

            # Calculate cost estimate
            self.processor.reset_statistics()

            analysis = self.processor.process_text(
                text=sample.text,
                system_prompt=prompt_config.system_prompt,
                user_prompt=prompt_config.user_prompt,
                output_mode=output_mode
            )

            cost = self.processor.get_cost_estimate()['total_cost']

            # Extract findings
            cases = []
            if analysis.success:
                # Handle different JSON structures
                data = analysis.data
                if 'gevonden_delegitimatie' in data:
                    cases = data['gevonden_delegitimatie']
                elif 'cases' in data:
                    cases = data['cases']

            confidence_scores = []
            types_found = []
            targets_found = []

            for case in cases:
                if 'confidence' in case:
                    confidence_scores.append(case['confidence'])
                if 'type' in case:
                    types_found.append(case['type'])
                if 'doelgroep' in case:
                    targets_found.append(case['doelgroep'])
                elif 'target' in case:
                    targets_found.append(case['target'])

            result = PromptTestResult(
                prompt_name=prompt_name,
                speech_id=sample.id,
                success=analysis.success,
                cases_found=len(cases),
                confidence_scores=confidence_scores,
                types_found=types_found,
                targets_found=targets_found,
                token_usage=analysis.token_usage.to_dict(),
                response_time=analysis.response_time,
                cost_estimate=cost,
                raw_result=analysis.data,
                error=analysis.error
            )

            results.append(result)
            self.test_results[prompt_name].append(result)

        return results

    def run_comparison_test(
        self,
        samples: List[SpeechSample],
        prompt_names: Optional[List[str]] = None,
        output_mode: OutputMode = OutputMode.STANDARD
    ) -> PromptComparison:
        """
        Test multiple prompts on the same samples and compare results.

        Args:
            samples: List of speech samples
            prompt_names: List of prompt names to test (None = all)
            output_mode: Output verbosity mode

        Returns:
            PromptComparison object with results
        """
        if prompt_names is None:
            prompt_names = list(self.prompts.keys())

        results = {}

        for prompt_name in prompt_names:
            logger.info(f"Testing prompt: {prompt_name}")
            results[prompt_name] = self.test_prompt(prompt_name, samples, output_mode)

        return PromptComparison(results)

    def save_results(self, output_path: str) -> None:
        """Save all test results to JSON file."""
        output = {
            "timestamp": datetime.now().isoformat(),
            "prompts_tested": list(self.test_results.keys()),
            "results": {
                prompt_name: [asdict(r) for r in results]
                for prompt_name, results in self.test_results.items()
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Results saved to {output_path}")

    def generate_report(self, comparison: PromptComparison) -> str:
        """Generate a human-readable comparison report."""
        report = comparison.get_summary_report()

        lines = [
            "=" * 60,
            "PROMPT COMPARISON REPORT",
            "=" * 60,
            "",
            "AGREEMENT METRICS:",
            f"  Presence Agreement: {report['agreement_metrics']['presence_agreement']:.2%}",
            f"  Count Agreement: {report['agreement_metrics']['count_agreement']:.2%}",
            f"  Type Agreement: {report['agreement_metrics']['type_agreement']:.2%}",
            "",
            "EFFICIENCY METRICS:",
        ]

        for prompt_name, metrics in report['efficiency_metrics'].items():
            lines.extend([
                f"\n  {prompt_name}:",
                f"    Avg Tokens/Analysis: {metrics['avg_tokens_per_analysis']:.0f}",
                f"    Tokens/Case Found: {metrics['tokens_per_case']:.0f}",
                f"    Avg Response Time: {metrics['avg_response_time']:.2f}s",
                f"    Total Cost: ${metrics['total_cost']:.4f}",
                f"    Cost/Case: ${metrics['cost_per_case']:.4f}",
                f"    Success Rate: {metrics['success_rate']:.2%}",
                f"    Total Cases Found: {metrics['total_cases_found']}"
            ])

        lines.extend([
            "",
            "PROMPT SUMMARY:",
        ])

        for prompt_name, summary in report['prompt_summary'].items():
            lines.extend([
                f"\n  {prompt_name}:",
                f"    Total Analyses: {summary['total_analyses']}",
                f"    Total Cases: {summary['total_cases']}",
                f"    Avg Confidence: {summary['avg_confidence']:.3f}",
                f"    Errors: {summary['errors']}"
            ])

        lines.extend(["", "=" * 60])

        return "\n".join(lines)


def main():
    """Example usage of the prompt tester."""
    import argparse

    parser = argparse.ArgumentParser(description='Test and compare prompts')
    parser.add_argument('--csv', type=str, default='data/apb/apb_speeches.csv',
                        help='Path to speech data')
    parser.add_argument('--samples', type=int, default=5,
                        help='Number of samples to test')
    parser.add_argument('--min-length', type=int, default=200,
                        help='Minimum speech length')
    parser.add_argument('--prompts', type=str, nargs='+', default=None,
                        help='Specific prompts to test')
    parser.add_argument('--output', type=str, default='prompt_test_results.json',
                        help='Output file for results')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Initialize
    processor = ClaudeProcessor()
    tester = PromptTester(processor)

    # Load samples
    samples = tester.load_test_samples(
        args.csv,
        n_samples=args.samples,
        min_length=args.min_length
    )

    # Run comparison
    comparison = tester.run_comparison_test(samples, args.prompts)

    # Generate and print report
    report = tester.generate_report(comparison)
    print(report)

    # Save detailed results
    tester.save_results(args.output)

    # Save report
    report_path = args.output.replace('.json', '_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
