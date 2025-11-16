# Political Speech Analyzer (TK-Analyser)

A robust tool for analyzing Dutch parliamentary speeches for delegitimization patterns and sentiment. Version 2 uses the Claude API (Anthropic) with enhanced features for reliability, cost optimization, and prompt experimentation.

## Features

### Core Features
- **Multiple AI Backends**: Claude API (v2) or Fireworks AI (v1)
- **Robust Error Handling**: Automatic retries with exponential backoff
- **Token Usage Tracking**: Monitor costs in real-time
- **Multiple Output Modes**: Concise (cheap), Standard, or Elaborate (detailed)
- **Prompt Comparison System**: Test different prompts and compare results
- **Sentiment Analysis**: Multiple Dutch-optimized sentiment libraries with comparison tools
- **Batch Processing**: Handle large datasets with progress tracking
- **Configurable Analysis**: YAML-based configuration for all parameters

### Analysis Capabilities
- **Delegitimization Detection**: Identify undermining rhetoric in political discourse
- **Sentiment Analysis**: Score emotional tone using multiple backends
- Sample or comprehensive analysis
- Party-specific analysis
- Confidence scoring with thresholds
- Type and target distribution tracking
- Intermediate result saving for long runs

### Sentiment Analysis Backends
- **RobBERT v2** - Dutch transformer model (~93% accuracy)
- **BERTje** - Dutch BERT fine-tuned for sentiment
- **Multilingual BERT** - Supports 6 languages including Dutch
- **Pattern.nl** - Rule-based, fast (4000-word Dutch lexicon)
- **TextBlob-NL** - Dutch Pattern analyzer with TextBlob interface
- **VADER** - English baseline for comparison
- **Claude API** - LLM-based sentiment with customizable prompts

## Quick Start (Version 2 - Claude API)

### 1. Installation

```bash
pip install anthropic pandas pyyaml tqdm
```

### 2. Set API Key

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

### 3. Run Analysis

```bash
# Sample analysis (10 speeches, standard mode)
python claude_analysis.py --sample 10

# Concise mode (token-efficient)
python claude_analysis.py --sample 20 --mode concise

# Elaborate mode (detailed reasoning)
python claude_analysis.py --sample 5 --mode elaborate

# Full analysis with custom config
python claude_analysis.py --config config/default_config.yaml
```

## Usage Examples

### Basic Analysis

```bash
# Analyze 10 random speeches with standard output
python claude_analysis.py --sample 10 --mode standard

# Analyze all speeches (warning: expensive!)
python claude_analysis.py --min-length 100

# Analyze specific parties
python claude_analysis.py --parties PVV VVD CDA --sample 20
```

### Prompt Testing and Comparison

Test different prompts on the same data to compare effectiveness:

```bash
# Test all prompts on 5 samples
python prompt_tester.py --samples 5 --min-length 200

# Test specific prompts
python prompt_tester.py --prompts concise standard elaborate --samples 10

# Save detailed results
python prompt_tester.py --samples 5 --output my_test_results.json
```

This generates a comparison report showing:
- Inter-prompt agreement rates
- Token efficiency per prompt
- Cost per case found
- Confidence score distributions

### Configuration Files

Use YAML configs for reproducible analyses:

```bash
# Use default config
python claude_analysis.py --config config/default_config.yaml

# Use concise (cheap) config
python claude_analysis.py --config config/concise_config.yaml

# Use elaborate (detailed) config
python claude_analysis.py --config config/elaborate_config.yaml
```

## Sentiment Analysis

TK-Analyser includes a comprehensive sentiment analysis system with multiple backends optimized for Dutch text.

### Quick Start - Sentiment Analysis

```bash
# Install sentiment libraries (choose based on needs)
pip install transformers torch  # For RobBERT (best accuracy)
pip install pattern             # For Pattern.nl (fast, lightweight)
pip install vaderSentiment      # For VADER baseline

# List available backends
python sentiment_analysis.py --list-backends

# Run sentiment analysis on 10 samples
python sentiment_analysis.py --sample 10

# Use specific backends
python sentiment_analysis.py --backends robbert pattern --sample 20

# Add Claude-based sentiment analysis
python sentiment_analysis.py --claude --claude-prompt prompts/sentiment/standard.yaml --sample 10

# Analyze sentiment by party
python sentiment_analysis.py --parties PVV VVD CDA
```

### Compare Sentiment Libraries

Test and evaluate different sentiment analysis backends:

```bash
# Compare all available backends
python sentiment_tester.py --samples 10 --min-length 200

# Test specific backends
python sentiment_tester.py --backends robbert pattern vader --samples 20

# Generate comparison report
python sentiment_tester.py --samples 15 --output my_sentiment_test.json
```

This generates reports showing:
- Inter-backend agreement rates
- Processing speed per backend
- Sentiment distribution by party
- Label consistency analysis

### Sentiment Analysis Backends Comparison

| Backend | Type | Accuracy | Speed | Dutch-Specific | Requirements |
|---------|------|----------|-------|----------------|--------------|
| **RobBERT v2** | Transformer | ~93% | Slow | Yes | transformers, torch |
| **BERTje** | Transformer | High | Slow | Yes | transformers, torch |
| **Multilingual BERT** | Transformer | Good | Slow | No | transformers, torch |
| **Pattern.nl** | Rule-based | Moderate | Fast | Yes | pattern |
| **TextBlob-NL** | Rule-based | Moderate | Fast | Yes | textblob-nl |
| **VADER** | Rule-based | Low (Dutch) | Very Fast | No | vaderSentiment |
| **Claude API** | LLM | High | Moderate | Custom | anthropic |

### Sentiment Output Format

```json
{
  "speech_metadata": {
    "speaker_name": "De heer Wilders",
    "speaker_party": "PVV",
    "date": "2021-09-23"
  },
  "library_results": {
    "robbert": {
      "label": "negative",
      "score": 0.85,
      "processing_time": 0.234
    },
    "pattern": {
      "label": "negative",
      "score": 0.72,
      "raw_scores": {
        "polarity": -0.44,
        "subjectivity": 0.65
      }
    }
  },
  "consensus": {
    "label": "negative",
    "confidence": 0.85,
    "intensity": "strong"
  }
}
```

## Output Modes

### Concise Mode (`--mode concise`)
- **Token Usage**: ~1000 tokens/analysis
- **Cost**: Lowest
- **Output**: Minimal JSON with just findings
- **Use Case**: Large-scale screening, budget constraints

### Standard Mode (`--mode standard`)
- **Token Usage**: ~2000-4000 tokens/analysis
- **Cost**: Moderate
- **Output**: Balanced detail with explanations
- **Use Case**: Regular analysis, good balance

### Elaborate Mode (`--mode elaborate`)
- **Token Usage**: ~4000-8000 tokens/analysis
- **Cost**: Highest
- **Output**: Comprehensive reasoning, alternative interpretations
- **Use Case**: Academic research, detailed case studies

## Output Format

### Standard Analysis Output

```json
{
  "timestamp": "2025-01-15T10:30:00",
  "config": {
    "output_mode": "standard",
    "sample_size": 10
  },
  "prompt_used": {
    "name": "standard",
    "version": "1.0"
  },
  "summary": {
    "total_speeches_analyzed": 10,
    "successful_analyses": 9,
    "total_cases_found": 15,
    "avg_cases_per_speech": 1.67,
    "avg_confidence": 0.82,
    "type_distribution": {
      "demonisering": 8,
      "discrediting_information": 4,
      "bedreiging": 3
    },
    "target_distribution": {
      "politici": 10,
      "media": 3,
      "minderheden": 2
    }
  },
  "analyses": [...],
  "processor_statistics": {
    "request_count": 10,
    "error_count": 1,
    "token_usage": {
      "input_tokens": 25000,
      "output_tokens": 8000
    },
    "estimated_cost": {
      "total_cost": 0.195
    }
  }
}
```

### Prompt Comparison Report

```
============================================================
PROMPT COMPARISON REPORT
============================================================

AGREEMENT METRICS:
  Presence Agreement: 85.00%
  Count Agreement: 70.00%
  Type Agreement: 75.00%

EFFICIENCY METRICS:

  concise:
    Avg Tokens/Analysis: 1200
    Cost/Case: $0.0015
    Success Rate: 100.00%

  standard:
    Avg Tokens/Analysis: 3500
    Cost/Case: $0.0042
    Success Rate: 100.00%

  elaborate:
    Avg Tokens/Analysis: 6800
    Cost/Case: $0.0089
    Success Rate: 95.00%
============================================================
```

## Delegitimation Categories

The analyzer identifies four types of delegitimation:

| Type | Description | Example |
|------|-------------|---------|
| `breakdown_communication` | Refusing dialogue, walking out | "Ik weiger nog langer naar deze onzin te luisteren!" |
| `discrediting_information` | Unfounded doubt about facts/experts | "Die cijfers kloppen niet, ik vertrouw ze niet." |
| `demonisering` | Personal attacks, ridicule, misrepresentation | "U bent een gevaar voor de democratie!" |
| `bedreiging` | Threats against persons/groups | "Als wij de macht hebben, gaan we jullie aanpakken!" |

Target groups:
- `parlement` - Parliament as institution
- `politici` - Politicians or parties
- `rechtspraak` - Judiciary/legal system
- `media` - Press and journalists
- `ambtenaren` - Civil servants
- `minderheden` - Minority groups

## Project Structure

```
TK-analyser/
├── claude_analysis.py        # Delegitimization analysis (v2)
├── claude_processor.py       # Claude API integration
├── prompt_tester.py          # Prompt comparison system
├── sentiment_analyzer.py     # Multi-backend sentiment analysis
├── sentiment_analysis.py     # Sentiment analysis runner
├── sentiment_tester.py       # Sentiment library comparison
├── apb_analysis.py           # Original analysis script (v1)
├── fireworks_processor.py    # Fireworks AI integration (v1)
├── requirements.txt          # Python dependencies
├── config/
│   ├── default_config.yaml   # Default configuration
│   ├── concise_config.yaml   # Token-efficient config
│   └── elaborate_config.yaml # Detailed analysis config
├── prompts/
│   ├── delegitimatie.yaml    # Original prompt (v1)
│   ├── variants/             # Delegitimization prompts
│   │   ├── concise.yaml      # Minimal token usage
│   │   ├── standard.yaml     # Balanced output
│   │   ├── elaborate.yaml    # Comprehensive reasoning
│   │   └── structured_cot.yaml # Chain-of-thought
│   └── sentiment/            # Sentiment analysis prompts
│       ├── concise.yaml      # Fast sentiment scoring
│       ├── standard.yaml     # Balanced sentiment analysis
│       └── elaborate.yaml    # Comprehensive emotional analysis
├── data/
│   └── apb/
│       └── apb_speeches.csv  # Speech dataset
├── analysis_output/          # Delegitimization results
└── sentiment_output/         # Sentiment results
```

## Error Handling

The v2 system includes comprehensive error handling:

- **Automatic Retries**: Up to 3 retries with exponential backoff
- **Rate Limit Handling**: Automatic wait and retry on rate limits
- **Connection Errors**: Retry on network failures
- **JSON Parsing**: Multiple fallback strategies for parsing responses
- **Graceful Interruption**: Save partial results on Ctrl+C
- **Detailed Logging**: Full error traces for debugging

## Cost Estimation

The system tracks token usage and estimates costs:

```python
# Example cost calculation (Claude Sonnet 4)
# Input: $3.00 per million tokens
# Output: $15.00 per million tokens

# For 100 speeches in standard mode:
# ~350,000 input tokens = $1.05
# ~120,000 output tokens = $1.80
# Total ≈ $2.85

# Sentiment analysis (library-based):
# FREE - no API calls needed
# RobBERT: ~0.2-0.5s per speech (GPU recommended)
# Pattern.nl: ~0.01s per speech (CPU only)
```

## Installation Options

### Minimal (Claude API only)
```bash
pip install anthropic pandas pyyaml tqdm
```

### With Transformer Sentiment (Best Accuracy)
```bash
pip install anthropic pandas pyyaml tqdm transformers torch
```

### With Lightweight Sentiment (Fast, No GPU)
```bash
pip install anthropic pandas pyyaml tqdm pattern vaderSentiment
```

### Full Installation
```bash
pip install anthropic pandas pyyaml tqdm transformers torch pattern textblob-nl vaderSentiment
```

## Version 1 (Fireworks AI)

The original version using Fireworks AI is still available:

```bash
export FIREWORKS_API_KEY="your-key"
python apb_analysis.py --sample 10 --min-length 100
```

## License

Licensed under GPLv3

## Contributing

- Open an issue
- Submit a pull request
- Contact: p.vanboheemen@democratiemonitor.nl

## Citation

If you use this tool in academic research, please cite:

```
TK-Analyser: Political Speech Delegitimation Detection Tool
Democratie Monitor
https://github.com/[repository]
```
