# TK-Analyser: Claude Code Setup Instructions

Quick guide for deploying, configuring, and testing this app using Claude Code in VS Code.

## 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## 2. Configure API Key

```bash
# Set your Anthropic API key
export ANTHROPIC_API_KEY="your-key-here"

# Or create .env file
echo 'ANTHROPIC_API_KEY=your-key-here' > .env
```

## 3. Prepare Data

Place your parliamentary speech CSV in the project root with columns:
- `speech_text` - The speech content
- `speaker_name` - Speaker's name
- `speaker_party` - Political party
- `date` (optional) - Speech date

## 4. Test Prompts on Sample Data

```bash
# Compare different prompt variants
python prompt_tester.py --input your_data.csv --sample-size 10
```

## 5. Run Delegitimization Analysis

```bash
# Concise mode (saves tokens)
python claude_analysis.py --input data.csv --output results.csv --mode concise

# Elaborate mode (detailed analysis)
python claude_analysis.py --input data.csv --output results.csv --mode elaborate
```

## 6. Run Sentiment Analysis

```bash
# Test different sentiment libraries
python sentiment_tester.py --input data.csv --sample-size 20

# Full sentiment analysis
python sentiment_analysis.py --input data.csv --output sentiment_results.csv
```

## 7. Enable Advanced Features

```python
# Async processing (parallel API calls)
from async_processor import AsyncClaudeProcessor
processor = AsyncClaudeProcessor(api_key, max_concurrent=5)

# Memory management
from memory_manager import MemoryValidator
validator = MemoryValidator()
validated_texts = validator.validate_batch(texts)

# Structured JSON logging
from logging_config import setup_logging
setup_logging(log_dir="logs", json_output=True)
```

## 8. Monitor Logs

```bash
# View human-readable logs
tail -f logs/analysis_*.log

# Parse JSON logs for metrics
cat logs/analysis_*.jsonl | jq '.level == "INFO"'
```

## Common Claude Code Commands

Ask Claude Code to:
- "Run the prompt tester on 5 samples"
- "Analyze sentiment using RobBERT backend only"
- "Show me the token usage from the last run"
- "Compare concise vs elaborate mode outputs"
- "Check GPU memory usage during sentiment analysis"

## Troubleshooting

- **API errors**: Check `ANTHROPIC_API_KEY` is set correctly
- **Memory issues**: Reduce batch size or use `MemoryValidator`
- **GPU OOM**: Use `cleanup()` method after sentiment analysis
- **Missing columns**: Ensure CSV has required fields

## File Structure

```
├── claude_analysis.py      # Main delegitimization analyzer
├── claude_processor.py     # Claude API integration
├── async_processor.py      # Parallel API processing
├── sentiment_analyzer.py   # Multi-backend sentiment
├── memory_manager.py       # Memory/GPU management
├── logging_config.py       # Structured logging
├── prompt_tester.py        # Prompt comparison tool
└── prompts/                # YAML prompt templates
    ├── variants/           # Delegitimization prompts
    └── sentiment/          # Sentiment prompts
```
