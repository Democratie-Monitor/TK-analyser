# Political Speech Analyzer

This tool analyzes political speeches for patterns using the Fireworks AI API. It currently processes speech transcripts to identify and categorize instances where speakers attempt to undermine the legitimacy of institutions, individuals, or groups. But this can be easily changed by changing the prompt template.

## Features

- Analyze individual speeches or entire datasets
- Support for both sample and comprehensive analysis
- Configurable prompt templates via YAML
- Detailed JSON output with confidence scoring
- Robust error handling and logging
- Support for processing multiple CSV files

## Requirements

- Python 3.7+
- Fireworks AI API key
- Required Python packages (install via `pip install -r requirements.txt`):
  - pandas
  - requests
  - pyyaml
  - tqdm

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install dependencies:
```bash
pip install pandas requests pyyaml tqdm
```

3. Set up your Fireworks API key:
```bash
export FIREWORKS_API_KEY="your-api-key-here"
```

## Usage

### Command Line Arguments

The script supports the following command-line arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv_path` | `data/apb/apb_speeches.csv` | Path to CSV file(s) containing speeches |
| `--prompt_path` | `prompts/delegitimatie.yaml` | Path to YAML file with prompt template |
| `--output_dir` | `analysis_output` | Directory to save analysis results |
| `--sample` | `0` | Number of random speeches to analyze (0 for full analysis) |
| `--min-length` | `50` | Minimum text length of a speech to analyze |

### Example Commands

Run a sample analysis of 10 speeches:
```bash
python apb_analysis.py --sample 10 --min-length 100
```

Analyze all speeches in a specific CSV:
```bash
python apb_analysis.py --csv_path path/to/speeches.csv --output_dir results
```

### Output Format

The analysis generates a JSON file with the following structure:

```json
{
  "status": "success",
  "metadata": {
    "min_length": 50,
    "requested_samples": 10,
    "successful_analyses": 8
  },
  "analyses": {
    "speech_id": {
      "speaker_name": "...",
      "speaker_party": "...",
      "date": "YYYY-MM-DD",
      "text_length": 1234,
      "source_file": "speeches_2023.csv",
      "analysis": {
        "gevonden_delegitimatie": [...],
        "samenvatting": {...}
      }
    }
  }
}
```

## Delegitimation Categories

The analyzer currently identifies four types of delegitimation:

1. `breakdown_communication`: Abrupt termination of debate
2. `discrediting_information`: Unfounded doubt about facts/experts
3. `demonisering`: Distortion or exaggeration of positions
4. `bedreiging`: Threats of action against persons/groups

Target groups include:
- Parliament
- Politicians
- Judiciary
- Media
- Civil servants
- Minority groups

## Configuration

### Prompt Template

The prompt template (`delegitimatie.yaml`) defines:
- Analysis instructions
- Delegitimation definitions
- Output format requirements
- Confidence scoring guidelines

Modify the template to adjust analysis criteria or output format.

### Fireworks Processor

The `FireworksProcessor` class handles API communication with configurable parameters:
- Maximum tokens
- API timeout
- Parse timeout
- Model selection
- Temperature and sampling parameters

## Error Handling

The tool includes comprehensive error handling for:
- File loading issues
- API communication errors
- JSON parsing problems
- Timeout scenarios

All errors are logged with detailed information for debugging.

## License

The work is licensed under GPLv3

## Contributing

Open an issue, submit a pull request or contact the developer p.vanboheemen@democratiemonitor.nl 
