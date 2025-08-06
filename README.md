# LLM Smoking Cessation Project

Evaluate LLM-generated smoking cessation messages with configurable prompt strategies and calibration.

## 🎯 Core Components

```
analysis-script/
├── prompt_config.py      # ⭐ CORE - All prompt generation functions
├── main_eval.py          # ⭐ CORE - Main evaluation pipeline  
├── calibration_eval.py   # ⭐ CORE - Calibration evaluation
├── plot_results.py       # 📊 Results visualization
├── preprocess_data.py    # 🔧 Data preprocessing
└── find_unmatched_messages.py  # 🔍 Data validation
```

### Key Features
- **6 Prompt Configurations** testing different strategies
- **3 LLM Models** (GPT-4o-mini, GPT-4o, O3-2025-04-16)
- **2 Evaluation Modes** (text-only, vision)
- **Calibration Analysis** for confidence assessment

## 📊 Data Preparation

### Required Raw Data Files
Before running any evaluations, you need these files in the `data/` directory:

```
data/
├── R01 Message Summary for message testing paper.xlsx    # Message metadata
├── Messaging_Testing_Data.xlsx                          # Participant responses
└── downloaded_smoke_images/                             # Image files (A1.jpg, A2.jpg, etc.)
```

### Step 1: Generate Processed Data
```bash
# Generate the main processed JSON file
python analysis-script/preprocess_data.py
```
**Output:** `data/processed_llm_data.json` (5.2MB, ~8000 entries)

### Step 2: Validate Data (Optional)
```bash
# Check for unmatched messages between data files
python analysis-script/find_unmatched_messages.py
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set API key:**
   ```bash
   export CHEN_OPENAI_API_KEY="your-api-key-here"
   ```

3. **Prepare data (if not already done):**
   ```bash
   python analysis-script/preprocess_data.py
   ```

4. **🚀 Run all evaluations:**
   ```bash
   ./run_categorical_eval.sh    # 18 categorical evaluations
   ```

## Prompt Configurations

The system supports 6 different prompt strategies, each testing different approaches:

| Configuration | Description | Features Used | Examples |
|---------------|-------------|---------------|----------|
| `zero-shot` | All participant features | Full metadata | None |
| `zero-shot-feature-select` | Key features only | Selected features | None |
| `zero-shot-feature-select-balanced` | Key features + rating range emphasis | Selected features | None |
| `few-shot` | All features + examples | Full metadata | High/low examples |
| `few-shot-feature-select` | Key features + examples | Selected features | High/low examples |
| `few-shot-feature-select-balanced` | Key features + all rating examples | Selected features | All 5 rating levels |

### Selected Features
The feature-selected configurations use only these key participant demographics:
- `age_years`
- `gender_identity` 
- `race_ethnicity`
- `quit_motivation_level`
- `social_support_to_quit`

## Models Supported

- `gpt-4o-mini` 
- `gpt-4o` 
- `o3-2025-04-16` 

## Manual Usage

### Basic Evaluation
```bash
# Specific configuration (vision mode)
python analysis-script/main_eval.py --mode vision --model gpt-4o --prompt-config few-shot-feature-select-balanced --sample-size 500

# Specific configuration (text-only mode)
python analysis-script/main_eval.py --mode text-only --model gpt-4o --prompt-config few-shot-feature-select-balanced --sample-size 500
```

### Calibration Analysis
```bash
# Calibration (vision mode)
python analysis-script/calibration_eval.py --mode vision --model gpt-4o

# Calibration (text-only mode)
python analysis-script/calibration_eval.py --mode text-only --model gpt-4o
```

### Results Visualization
```bash
# Generate plots for categorical results
python analysis-script/plot_results.py evaluation_results_gpt-4o_vision_few-shot.json
python analysis-script/plot_results.py evaluation_results_gpt-4o_text-only_few-shot.json

# Generate plots for calibration results
python analysis-script/plot_calibration_results.py calibrated_results_gpt-4o_vision.json
python analysis-script/plot_calibration_results.py calibrated_results_gpt-4o_text-only.json
```

## Output Files

### Categorical Evaluations
- Results: `evaluation_results_{model}_{mode}_{config}.json`
- Plots: `plots_{model}_{mode}_{config}/`

### Calibration Evaluations  
- Results: `calibrated_results_{model}_{mode}.json`
- Plots: `calibration_plots_{model}_{mode}/`

## Data Requirements

The system expects data in the following format:
```json
{
  "response_id": "unique_id",
  "input_message": "smoking cessation message text",
  "metadata": {
    "age_years": 35,
    "gender_identity": "Female",
    "race_ethnicity": "White",
    "quit_motivation_level": "Very motivated",
    "social_support_to_quit": "Very supportive"
  },
  "ratings": {
    "content": "Good",
    "design": "Very good", 
    "coping": "Very helpful",
    "quitting": "Extremely helpful"
  }
}
```

## Troubleshooting

### Common Issues
1. **API Key Not Set**: Ensure `CHEN_OPENAI_API_KEY` environment variable is set (or replace it with your local API key)
2. **Data Not Found**: Run `preprocess_data.py` first to generate required JSON files
3. **Memory Issues**: Reduce `--max-concurrent` parameter for large datasets
4. **Rate Limits**: The system includes automatic retry logic with exponential backoff

### Debug Mode
Add `--sample-size 10` to test with a small subset before running full evaluations.

### Data Validation
If you encounter data issues:
```bash
# Check for unmatched messages
python analysis-script/find_unmatched_messages.py

# Verify processed data structure
python -c "import json; data=json.load(open('data/processed_llm_data.json')); print(f'Loaded {len(data)} entries')"
```
