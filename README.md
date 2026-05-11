# LLM VC Decision - TextGrad Optimization

Optimizing venture capital investment decisions using TextGrad and large language models with prompt caching and comprehensive logging.

## 🎯 Project Goals

- Evaluate startup profiles using LLM-based VC agents
- Optimize agent prompts using TextGrad for improved decision accuracy
- Track all API calls and costs with prompt caching (~90% token savings)
- Build reproducible results for research and thesis chapters

## 🏗️ Architecture

### Core Components

1. **Data Pipeline** (`notebooks/data_processing.ipynb`)
   - Process 2013 Crunchbase snapshot data
   - Feature engineering with team aggregation
   - Anonymization and leakage detection
   - Dynamic university ranking integration

2. **Agents** (`src/agents/`)
   - **InvestorAgent**: Evaluates startups and provides recommendations
   - **JudgeAgent**: Reviews investor decisions for bias
   - Both use cached system prompts for cost efficiency

3. **Caching & Logging** (`src/utils/`)
   - **LLMClient**: Anthropic API with prompt caching
   - **APILogger**: SQLite-based response logging and analysis
   - Cost tracking and cache performance metrics

4. **Experiments** (`experiments/`)
   - Baseline evaluation pipeline
   - TextGrad optimization runs
   - Hyperparameter sweeps

## 🚀 Quick Start

### 1. Setup

```bash
# Clone and enter directory
cd llm-vc-decision-textgrad

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the project and its runtime dependencies
pip install -e .
```

### 2. Configure API Key

```bash
# Add Anthropic API key to .env
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
```

### 3. Test Caching & Logging

```bash
# Run quick start to verify setup
python scripts/quickstart.py

# Or run full example pipeline
python scripts/example_caching_logging.py
```

## 💾 Prompt Caching

### Key Features

- **90% token cost savings** on system prompts after first use
- **Automatic tracking** of cache creation and read tokens
- **5-minute cache window** for prompt variants
- **Cost calculation** included in all logs

### How It Works

```python
from src.agents.investor import InvestorAgent

# Initialize agent (system prompt will be cached)
investor = InvestorAgent(use_cache=True)

# First call: prompt cached (25% cost)
# Subsequent calls: read from cache (10% cost)
result = investor.evaluate_startup(profile)

# Monitor savings
stats = investor.llm_client.get_cache_stats()
print(f"Cache savings: ${stats['estimated_savings_usd']:.2f}")
```

### Example Savings

For a 500-token system prompt with 10 evaluations:
- **Without caching**: 10 × $0.0015 = $0.015
- **With caching**: $0.001875 + (9 × $0.00015) = $0.002625
- **Savings**: ~82%

## 📊 Response Logging

All API calls are logged to `results/logs/api_calls.db`:

```python
from src.utils.logging import APILogger

logger = APILogger()

# View summary
logger.print_summary(agent_name="investor")

# Get logs as DataFrame
df = logger.get_dataframe(hours=24)

# Analyze costs
breakdown = logger.cost_breakdown()
print(f"Total cost: ${breakdown['total_cost']:.4f}")
```

### Logged Fields

- Timestamp, model, agent name
- Input/output tokens + cache metrics
- Full request and response text
- Calculated cost per Anthropic pricing
- Error messages (if any)

### Database Schema

```sql
CREATE TABLE api_calls (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    model TEXT,
    agent_name TEXT,
    system_prompt_hash TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_creation_input_tokens INTEGER,
    cache_read_input_tokens INTEGER,
    total_cost_usd REAL,
    user_message TEXT,
    assistant_response TEXT,
    stop_reason TEXT,
    error_message TEXT
)
```

## 📁 Directory Structure

```
├── notebooks/
│   ├── data_processing.ipynb      # Data pipeline and feature engineering
│   ├── agent_data_quality_audit.ipynb  # Data quality verification
│   └── analysis.ipynb             # Results analysis
├── src/
│   ├── agents/
│   │   ├── base_agent.py          # Base agent with caching
│   │   ├── investor.py            # Investor evaluation agent
│   │   └── judge.py               # Judge review agent
│   ├── utils/
│   │   ├── llm_client.py          # Cached LLM client + logging
│   │   └── logging.py             # Log analysis utilities
│   ├── prompts/
│   │   ├── investor_prompt.txt
│   │   └── judge_prompt.txt
│   └── textgrad/
│       ├── optimizer.py
│       └── feedback_parser.py
├── experiments/
│   ├── run_baseline.py            # Baseline evaluation
│   ├── run_textgrad.py            # TextGrad optimization
│   └── sweep.py                   # Hyperparameter sweep
├── data/
│   ├── raw/                       # Crunchbase CSVs
│   └── processed/                 # Parquet output
├── scripts/
│   ├── quickstart.py              # Verify setup
│   ├── example_caching_logging.py # Full pipeline demo
│   └── split_objects_by_entity_type.py
├── results/
│   ├── logs/api_calls.db          # All API calls logged here
│   ├── metrics/
│   └── plots/
├── tests/
│   └── test_agents.py
├── configs/
│   ├── base.yaml
│   └── agents.yaml
├── CACHING_AND_LOGGING.md         # Detailed caching/logging guide
├── README.md                      # This file
└── requirements.txt
```

## 🔧 Core Files

### `src/utils/llm_client.py`
Centralized LLM client with:
- Prompt caching via `cache_control: {"type": "ephemeral"}`
- Automatic response logging to SQLite
- Cost calculation per Anthropic pricing
- Cache statistics and performance metrics

### `src/agents/base_agent.py`
Base class for all agents:
- Wraps `CachedLLMClient`
- Handles system prompt caching
- Provides `call()` interface
- Exposes cache statistics

### `src/utils/logging.py`
Log analysis utilities:
- Query logs as pandas DataFrame
- Print summaries and recent calls
- Cost breakdown by agent/time
- Statistics and savings calculation

## 📈 Experiments

### Baseline (`experiments/run_baseline.py`)
Evaluate agents without optimization:
```bash
python experiments/run_baseline.py --config configs/base.yaml
```

### TextGrad Optimization (`experiments/run_textgrad.py`)
Optimize prompts using TextGrad:
```bash
python experiments/run_textgrad.py \
    --baseline_results results/baseline.json \
    --num_iterations 5
```

### Results
All results logged to `results/`:
- `metrics/` - Performance metrics
- `plots/` - Visualizations
- `logs/api_calls.db` - All API calls

## 📚 Documentation

- **[CACHING_AND_LOGGING.md](CACHING_AND_LOGGING.md)** - Complete guide to prompt caching and response logging
- **[data_processing.ipynb](notebooks/data_processing.ipynb)** - Data pipeline walkthrough
- **[agent_data_quality_audit.ipynb](notebooks/agent_data_quality_audit.ipynb)** - Data quality checks

## 🔍 Monitoring

### Check Cache Performance
```python
from src.agents.investor import InvestorAgent

agent = InvestorAgent(use_cache=True)
stats = agent.llm_client.get_cache_stats()

print(f"Total calls: {stats['total_calls']}")
print(f"Cache usage: {stats['cache_usage_percentage']:.1f}%")
print(f"Savings: ${stats['estimated_savings_usd']:.4f}")
```

### View Recent API Calls
```bash
python -c "from src.utils.logging import APILogger; APILogger().print_recent(5)"
```

### Cost Analysis
```python
from src.utils.logging import APILogger

logger = APILogger()
df = logger.get_dataframe()

# By agent
print(df.groupby('agent_name')['total_cost_usd'].sum())

# Timeline
print(df.groupby(df['timestamp'].str[:10])['total_cost_usd'].sum())
```

## 🛠️ Development

### Running Tests
```bash
pytest tests/ -v
```

### Linting
```bash
black src/ scripts/
flake8 src/ scripts/
```

### Adding New Agents
1. Extend `BaseAgent` in `src/agents/`
2. Define system prompt with cache support
3. Implement agent-specific methods
4. Logs will work automatically

## 📊 For Your Thesis

### Methodology Section
Use logged data to show:
- Prompt engineering approach
- API call statistics
- Cost analysis and efficiency gains
- Cache performance metrics

### Results Section
Include:
- Agent performance metrics
- TextGrad optimization improvements
- Cost comparisons (baseline vs optimized)
- Cache usage statistics

### Appendix
Export logs for analysis:
```python
df = APILogger().get_dataframe()
df.to_csv('appendix_api_logs.csv')
```

## ⚙️ Configuration

### Environment Variables (`.env`)
```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...        # Optional, for future use
AZURE_OPENAI_API_KEY=...     # Optional, for future use
```

### .gitignore
```
.env                          # API keys (never commit)
.venv/                        # Virtual environment
results/logs/api_calls.db    # Optional: exclude large logs
data/**/*.csv                # Raw data (optional)
```

## 🐛 Troubleshooting

### Missing API Key
```bash
# Check .env file exists and has ANTHROPIC_API_KEY
cat .env

# Add if missing
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
```

### Cache Not Working
```python
# Check if caching is enabled
agent = InvestorAgent(use_cache=True)  # Must be True

# Verify cache stats
stats = agent.llm_client.get_cache_stats()
if stats['cache_read_tokens'] == 0:
    print("Cache warming up - make more calls to see benefits")
```

### Database Errors
```python
# Reset logs (if corrupted)
import os
os.remove('results/logs/api_calls.db')
# Will recreate on next call
```

## 📖 References

- [Anthropic Docs - Prompt Caching](https://docs.anthropic.com/docs/build-a-chatbot#token-counting)
- [Anthropic Pricing](https://www.anthropic.com/pricing/claude)
- [TextGrad Paper](https://arxiv.org/abs/2401.08546)

## 📝 Citation

If using this code, please cite:

```bibtex
@thesis{llm-vc-decision,
  title={Optimizing VC Investment Decisions with TextGrad and LLMs},
  author={Your Name},
  year={2026}
}
```

## 📄 License

MIT License - See LICENSE file for details

---

**Questions?** Check [CACHING_AND_LOGGING.md](CACHING_AND_LOGGING.md) for detailed documentation.
