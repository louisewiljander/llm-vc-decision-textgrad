# Prompt Caching and Response Logging Setup

## Overview

This project implements two critical features for optimizing Anthropic API usage:

1. **Prompt Caching** - Reuse cached system prompts to save ~90% on input token costs
2. **Response Logging** - Log all API calls to SQLite for analysis, debugging, and results chapters

## How Prompt Caching Works

### The Problem
Your VC agents all use the same system prompt repeatedly. Without caching, every API call charges full price for the system prompt tokens.

### The Solution
Anthropic's `cache_control` feature marks system prompts for caching:
- **First call**: System prompt is cached (charged at 25% of normal input cost)
- **Subsequent calls**: System prompt is read from cache (charged at 10% of normal input cost)
- **Savings**: ~90% reduction on system prompt tokens after first call

### Example Cost Savings
If your investor system prompt is 500 tokens:

**Without caching:**
- Each call: 500 tokens × $3/1M = $0.0015

**With caching:**
- First call: 500 × $3.75/1M = $0.001875 (create cost)
- Each call after: 500 × $0.30/1M = $0.00015 (read cost)
- **90% savings** on recurring calls

## Implementation

### 1. LLM Client with Caching (`src/utils/llm_client.py`)

```python
from src.utils.llm_client import CachedLLMClient

client = CachedLLMClient(use_cache=True)
response = client.call(
    system_prompt="You are a VC investor...",
    user_message="Evaluate this startup profile...",
    agent_name="investor"
)
```

**Key Features:**
- Automatic prompt caching (enabled by default)
- Response logging to SQLite
- Cost tracking and calculation
- Cache statistics

### 2. Base Agent Class (`src/agents/base_agent.py`)

```python
from src.agents.base_agent import BaseAgent

agent = BaseAgent(
    system_prompt="Your system prompt here",
    agent_name="my_agent",
    use_cache=True  # Enable caching
)

response = agent.call("Your message here")
```

### 3. Specialized Agents

#### Investor Agent (`src/agents/investor.py`)
Evaluates startups and provides investment recommendations in JSON format.

```python
from src.agents.investor import InvestorAgent

investor = InvestorAgent(use_cache=True)
analysis = investor.evaluate_startup({
    "company": "TechCorp",
    "funding_raised": 5_000_000,
    "team_size": 12,
    # ... more fields
})
```

#### Judge Agent (`src/agents/judge.py`)
Reviews investor decisions for bias and provides independent assessment.

```python
from src.agents.judge import JudgeAgent

judge = JudgeAgent(use_cache=True)
verdict = judge.evaluate_decision(investor_analysis, startup_profile)
```

## Response Logging

All API calls are automatically logged to SQLite at `results/logs/api_calls.db`.

### Logged Data
- **Metadata**: timestamp, model, agent name
- **Tokens**: input, output, cache creation, cache read
- **Costs**: calculated per Anthropic pricing
- **Request/Response**: full user message and assistant response
- **Errors**: captured for debugging

### Querying Logs

```python
from src.utils.logging import APILogger

logger = APILogger()

# Print summary statistics
logger.print_summary(agent_name="investor")

# View recent calls
logger.print_recent(limit=10)

# Get as pandas DataFrame
df = logger.get_dataframe(hours=24)

# Cost breakdown
breakdown = logger.cost_breakdown()
print(f"Total cost: ${breakdown['total_cost']:.4f}")
```

### Log Schema

```sql
CREATE TABLE api_calls (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,                       -- UTC timestamp of call
    model TEXT,                          -- Model used (e.g., claude-3-5-sonnet)
    agent_name TEXT,                     -- Name of agent making call
    system_prompt_hash TEXT,             -- Hash of system prompt (for tracking)
    input_tokens INTEGER,                -- Input tokens (excluding cache)
    output_tokens INTEGER,               -- Output tokens
    cache_creation_input_tokens INTEGER, -- Tokens used to create cache
    cache_read_input_tokens INTEGER,     -- Tokens read from cache
    total_cost_usd REAL,                 -- Total cost of this call
    user_message TEXT,                   -- Full user message
    assistant_response TEXT,             -- Full assistant response
    stop_reason TEXT,                    -- Reason response stopped
    error_message TEXT                   -- Error message if failed
)
```

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API Key
Add your Anthropic API key to `.env`:
```bash
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run Example Pipeline
```bash
python scripts/example_caching_logging.py
```

This will:
- Initialize investor and judge agents
- Evaluate 2 sample startups
- Display cache statistics
- Show cost breakdown
- Print API logs

### 4. Access Logs
```python
from src.utils.logging import APILogger

logger = APILogger()
logger.print_summary()  # Summary statistics
logger.print_recent(5)  # Last 5 calls
```

## Cache Behavior

### When Caching is Active
- System prompt is marked with `cache_control: {"type": "ephemeral"}`
- First use: Prompt cached (charged 25% for input tokens)
- Subsequent uses: Cached prompt reused (charged 10% for input tokens)
- Cache lasts for 5 minutes per prompt variant

### Monitoring Cache Usage
```python
agent = InvestorAgent(use_cache=True)

# After making calls...
stats = agent.llm_client.get_cache_stats()
print(f"Cache read tokens: {stats['cache_read_tokens']}")
print(f"Cache usage: {stats['cache_usage_percentage']:.1f}%")
print(f"Estimated savings: ${stats['estimated_savings_usd']:.4f}")
```

## Cost Tracking

The system automatically tracks costs using Anthropic's pricing:
- **Input**: $3/1M tokens
- **Output**: $15/1M tokens  
- **Cache creation**: $3.75/1M tokens (25% of input)
- **Cache read**: $0.30/1M tokens (10% of input)

All costs are logged per API call and can be queried:

```python
df = logger.get_dataframe()
print(f"Total spend: ${df['total_cost_usd'].sum():.2f}")
print(f"By agent: {df.groupby('agent_name')['total_cost_usd'].sum()}")
```

## Best Practices

### 1. Keep System Prompts Stable
Caching is most effective when system prompts don't change. If prompts vary:
- Use consistent base prompt + variable user context
- Keep system prompts in separate file (not hardcoded)

### 2. Monitor Cache Hit Rate
```python
cache_stats = agent.llm_client.get_cache_stats()
cache_pct = cache_stats['cache_usage_percentage']
if cache_pct < 10:
    print("Warning: Low cache hit rate. Check if prompts are varying.")
```

### 3. Log Analysis for Results Chapter
Export logs for analysis:
```python
df = logger.get_dataframe()
df.to_csv('api_logs_analysis.csv', index=False)
# Use this for your results/methodology chapters
```

### 4. Regular Audits
Monitor costs and cache performance regularly:
```bash
# Check logs and costs
python -c "from src.utils.logging import APILogger; logger = APILogger(); logger.print_summary()"
```

## Integration with TextGrad

When running TextGrad optimization:

1. **Logs capture all optimization calls** - Every gradient step is logged
2. **Compare baseline vs optimized** - Query logs with `hours=6` to compare runs
3. **Track cost evolution** - See how token usage changes during optimization

```python
# Before optimization
logger.print_summary(agent_name="investor", hours=1)

# Run TextGrad optimization...

# After optimization
logger.print_summary(agent_name="investor", hours=2)
```

## Troubleshooting

### No Cache Reads Happening
1. Check that `use_cache=True` in agent initialization
2. Verify system prompts are identical between calls
3. Check that calls happen within 5 minutes of each other
4. Look at `system_prompt_hash` in logs to compare prompts

### Database Issues
```python
# Check database integrity
import sqlite3
conn = sqlite3.connect('results/logs/api_calls.db')
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM api_calls")
print(f"Total logged calls: {cursor.fetchone()[0]}")
```

### Missing Cost Data
Costs are auto-calculated from token counts. If showing as 0:
- Check that `input_tokens` and `output_tokens` are populated
- Verify response.usage contains token data
- Print raw response object for debugging

## API Reference

### CachedLLMClient

```python
client = CachedLLMClient(
    api_key=None,        # Optional; uses ANTHROPIC_API_KEY env var
    log_db_path=None,    # Optional; defaults to results/logs/api_calls.db
    model="claude-3-5-sonnet-20241022"
)

# Make cached call
response = client.call(
    system_prompt="...",
    user_message="...",
    agent_name="my_agent",
    use_cache=True,
    max_tokens=2048,
    temperature=0.7
)

# Get statistics
stats = client.get_cache_stats()
# Returns: {
#   'total_calls': int,
#   'total_input_tokens': int,
#   'cache_created_tokens': int,
#   'cache_read_tokens': int,
#   'cache_usage_percentage': float,
#   'total_cost_usd': float,
#   'estimated_savings_usd': float
# }

# Query logs
logs = client.get_logs(agent_name="investor", limit=50)
```

### APILogger

```python
logger = APILogger(db_path=None)

# Get logs as DataFrame
df = logger.get_dataframe(
    agent_name=None,
    hours=None,
    include_errors=False
)

# Print statistics
logger.print_summary(agent_name=None, hours=None)
logger.print_recent(limit=10, agent_name=None)

# Cost breakdown
breakdown = logger.cost_breakdown()
```

## Questions?

For Anthropic API documentation on prompt caching, see:
https://docs.anthropic.com/docs/build-a-chatbot#token-counting

For cost calculations and current pricing:
https://www.anthropic.com/pricing/claude
