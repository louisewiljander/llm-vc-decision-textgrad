# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Your VC Decision System                       │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                      EXPERIMENT LAYER                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  experiments/run_baseline.py  ←─→  experiments/run_textgrad.py   │
│                                                                    │
│  • Evaluates startups           • Optimizes prompts               │
│  • TBD cost tracking           • Logs optimization steps          │
│  • Logs results                 • Measures improvements           │
│                                                                    │
└────────────────────┬─────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────────┐
│                      AGENT LAYER                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────────────────┐    ┌──────────────────────────┐   │
│  │   InvestorAgent          │    │   JudgeAgent             │   │
│  ├──────────────────────────┤    ├──────────────────────────┤   │
│  │ • Extends BaseAgent      │    │ • Extends BaseAgent      │   │
│  │ • Uses cached prompts    │    │ • Uses cached prompts    │   │
│  │ • Evaluates startups     │    │ • Reviews decisions      │   │
│  │ • JSON output            │    │ • JSON output            │   │
│  │ • Auto-logging           │    │ • Auto-logging           │   │
│  └──────────────────────────┘    └──────────────────────────┘   │
│           ▲                              ▲                        │
│           │                              │                        │
│           └──────────────┬───────────────┘                        │
│                          │                                         │
│              ┌───────────▼──────────┐                             │
│              │   BaseAgent          │                             │
│              ├──────────────┬───────┤                             │
│              │ • Wraps      │ Cache │                             │
│              │   LLMClient  │enabled│                             │
│              │ • Simple     │ by    │                             │
│              │   call()     │default│                             │
│              │   interface  │       │                             │
│              └───────────┬──────────┘                             │
│                          │                                         │
│                          ▼                                         │
│              ┌───────────────────────┐                            │
│              │   CachedLLMClient     │                            │
│              │    (llm_client.py)    │                            │
│              └───────────────────────┘                            │
│                                                                    │
└────────────────────┬─────────────────────────────────────────────┘
                     │
            ┌────────┴────────┐
            ▼                 ▼
┌──────────────────────────┐  ┌─────────────────────────────────┐
│  ANTHROPIC API           │  │  LOGGING LAYER                  │
│  (Claude 3.5 Sonnet)     │  │  (SQLite Database)              │
├──────────────────────────┤  ├─────────────────────────────────┤
│                          │  │                                 │
│ • Prompt caching         │  │ • Logs every API call           │
│   - 90% token savings    │  │ • Records input/output tokens   │
│ • Cache control types:   │  │ • Tracks cache hits/creation    │
│   - ephemeral (5 min)    │  │ • Calculates costs              │
│ • System prompt cached   │  │ • Stores full request/response  │
│ • Reuse across calls     │  │ • Records errors               │
│                          │  │                                 │
└──────────────────────────┘  └──────────────────┬──────────────┘
                                                 │
                                    ┌────────────▼────────────┐
                                    │  results/logs/          │
                                    │  api_calls.db           │
                                    │                         │
                                    │  Table: api_calls       │
                                    │  • 13 columns          │
                                    │  • Timestamp, tokens    │
                                    │  • Cost, cache info     │
                                    │  • Request/response     │
                                    │  • Error tracking       │
                                    └─────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                    ANALYSIS LAYER                                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────────────────┐    ┌──────────────────────────┐   │
│  │   APILogger              │    │   Pandas DataFrames      │   │
│  │   (logging.py)           │    │                          │   │
│  ├──────────────────────────┤    ├──────────────────────────┤   │
│  │ • Query logs             │    │ • Load logs as DF        │   │
│  │ • Print summaries        │    │ • Filter/group/agg       │   │
│  │ • Cost breakdown         │    │ • Time series analysis   │   │
│  │ • Cache statistics       │    │ • Export to CSV/JSON    │   │
│  └──────────────────────────┘    └──────────────────────────┘   │
│           ▲                              ▲                        │
│           │                              │                        │
│           └──────────────┬───────────────┘                        │
│                          │                                         │
│              ┌───────────▼──────────┐                             │
│              │   Scripts            │                             │
│              ├──────────────────────┤                             │
│              │ • quickstart.py      │                             │
│              │ • example_*.py       │                             │
│              │ • verify_setup.py    │                             │
│              └──────────────────────┘                             │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
STARTUP PROFILE
     │
     ▼
┌─────────────────────────────────────────┐
│  InvestorAgent.evaluate_startup()       │
├─────────────────────────────────────────┤
│  1. Format startup data                 │
│  2. Call BaseAgent.call()               │
│  3. System prompt marked for cache      │
│  4. First call: prompt cached (25% cost)│
│  5. Parse JSON response                 │
└─────────────────────────────────────────┘
     │
     └─→ CachedLLMClient.call()
         │
         ├─→ Anthropic API
         │   │
         │   ├─→ First call: Cache prompt
         │   │   Cost: 25% of normal input
         │   │
         │   └─→ Later calls: Read from cache
         │       Cost: 10% of normal input
         │
         ├─→ Log to SQLite
         │   • Timestamp: 2026-04-28 15:42:00
         │   • Model: claude-3-5-sonnet
         │   • Agent: investor
         │   • Tokens: 425 in → 187 out
         │   • Cache: 200 tokens read (10% cost)
         │   • Cost: $0.00285
         │   • Response: {rec: "BUY", conf: 78}
         │
         └─→ Return response

     ▼
INVESTOR_ANALYSIS (JSON)
     │
     ├─→ Store in memory
     │
     ├─→ Pass to Judge
     │
     └─→ Log for results
```

## Cache Lifecycle

```
TIMELINE (5-minute cache window per prompt)

Time 0:00
┌─────────────────────────────────┐
│ Call 1: InvestorAgent           │
│ System prompt: "Evaluate..." (500 tok)
│ → CACHE: Create (25% cost)
│ → Cost: $0.001875
└─────────────────────────────────┘
     │
     ▼
Time 0:15
┌─────────────────────────────────┐
│ Call 2: InvestorAgent           │
│ System prompt: Same (500 tok)   │
│ → CACHE: HIT! Read (10% cost)
│ → Savings vs Call 1: 87%
│ → Cost: $0.00015
└─────────────────────────────────┘
     │
     ▼
Time 4:50
┌─────────────────────────────────┐
│ Call 10: InvestorAgent          │
│ System prompt: Same (500 tok)   │
│ → CACHE: HIT! Read (10% cost)
│ → Still valid (< 5 min)
│ → Cost: $0.00015
└─────────────────────────────────┘
     │
     ▼
Time 5:01
┌─────────────────────────────────┐
│ Call 11: InvestorAgent          │
│ System prompt: Same (500 tok)   │
│ → CACHE: EXPIRED (> 5 min)
│ → New cache created (25% cost)
│ → Cost: $0.001875
└─────────────────────────────────┘

COST SUMMARY (11 calls):
Without caching: 11 × $0.0015 = $0.0165
With caching:    $0.001875 + $0.00015×8 + $0.001875×1 = $0.00345
SAVINGS: 79%
```

## Cache Statistics Display

```
After running multiple evaluations:

📊 Cache Performance:
   Total API calls: 15
   Input tokens: 6,375
   Cache created tokens: 500     (first call)
   Cache read tokens: 7,000      (calls 2-15)
   Cache usage: 52.4%
   Total cost: $0.0287
   Estimated cache savings: $0.0213

💾 Cost Breakdown:
   Input cost: $0.0191
   Output cost: $0.0271
   Cache creation: $0.0019
   Cache read: $0.0002
   
📈 By Agent:
   Investor: 10 calls, $0.0198
   Judge: 5 calls, $0.0089
```

## Integration with TextGrad

```
BASELINE EXPERIMENT
│
├─→ Run 100 startup evaluations
│   • Investor: 100 calls (system prompt cached after first)
│   • Judge: 100 calls (system prompt cached after first)
│
├─→ Log all API calls
│   • Each call: input tokens, output tokens, cache hits, cost
│   • Total logged: ~1000 API calls × 13 columns
│   • Database size: ~2-5 MB

├─→ Calculate metrics
│   • Accuracy: baseline recommendations vs ground truth
│   • Cost: $0.XX for 200 evaluations
│   • Cache savings: ~$Y.YY
│
▼

TEXTGRAD OPTIMIZATION (5 iterations)
│
├─→ Iteration 1: Optimize investor prompt
│   • 50 gradient steps
│   • Each step: evaluate + judge + score
│   • Cache still active!
│   • Logs every step
│
├─→ Iteration 2-5: Refine both prompts
│   • Each iteration improves performance
│   • Logs track: performance, tokens, cost evolution
│
▼

COMPARISON
│
├─→ Query logs for baseline vs optimized
│   df_baseline = logger.get_dataframe(hours=1)
│   df_optimized = logger.get_dataframe(hours=2)
│
├─→ Analyze improvements
│   • Recommendation accuracy: improved X%
│   • Cost per evaluation: $A baseline → $B optimized
│   • Cache hit rate improved/maintained
│
▼

RESULTS FOR THESIS
│
├─→ Performance comparison
│   • Baseline accuracy: XX%
│   • TextGrad optimized: YY%
│   • Improvement: +ZZ%
│
├─→ Cost analysis  
│   • Baseline total: $XXX
│   • TextGrad total: $YYY
│   • Cost per optimization: $ZZZ
│
├─→ Cache impact
│   • Cache savings in baseline: $X
│   • Cache savings during optimization: $Y
│   • Total savings: $X + $Y
```

## Key Metrics Tracked

```
PER CALL METRICS:
├─ Timestamp (for timeline analysis)
├─ Input tokens (request size)
├─ Output tokens (response size)
├─ Cache creation tokens (first use)
├─ Cache read tokens (subsequent uses)
├─ Total cost ($)
├─ Model used
├─ Agent name
├─ Stop reason (completion status)
└─ Errors (if any)

AGGREGATE METRICS:
├─ Total API calls
├─ Total tokens (input + output)
├─ Total cost ($)
├─ Cache usage percentage
├─ Estimated cache savings ($)
├─ Cache hit rate (after first call)
├─ Average cost per call
├─ Cost per agent
└─ Cost timeline
```

---

For detailed implementation, see:
- [CACHING_AND_LOGGING.md](CACHING_AND_LOGGING.md) - Complete feature guide
- [SETUP_SUMMARY.md](SETUP_SUMMARY.md) - What was implemented
- [README.md](README.md) - Project overview
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration options
