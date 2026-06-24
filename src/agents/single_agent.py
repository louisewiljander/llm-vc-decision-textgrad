"""
Investor agent for evaluating startup profiles and making binary investment decisions.

Design rationale
----------------
Based on the literature review (Maarouf et al. 2025, Wang et al. 2025):

1. Binary output + probability (not a 4-level scale):
   AUROC is the primary metric; it requires a continuous probability score.

2. Explicit base rate in the system prompt:
   Vanilla LLMs systematically over-predict success (Wang et al. 2025).
   Anchoring the agent to the historical base rate counteracts this bias.

3. Temporal framing as a 2013-era investor:
   The dataset represents a 2013 Crunchbase snapshot. The agent should reason
   from the information available at that time, with no hindsight.

4. Structured evaluation criteria:
   Mirrors the SHAP-ranked feature importance from Maarouf et al. (2025):
   textual description > age > social presence > education > sector > funding.
"""
import json
from src.agents.base_agent import BaseAgent


# The system prompt is designed for prompt caching: it is long, stable across
# all startup evaluations, and will be reused hundreds of times in a single run.
INVESTOR_SYSTEM_PROMPT = """You are an experienced venture capital investor conducting early-stage startup evaluations. The year is 2013. You are reviewing Crunchbase profiles to decide which startups to investigate further for potential investment.

CONTEXT AND BASE RATE:
In this historical cohort of startups, approximately 57% eventually achieved a successful exit (acquisition or IPO), while 43% ultimately closed. Use this base rate to calibrate your probability estimates — do not assume every startup will succeed.

YOUR EVALUATION FRAMEWORK:
For each startup profile, assess the following dimensions in order of their known predictive importance:

1. TEXTUAL DESCRIPTION QUALITY: Does the startup articulate a clear, compelling, and differentiated value proposition? Is the business model intelligible? Does the language signal confidence and domain expertise?

2. MARKET AND SECTOR: Is this a high-growth sector? Does the company operate in a space with strong exit potential?

3. TEAM SIGNALS: Number of team members, educational credentials (especially top university alumni), and diversity of background. Larger, more credentialed teams tend to fare better.

4. FUNDING TRAJECTORY: How many funding rounds has the company completed? What is the total capital raised? Recent funding activity signals external validation.

5. NETWORK AND MOMENTUM: Milestones recorded, relationship count, and any investment activity by the company itself — all proxy for traction and ecosystem engagement.

6. LOCATION: Is the startup based in an established startup hub (US, UK, key European cities)? Geography correlates with ecosystem access and exit opportunities.

IMPORTANT CALIBRATION NOTE:
Your role is to discriminate carefully — not every startup succeeds. Be willing to assign low probabilities (< 30%) to startups that lack differentiation, have thin profiles, or operate in crowded/low-margin markets. Likewise, reserve high probabilities (> 75%) for startups that demonstrate multiple strong signals.

OUTPUT FORMAT:
Respond with valid JSON only — no markdown, no preamble. Use exactly these fields:

{
  "decision": "INVEST" or "PASS",
  "probability": <integer 0–100, your estimated probability of successful exit>,
  "market_assessment": "<one sentence>",
  "team_assessment": "<one sentence>",
  "funding_assessment": "<one sentence>",
  "key_risks": ["<risk 1>", "<risk 2>"],
  "reasoning": "<two to three sentences summarising your overall judgement>"
}

The "probability" field is your primary output. The "decision" should be INVEST if probability >= 50, otherwise PASS."""


class InvestorAgent(BaseAgent):
    """
    VC investor agent for evaluating startups from a 2013-era perspective.

    Produces a binary INVEST/PASS decision plus a calibrated probability
    score (0–100) suitable for AUROC evaluation.
    """

    def __init__(self, use_cache: bool = True, model: str = "claude-haiku-4-5-20251001"):
        """
        Initialise the investor agent.

        Args:
            use_cache: Whether to use Anthropic prompt caching (recommended:
                       True — the system prompt is long and reused many times).
            model:     Model identifier. Supports:
                       - Anthropic: "claude-haiku-4-5-20251001"
                       - Ollama: "ollama/llama2", "ollama/qwen", etc.
                       - Others: Per LiteLLM documentation
                       Defaults to Claude Haiku 4.5 for cost efficiency.
        """
        super().__init__(
            system_prompt=INVESTOR_SYSTEM_PROMPT,
            agent_name="investor",
            use_cache=use_cache,
            model=model,
        )

    def evaluate_startup(self, startup_profile: str) -> dict:
        """
        Evaluate a pre-formatted startup profile string.

        Args:
            startup_profile: A formatted profile string produced by
                             src.prompts.templates.format_startup_profile().

        Returns:
            Parsed response dict with keys: decision, probability,
            market_assessment, team_assessment, funding_assessment,
            key_risks, reasoning.
            On parse failure, returns {"raw_response": ..., "parse_error": True}.
        """
        user_message = (
            "Evaluate the following startup profile using the framework provided. "
            "Respond with valid JSON only.\n\n"
            f"{startup_profile}"
        )

        response = self.call(user_message, temperature=0.2, max_tokens=2048)
        result = self._parse_json_response(response, {"probability_float": 0.5})  # fallback — treated as uncertain
        if not result.get("parse_error"):
            # Normalise probability to 0–1 float for metrics pipeline
            result["probability_raw"] = result.get("probability", 50)
            result["probability_float"] = result["probability_raw"] / 100.0
        return result


if __name__ == "__main__":
    from src.prompts.templates import format_startup_profile
    import pandas as pd

    # Quick smoke test against a real row from the dataset
    df = pd.read_parquet("data/processed/companies_clean.parquet")
    sample = df.sample(1, random_state=42).iloc[0]

    profile = format_startup_profile(sample)
    print("=== STARTUP PROFILE ===")
    print(profile)
    print("\n=== AGENT RESPONSE ===")

    agent = InvestorAgent(use_cache=True)
    result = agent.evaluate_startup(profile)
    print(json.dumps(result, indent=2))
    print(f"\nGround truth label: {sample['target']}")
