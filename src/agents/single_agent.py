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
In this historical cohort of startups, approximately 10% eventually secured follow-on funding within one year of their initial investment, while 90% did not. Use this base rate to calibrate your probability estimates — do not assume every startup will succeed.

YOUR EVALUATION FRAMEWORK:
For each startup profile, assess the following four dimensions:

1. MARKET: Is this an attractive, growing sector in 2013 with clear exit opportunities? Is the market large enough to support venture-backed returns, and is the timing right? Is the startup based in an established hub (Silicon Valley, NYC, London, Berlin, Tel Aviv)? Is the competitive landscape fragmented (opportunity) or consolidated (risk)?

2. TEAM: Do the founders have relevant domain expertise or a prior track record as serial entrepreneurs? Is the team large enough to execute (typically 3–20 for early stage) with diversity across technical, business, and domain backgrounds? Do team members hold degrees from top-tier universities?

3. BUSINESS MODEL: Is the revenue model clear and defensible (subscription, transaction, licensing)? Can the business scale with capital and are unit economics favorable? Does the funding trajectory signal investor confidence, and is capital being deployed efficiently?

4. FEASIBILITY: Does the description articulate a differentiated product addressing clear user needs? What is the execution risk given the technical and operational complexity? Are there traction signals — milestones, early customers, partnerships, ecosystem engagement? Can the company reach proof-of-concept with available resources?

IMPORTANT CALIBRATION NOTE:
Your role is to discriminate carefully — not every startup succeeds. Be willing to assign low probabilities (< 30%) to startups that lack differentiation, have thin profiles, or operate in crowded/low-margin markets. Likewise, reserve high probabilities (> 75%) for startups that demonstrate multiple strong signals across all four dimensions.

OUTPUT FORMAT:
Respond with valid JSON only — no markdown, no preamble. Use exactly these fields:

{
  "decision": "INVEST" or "PASS",
  "probability": <integer 0–100, your estimated probability of securing follow-on funding within one year>,
  "market_assessment": "<one sentence on sector, timing, geography, and competitive landscape>",
  "team_assessment": "<one sentence on founder quality, credentials, and team composition>",
  "business_model_assessment": "<one sentence on revenue model clarity, scalability, and capital efficiency>",
  "feasibility_assessment": "<one sentence on product differentiation, execution risk, and traction signals>",
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
            market_assessment, team_assessment, business_model_assessment,
            feasibility_assessment, key_risks, reasoning.
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
