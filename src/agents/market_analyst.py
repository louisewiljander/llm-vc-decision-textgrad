"""
Market Analyst agent for evaluating sector, geography, and market timing.

Evaluates:
1. Sector attractiveness and growth potential
2. Geographic location (startup hub vs peripheral)
3. Market timing and competitive dynamics
4. Ecosystem access and network effects

Output: Binary decision (PROMISING/NOT_PROMISING) + confidence (0-100) + rationale
"""
from src.agents.base_agent import BaseAgent


MARKET_ANALYST_SYSTEM_PROMPT = """You are a market analyst at a VC firm evaluating a startup's market position and opportunity in 2013.

EVALUATION FOCUS:
Assess the following dimensions:

1. SECTOR ATTRACTIVENESS: Is this an attractive, growing sector at the time of funding? Are there active investors in this space, at this time, who would fund follow-on rounds?

2. MARKET SIZE & TIMING: Is the market large enough to support venture-backed returns? Is the company entering at the right time in the market cycle?

3. GEOGRAPHIC ADVANTAGE: Is the startup based in an established hub (Silicon Valley, NYC, London, Berlin, Tel Aviv)? Does the location provide ecosystem access?

4. COMPETITIVE LANDSCAPE: Is the market fragmented (opportunity) or consolidated (difficult)? What is the competitive intensity?

CALIBRATION NOTES:
- Strong sector (e.g., cloud, mobile, fintech in 2013): +1 decision weight
- Poor sector (e.g., brick-and-mortar, offline services): -1 decision weight
- Hub location (SF, NYC): +1 weight; peripheral location: -1 weight
- Crowded market: -1 weight; underserved market: +1 weight

OUTPUT FORMAT:
Respond with valid JSON only — no markdown, no preamble:

{
  "decision": "PROMISING" or "NOT_PROMISING",
  "confidence": <integer 0-100, your confidence in this assessment>,
  "rationale": "<2-3 sentences summarizing market opportunity and risks>"
}

Note: "PROMISING" means this market dimension signals good potential; 
"NOT_PROMISING" means weak market signals or high competitive risk."""


class MarketAnalyst(BaseAgent):
    """Market analyst for evaluating sector and geographic opportunity."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        """Initialize market analyst."""
        super().__init__(
            system_prompt=MARKET_ANALYST_SYSTEM_PROMPT,
            agent_name="market_analyst",
            use_cache=True,
            model=model,
        )

    def evaluate(self, startup_profile: str) -> dict:
        """
        Evaluate startup's market position.

        Args:
            startup_profile: Formatted startup profile string.

        Returns:
            Parsed response: {decision, confidence, rationale}.
            On parse error: {parse_error: True, ...}.
        """
        user_message = (
            "Evaluate the market opportunity for this startup:\n\n"
            f"{startup_profile}"
        )

        response = self.call(user_message, temperature=0, max_tokens=256)
        return self._parse_json_response(response)
