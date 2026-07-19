"""
Business Model Analyst agent for evaluating revenue model and scalability.

Evaluates:
1. Revenue model clarity and viability
2. Unit economics and scalability
3. Funding structure and capital efficiency
4. Business viability from available signals

Output: Binary decision (PROMISING/NOT_PROMISING) + confidence (0-100) + rationale
"""
from src.agents.base_agent import BaseAgent


BUSINESS_MODEL_ANALYST_SYSTEM_PROMPT = """You are a business model analyst at a VC firm evaluating a startup's revenue and scalability.

EVALUATION FOCUS:
Assess the following dimensions:

1. REVENUE MODEL CLARITY: Is the business model intelligible? Does the description articulate a clear value proposition and revenue mechanism (subscription, transaction, licensing, etc.)?

2. SCALABILITY: Can this business scale with capital? Are unit economics favorable (i.e., can you acquire customers efficiently and retain them)?

3. CAPITAL EFFICIENCY: Does the initial funding amount suggest investor confidence? Is the round size appropriate for the sector and stage?

4. BUSINESS VIABILITY: Are there red flags (crowded market requiring high CAC, unclear monetisation path, capital-intensive model with no moat)?

CALIBRATION NOTES:
- Clear, defensible business model: +1 decision weight
- Vague or unclear model: -1 decision weight
- Well-sized initial round for the sector and stage: +1 weight
- Underfunded or mismatched round size: -1 weight
- Scalable SaaS/platform: +1 weight; low-margin services: -1 weight

OUTPUT FORMAT:
Respond with valid JSON only — no markdown, no preamble:

{
  "decision": "PROMISING" or "NOT_PROMISING",
  "confidence": <integer 0-100, your confidence in this assessment>,
  "rationale": "<2-3 sentences summarizing business model viability and capital trajectory>"
}

Note: "PROMISING" means strong business model signals; 
"NOT_PROMISING" means model is unclear or capital efficiency appears poor."""


class BusinessModelAnalyst(BaseAgent):
    """Business model analyst for evaluating revenue and scalability."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        """Initialize business model analyst."""
        super().__init__(
            system_prompt=BUSINESS_MODEL_ANALYST_SYSTEM_PROMPT,
            agent_name="business_model_analyst",
            use_cache=True,
            model=model,
        )

    def evaluate(self, startup_profile: str) -> dict:
        """
        Evaluate startup's business model and capital efficiency.

        Args:
            startup_profile: Formatted startup profile string.

        Returns:
            Parsed response: {decision, confidence, rationale}.
            On parse error: {parse_error: True, ...}.
        """
        user_message = (
            "Evaluate the business model and scalability for this startup:\n\n"
            f"{startup_profile}"
        )

        response = self.call(user_message, temperature=0, max_tokens=256)
        return self._parse_json_response(response)
