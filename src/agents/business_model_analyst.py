"""
Business Model Analyst agent for evaluating revenue model and scalability.

Evaluates:
1. Revenue model clarity and viability
2. Unit economics and scalability
3. Funding structure and capital efficiency
4. Business viability from available signals

Output: Binary decision (PROMISING/NOT_PROMISING) + confidence (0-100) + rationale
"""
import json
from src.agents.base_agent import BaseAgent


BUSINESS_MODEL_ANALYST_SYSTEM_PROMPT = """You are a business model analyst evaluating a startup's revenue and scalability.

EVALUATION FOCUS:
Assess the following dimensions:

1. REVENUE MODEL CLARITY: Is the business model intelligible? Does the description articulate a clear value proposition and revenue mechanism (subscription, transaction, licensing, etc.)?

2. SCALABILITY: Can this business scale with capital? Are unit economics favorable (i.e., can you acquire customers efficiently and retain them)?

3. CAPITAL EFFICIENCY: How much capital has been raised? Does the funding trajectory suggest investor confidence? Are later rounds tracking well?

4. BUSINESS VIABILITY: Are there red flags (unsustainable burn, failing to raise follow-on rounds, crowded market requiring high CAC)?

CALIBRATION NOTES:
- Clear, defensible business model: +1 decision weight
- Vague or unclear model: -1 decision weight
- High capital raised with strong follow-on funding: +1 weight
- Difficulty raising (gaps in funding history): -1 weight
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

        response = self.call(user_message, temperature=0.2, max_tokens=256)

        # Strip markdown if present
        text = response.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.rsplit("```", 1)[0]

        try:
            result = json.loads(text.strip())
            return result
        except json.JSONDecodeError:
            return {
                "raw_response": response,
                "parse_error": True,
            }
