"""
Feasibility Analyst agent for evaluating product, execution, and traction.

Evaluates:
1. Product viability and differentiation
2. Execution risk and team capability
3. Time-to-traction and milestone signals
4. Market proof-of-concept indicators

Output: Binary decision (PROMISING/NOT_PROMISING) + confidence (0-100) + rationale
"""
import json
from src.agents.base_agent import BaseAgent


FEASIBILITY_ANALYST_SYSTEM_PROMPT = """You are a feasibility analyst at a VC firm evaluating a startup's execution capability and product viability in 2013.

EVALUATION FOCUS:
Assess the following dimensions:

1. PRODUCT VIABILITY: Does the description articulate a differentiated product? Are there clear user needs being addressed? Is the product ambitious yet achievable?

2. EXECUTION RISK: What is the technical and operational complexity? Are there clear go-to-market risks?

3. TRACTION SIGNALS: Are there milestone indicators (product launch, early customers, partnerships, media coverage)? What is the relationship and investment activity (does the company engage with the ecosystem)?

4. TIME-TO-VALUE: How quickly can the company prove product-market fit? Is the funding sufficient to reach key milestones?

CALIBRATION NOTES:
- Clear product differentiation: +1 decision weight
- Vague or "me-too" product: -1 decision weight
- Strong execution signals + engagement: +1 weight
- Limited engagement or traction: -1 weight
- Early traction or proof-of-concept: +1 weight
- Pre-launch with high complexity: -1 weight

OUTPUT FORMAT:
Respond with valid JSON only — no markdown, no preamble:

{
  "decision": "PROMISING" or "NOT_PROMISING",
  "confidence": <integer 0-100, your confidence in this assessment>,
  "rationale": "<2-3 sentences on product viability, execution risk, and traction signals>"
}

Note: "PROMISING" means product and execution appear viable with clear traction path;
"NOT_PROMISING" means high execution risk or weak product signals."""


class FeasibilityAnalyst(BaseAgent):
    """Feasibility analyst for evaluating product and execution."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        """Initialize feasibility analyst."""
        super().__init__(
            system_prompt=FEASIBILITY_ANALYST_SYSTEM_PROMPT,
            agent_name="feasibility_analyst",
            use_cache=True,
            model=model,
        )

    def evaluate(self, startup_profile: str) -> dict:
        """
        Evaluate startup's product viability and execution capability.

        Args:
            startup_profile: Formatted startup profile string.

        Returns:
            Parsed response: {decision, confidence, rationale}.
            On parse error: {parse_error: True, ...}.
        """
        user_message = (
            "Evaluate the product viability and execution feasibility for this startup:\n\n"
            f"{startup_profile}"
        )

        response = self.call(user_message, temperature=0, max_tokens=256)

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
            # Try to repair incomplete JSON (e.g., missing closing brace)
            repaired = text.strip()
            if repaired and not repaired.endswith("}"):
                repaired += "}"
            try:
                result = json.loads(repaired)
                return result
            except json.JSONDecodeError:
                return {
                    "raw_response": response,
                    "parse_error": True,
                }
