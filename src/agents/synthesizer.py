"""
Synthesizer agent that aggregates assessments from 4 specialist analysts.

The synthesizer receives all 4 analyst assessments (market, business model, 
feasibility, team) and produces a final binary investment decision.

It does NOT re-evaluate the startup independently — instead, it synthesizes 
the 4 signals, flags conflicts, and calibrates to the base rate.
"""
import json
from src.agents.base_agent import BaseAgent


SYNTHESIZER_SYSTEM_PROMPT = """You are the chief analyst at a venture capital firm. You receive evaluation reports from four independent specialist analysts and must synthesize their perspectives into a single investment recommendation.

Stay critical. Most startups will not succeed — your task is to identify the rare exceptions.

ANALYSTS:
1. Market Analyst (sector, geography, market timing)
2. Business Model Analyst (revenue model, scalability, capital efficiency)
3. Feasibility Analyst (product viability, execution, traction)
4. Team Analyst (founder quality, team composition, credentials)

YOUR TASK:
Weigh the four analyst reports and produce a single binary INVEST/PASS decision with a calibrated probability of successful exit.

OUTPUT FORMAT:
Respond with valid JSON only — no markdown, no preamble:

{
  "decision": "INVEST" or "PASS",
  "probability": <integer 0-100, your estimated probability of successful exit>,
  "num_promising": <integer 0-4, number of analysts who said PROMISING>,
  "num_not_promising": <integer 0-4>,
  "avg_confidence": <float 0-100, average of the four analyst confidences>,
  "conflicts": "<string describing any high-confidence disagreements, or 'None'>",
  "reasoning": "<2-3 sentences synthesizing the four assessments and explaining the decision>"
}"""


class SynthesizerAgent(BaseAgent):
    """Synthesizer agent for aggregating four analyst assessments."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        """Initialize synthesizer agent."""
        super().__init__(
            system_prompt=SYNTHESIZER_SYSTEM_PROMPT,
            agent_name="synthesizer",
            use_cache=True,
            model=model,
        )

    def synthesize(
        self,
        startup_profile: str,
        analyst_assessments: list[dict],
    ) -> dict:
        """
        Synthesize four analyst assessments into a final decision.

        Args:
            startup_profile: Original startup profile string.
            analyst_assessments: List of 4 dicts from analyst.evaluate():
                                [market_assessment, business_assessment, 
                                 feasibility_assessment, team_assessment]

        Returns:
            Synthesizer output: {decision, probability, reasoning, ...}
            On parse error: {parse_error: True, ...}.
        """
        # Format analyst assessments for the prompt
        analyst_names = ["Market", "Business Model", "Feasibility", "Team"]
        assessments_text = "ANALYST ASSESSMENTS:\n"
        for name, assessment in zip(analyst_names, analyst_assessments):
            assessments_text += f"\n{name} Analyst:\n"
            if assessment.get("parse_error"):
                assessments_text += "  [PARSE ERROR - assessment unavailable]\n"
            else:
                assessments_text += (
                    f"  Decision: {assessment.get('decision', 'UNKNOWN')}\n"
                    f"  Confidence: {assessment.get('confidence', 0)}/100\n"
                    f"  Rationale: {assessment.get('rationale', 'N/A')}\n"
                )

        user_message = (
            f"{assessments_text}\n\n"
            f"STARTUP PROFILE:\n{startup_profile}\n\n"
            "Provide your synthesized investment decision in valid JSON format."
        )

        response = self.call(user_message, temperature=0.2, max_tokens=384)

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
                    "probability": 50,  # Default to uncertain
                }
