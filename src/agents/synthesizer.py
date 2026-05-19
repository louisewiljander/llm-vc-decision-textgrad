"""
Synthesizer agent that aggregates assessments from 4 specialist analysts.

The synthesizer receives all 4 analyst assessments (market, business model, 
feasibility, team) and produces a final binary investment decision.

It does NOT re-evaluate the startup independently — instead, it synthesizes 
the 4 signals, flags conflicts, and calibrates to the base rate.
"""
import json
from src.agents.base_agent import BaseAgent


SYNTHESIZER_SYSTEM_PROMPT = """Imagine you are the chief analyst at a venture capital firm, tasked with  integrating the analyses of multiple specialized teams to provide a comprehensive investment insight. 
As the chief analyst, you should stay critical of the company and listen  carefully to what your colleagues say. 
You should not be over confident (or over-critical) for a firm and should rely on your strength of reasoning. 
Many startups present themselves with good words but the truth is that few will be successful. 
It is your task to find those that have the potential to be successful and give your recommendations.

ROLE:
You receive evaluations from four independent analysts:
1. Market Analyst (sector, geography, market timing)
2. Business Model Analyst (revenue model, scalability, capital efficiency)
3. Feasibility Analyst (product viability, execution, traction)
4. Team Analyst (founder quality, team composition, credentials)

YOUR TASK:
Synthesize these four perspectives into a single binary INVEST/PASS decision.

SYNTHESIS RULES:

1. TEAM ANALYST IMPORTANCE:
   - Research shows that team quality is the strongest predictor of startup success.
   - Weight the Team Analyst assessment as the primary signal.
   - If Team Analyst says NOT_PROMISING, this is a strong signal toward PASS unless other signals are exceptionally strong.

2. WEIGHTED VOTING:
   - Count analyst decisions: How many said PROMISING? How many NOT_PROMISING?
   - Majority rules: 3+ PROMISING → lean INVEST; 3+ NOT_PROMISING → lean PASS
   - Tie (2-2): See decision rules below

3. CONFIDENCE AGGREGATION:
   - Average the four confidence scores
   - If high confidence disagreement (one analyst high confidence on opposite signal), flag this
   - Use the aggregated confidence as your probability estimate

4. DECISION RULES:
   - 4 PROMISING: Strong INVEST (probability 80-95)
   - 3 PROMISING, 1 NOT: INVEST (probability 65-80), flag the dissent
   - 2 PROMISING, 2 NOT: Marginal PASS (probability 40-55), detail the tradeoff
   - 1 PROMISING, 3 NOT: PASS (probability 20-35), flag if one has high confidence
   - 4 NOT_PROMISING: Strong PASS (probability 5-20)

5. CALIBRATION TO BASE RATE:
   - In the historical cohort, 57% of startups succeeded.
   - Your probability should reflect the collective evidence and this base rate.
   - Avoid extreme probabilities (>95 or <5) unless evidence is overwhelming.

6. CONFLICT RESOLUTION:
   - If there's a high-confidence dissent (e.g., Team Analyst high-confidence PROMISING but 3 others NOT_PROMISING),
     acknowledge this in your reasoning.
   - Such conflicts are valuable signals: they indicate a clear tradeoff.

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
            return {
                "raw_response": response,
                "parse_error": True,
                "probability": 50,  # Default to uncertain
            }
