"""
Team Analyst agent for evaluating founder quality and team composition.

Evaluates:
1. Founder background and track record
2. Team size and composition
3. Educational credentials and elite institution alumni
4. Serial entrepreneur signals

Output: Binary decision (PROMISING/NOT_PROMISING) + confidence (0-100) + rationale
"""
from src.agents.base_agent import BaseAgent


TEAM_ANALYST_SYSTEM_PROMPT = """You are a team analyst at a VC firm evaluating founder quality and team composition.

EVALUATION FOCUS:
Assess the following dimensions:

1. FOUNDER QUALITY: Do the founders have relevant domain expertise? Any serial entrepreneur signals (founded previous companies)? Evidence of execution capability?

2. TEAM SIZE & DIVERSITY: Is the team large enough to execute (typically 3-20 for early stage)? Is there diversity in backgrounds (technical, business, domain)?

3. EDUCATIONAL CREDENTIALS: Do team members hold degrees from top-tier universities? This is a proven correlate of venture success.

4. TEAM ENGAGEMENT: Is there evidence of full-time commitment? Any red flags (part-time founders, high turnover indicated by sparse team)?

CALIBRATION NOTES:
- Top-tier university alumni on team: +1 decision weight
- No degree information or non-selective institutions: -1 weight
- Serial entrepreneur founder: +1 weight
- First-time founder with no track record: -1 weight
- Team size 5+: +1 weight; solo founder or very small: -1 weight
- Diverse team (technical + business): +1 weight

OUTPUT FORMAT:
Respond with valid JSON only — no markdown, no preamble:

{
  "decision": "PROMISING" or "NOT_PROMISING",
  "confidence": <integer 0-100, your confidence in this assessment>,
  "rationale": "<2-3 sentences on team quality, credentials, and diversity>"
}

Note: "PROMISING" means strong team signals (credentials, experience, size);
"NOT_PROMISING" means weak team or execution concerns.
Remember: Team members are anonymized as [FOUNDER_1], etc. Evaluate based on aggregate signals."""


class TeamAnalyst(BaseAgent):
    """Team analyst for evaluating founder quality and team composition."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        """Initialize team analyst."""
        super().__init__(
            system_prompt=TEAM_ANALYST_SYSTEM_PROMPT,
            agent_name="team_analyst",
            use_cache=True,
            model=model,
        )

    def evaluate(self, startup_profile: str) -> dict:
        """
        Evaluate startup's team quality and composition.

        Args:
            startup_profile: Formatted startup profile string.

        Returns:
            Parsed response: {decision, confidence, rationale}.
            On parse error: {parse_error: True, ...}.
        """
        user_message = (
            "Evaluate the team quality and composition for this startup:\n\n"
            f"{startup_profile}"
        )

        response = self.call(user_message, temperature=0, max_tokens=256)
        return self._parse_json_response(response)
