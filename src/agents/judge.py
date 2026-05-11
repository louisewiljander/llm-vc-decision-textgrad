"""
Judge agent for evaluating and comparing investment decisions.
"""
import json
from src.agents.base_agent import BaseAgent


class JudgeAgent(BaseAgent):
    """Judge agent for final investment decision evaluation."""
    
    SYSTEM_PROMPT = """You are a senior investment committee member evaluating investment decisions.
Your role is to assess the quality of invest/pass recommendations and provide a final verdict.

Your analysis should:
1. Review the logic and data supporting the recommendation
2. Identify any potential biases or missed factors
3. Challenge assumptions if needed
4. Provide your independent assessment
5. Recommend PROCEED / CHALLENGE / REJECT

Output your analysis in JSON format with these fields:
- decision_logic_sound: boolean
- key_strengths: list of strings
- key_concerns: list of strings
- missing_factors: list of strings
- final_verdict: PROCEED | CHALLENGE | REJECT
- alternative_view: string
- confidence: 0-100
"""
    
    def __init__(self, use_cache: bool = True):
        """Initialize judge agent."""
        super().__init__(
            system_prompt=self.SYSTEM_PROMPT,
            agent_name="judge",
            use_cache=use_cache
        )
    
    def evaluate_decision(self, investor_analysis: dict, startup_profile: dict) -> dict:
        """
        Review an investor's decision.
        
        Args:
            investor_analysis: Analysis from investor agent
            startup_profile: Original startup profile
            
        Returns:
            Parsed judge response as dictionary
        """
        user_message = f"""Please review this investment decision:

STARTUP PROFILE:
{json.dumps(startup_profile, indent=2)}

INVESTOR ANALYSIS:
{json.dumps(investor_analysis, indent=2)}

Provide your independent assessment in valid JSON format."""
        
        response = self.call(user_message, temperature=0.3)
        
        # Try to parse JSON response
        try:
            # Extract JSON from response if wrapped in markdown
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            else:
                json_str = response
            
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"raw_response": response, "parse_error": True}


if __name__ == "__main__":
    agent = JudgeAgent(use_cache=True)
    
    # Test decision
    investor_analysis = {
        "market_assessment": "Strong market with growing demand",
        "team_assessment": "Experienced founding team with relevant background",
        "financial_health": "Healthy unit economics, 30 month runway",
        "risk_factors": ["Competitive market", "High CAC"],
        "recommendation": "BUY",
        "confidence": 75
    }
    
    profile = {
        "company": "DataViz AI",
        "founded": 2021,
        "funding_raised": 8_500_000,
        "team_size": 18,
        "revenue": 1_200_000
    }
    
    result = agent.evaluate_decision(investor_analysis, profile)
    print(json.dumps(result, indent=2))
