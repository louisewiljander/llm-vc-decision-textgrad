"""
Base agent class using cached LLM client.
"""
import json
from pathlib import Path
from src.utils.litellm_client import CachedLLMClient


class BaseAgent:
    """Base agent class for VC decision-making agents."""
    
    def __init__(
        self,
        system_prompt: str,
        agent_name: str = "base_agent",
        use_cache: bool = True,
        model: str = "claude-haiku-4-5-20251001"
    ):
        """
        Initialize base agent.
        
        Args:
            system_prompt: System prompt for the agent
            agent_name: Name identifier for the agent
            use_cache: Whether to use prompt caching (Anthropic only).
                       Note: Cache is disabled for non-Anthropic models.
            model: Model identifier.
                   Anthropic: "claude-haiku-4-5-20251001"
                   Ollama: "ollama/llama2", "ollama/qwen", etc.
                   Others: Per LiteLLM documentation
        """
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.use_cache = use_cache
        self.llm_client = CachedLLMClient(model=model, use_cache=use_cache)
    
    def call(
        self,
        user_message: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Call the agent with a user message.
        
        Args:
            user_message: User message/prompt
            max_tokens: Maximum response tokens
            temperature: Response temperature
            **kwargs: Additional API arguments
            
        Returns:
            Agent response text
        """
        return self.llm_client.call(
            system_prompt=self.system_prompt,
            user_message=user_message,
            agent_name=self.agent_name,
            use_cache=self.use_cache,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
    
    @staticmethod
    def _parse_json_response(response: str, fallback_extra: dict | None = None) -> dict:
        """
        Strip markdown fences, parse JSON, attempt single-brace repair on failure.

        Args:
            response: Raw LLM response string.
            fallback_extra: Extra fields merged into the error dict on total parse failure.

        Returns:
            Parsed dict, or {raw_response, parse_error: True, ...fallback_extra} on failure.
        """
        # Strip markdown if present
        text = response.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.rsplit("```", 1)[0]

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            # Try to repair incomplete JSON (e.g., missing closing brace)
            repaired = text.strip()
            if repaired and not repaired.endswith("}"):
                repaired += "}"
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                error: dict = {"raw_response": response, "parse_error": True}
                if fallback_extra:
                    error.update(fallback_extra)
                return error

    def get_logs(self, limit: int = 50) -> list:
        """Get logs for this agent."""
        return self.llm_client.get_logs(agent_name=self.agent_name, limit=limit)


# Example usage and test
if __name__ == "__main__":
    # Create an agent with a system prompt
    investor_prompt = """You are an experienced venture capital investor evaluating startup profiles.
Your task is to assess the likelihood of success based on company metrics and market position.
Provide a structured analysis with investment recommendation."""
    
    agent = BaseAgent(
        system_prompt=investor_prompt,
        agent_name="investor_agent"
    )
    
    # Example call
    test_message = """
    Company: TechCorp
    - Founded: 2020
    - Funding: $5M Series A
    - Team size: 12
    - Market: B2B SaaS
    
    Should we invest?
    """
    
    response = agent.call(test_message, temperature=0.3)
    print("Response:", response)
    
    # Show cache stats
    stats = agent.llm_client.get_cache_stats()
    print("\nCache Statistics:")
    print(f"  Total calls: {stats['total_calls']}")
    print(f"  Cache read tokens: {stats['cache_read_tokens']}")
    print(f"  Estimated savings: ${stats['estimated_savings_usd']:.4f}")
