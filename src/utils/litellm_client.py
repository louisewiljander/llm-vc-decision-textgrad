"""
Model-agnostic LLM client using LiteLLM.

Supports multiple model providers:
  - Anthropic (Claude): claude-haiku-4-5-20251001, etc.
  - Ollama (local): ollama/llama2, ollama/qwen, etc.
  - HuggingFace: huggingface/model-name
  - Others: OpenAI, Cohere, Replicate via LiteLLM

Logs all API calls to CSV and JSON (JSONL) formats.
"""
import json
import csv
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class CachedLLMClient:
    """
    Model-agnostic LLM client with logging and cost tracking.
    
    Supports both LiteLLM (multi-model) and direct Anthropic API.
    Falls back gracefully if LiteLLM unavailable.
    """

    # Anthropic pricing (per 1M tokens)
    ANTHROPIC_PRICING = {
        "input": 3.0,           # $3 per 1M input tokens
        "output": 15.0,         # $15 per 1M output tokens
        "cache_creation": 3.75, # 25% of input cost
        "cache_read": 0.30,     # 10% of input cost
    }

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        use_cache: bool = True,
        log_dir: Optional[str] = None,
    ):
        """
        Initialize the client.

        Args:
            model: Model identifier.
                   - Anthropic: "claude-haiku-4-5-20251001"
                   - Ollama: "ollama/llama2", "ollama/qwen", etc.
                   - HuggingFace: "huggingface/model-name"
                   - Others: Per LiteLLM documentation
            use_cache: Whether to use prompt caching (Anthropic only).
            log_dir: Directory for log files (CSV and JSONL). Defaults to results/logs/.
        """
        self.model = model
        self.use_cache = use_cache

        # Parse model provider
        self.provider = self._infer_provider(model)

        # Initialize logging paths
        if log_dir is None:
            log_dir = "results/logs"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "api_calls.csv"
        self.jsonl_path = self.log_dir / "api_calls.jsonl"
        
        # Initialize CSV header if file doesn't exist
        self._init_csv_header()

        # Initialize API clients
        if self.provider == "anthropic":
            if ANTHROPIC_AVAILABLE:
                self.anthropic_client = Anthropic()
            else:
                raise ImportError(
                    "Anthropic SDK required for Claude models. "
                    "Install: pip install anthropic"
                )
        elif self.provider == "litellm":
            if not LITELLM_AVAILABLE:
                raise ImportError(
                    "LiteLLM required for this model. "
                    "Install: pip install litellm"
                )
            # LiteLLM handles model routing internally

        # Cache stats
        self.cache_stats = {
            "total_calls": 0,
            "total_input_tokens": 0,
            "cache_created_tokens": 0,
            "cache_read_tokens": 0,
            "total_cost_usd": 0.0,
        }

    def _infer_provider(self, model: str) -> str:
        """Infer model provider from model string."""
        if model.startswith("ollama/"):
            return "litellm"  # Use LiteLLM for Ollama
        elif model.startswith("huggingface/"):
            return "litellm"  # Use LiteLLM for HuggingFace
        elif model.startswith("claude"):
            return "anthropic"
        else:
            # Default to LiteLLM for unknown models
            return "litellm"

    def _init_csv_header(self) -> None:
        """Initialize CSV file with header if it doesn't exist."""
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "model",
                    "agent_name",
                    "system_prompt_hash",
                    "input_tokens",
                    "output_tokens",
                    "cache_creation_tokens",
                    "cache_read_tokens",
                    "total_cost_usd",
                    "user_message",
                    "assistant_response",
                    "error_message",
                ])

    def call(
        self,
        system_prompt: str,
        user_message: str,
        agent_name: str = "unknown",
        use_cache: bool = True,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """
        Call the LLM with system and user prompts.

        Args:
            system_prompt: System instruction.
            user_message: User query.
            agent_name: Name of the agent making the call (for logging).
            use_cache: Whether to use prompt caching (Anthropic only).
            max_tokens: Max output tokens.
            temperature: Sampling temperature.
            **kwargs: Additional model-specific parameters.

        Returns:
            Assistant response text.
        """
        system_prompt_hash = hashlib.sha256(system_prompt.encode()).hexdigest()[:8]

        try:
            if self.provider == "anthropic":
                response = self._call_anthropic(
                    system_prompt, user_message, use_cache, max_tokens, temperature
                )
            else:  # litellm
                response = self._call_litellm(
                    system_prompt, user_message, max_tokens, temperature, **kwargs
                )

            # Log the call
            self._log_call(
                agent_name=agent_name,
                system_prompt_hash=system_prompt_hash,
                user_message=user_message,
                assistant_response=response.get("content", ""),
                input_tokens=response.get("input_tokens", 0),
                output_tokens=response.get("output_tokens", 0),
                cache_creation_tokens=response.get("cache_creation_tokens", 0),
                cache_read_tokens=response.get("cache_read_tokens", 0),
                cost_usd=response.get("cost_usd", 0.0),
                error_message=None,
            )

            # Update cache stats
            self.cache_stats["total_calls"] += 1
            self.cache_stats["total_input_tokens"] += response.get("input_tokens", 0)
            self.cache_stats["cache_created_tokens"] += response.get(
                "cache_creation_tokens", 0
            )
            self.cache_stats["cache_read_tokens"] += response.get("cache_read_tokens", 0)
            self.cache_stats["total_cost_usd"] += response.get("cost_usd", 0.0)

            return response["content"]

        except Exception as e:
            error_msg = str(e)
            self._log_call(
                agent_name=agent_name,
                system_prompt_hash=system_prompt_hash,
                user_message=user_message,
                assistant_response="",
                input_tokens=0,
                output_tokens=0,
                cache_creation_tokens=0,
                cache_read_tokens=0,
                cost_usd=0.0,
                error_message=error_msg,
            )
            raise

    def _call_anthropic(
        self,
        system_prompt: str,
        user_message: str,
        use_cache: bool,
        max_tokens: int,
        temperature: float,
    ) -> dict:
        """Call Anthropic API with optional prompt caching."""
        cache_control = (
            {"type": "ephemeral"} if use_cache else None
        )

        # Build system message with cache control
        system = [
            {
                "type": "text",
                "text": system_prompt,
            }
        ]
        if cache_control:
            system[0]["cache_control"] = cache_control

        response = self.anthropic_client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[
                {
                    "role": "user",
                    "content": user_message,
                }
            ],
        )

        # Extract usage info
        usage = response.usage
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", 0)
        cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0)

        # Calculate cost
        cost_usd = self._calculate_anthropic_cost(
            input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens
        )

        return {
            "content": response.content[0].text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_tokens": cache_creation_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cost_usd": cost_usd,
        }

    def _call_litellm(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> dict:
        """Call via LiteLLM (supports Ollama, HuggingFace, etc.)."""
        response = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        # Extract content from ModelResponse object (standard LiteLLM format)
        content = ""
        input_tokens = 0
        output_tokens = 0
        
        # LiteLLM returns a ModelResponse object with choices
        if hasattr(response, 'choices') and response.choices:
            try:
                message = response.choices[0].message
                # Try main content field
                if hasattr(message, 'content') and message.content:
                    content = message.content
                # Fallback to reasoning_content (for models like DeepSeek-R1)
                elif hasattr(message, 'reasoning_content') and message.reasoning_content:
                    content = message.reasoning_content
            except (AttributeError, IndexError, TypeError):
                content = ""
        
        # Extract usage info
        if hasattr(response, 'usage'):
            usage = response.usage
            # Handle both dict and object usage formats
            if isinstance(usage, dict):
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
            else:
                input_tokens = getattr(usage, "prompt_tokens", 0)
                output_tokens = getattr(usage, "completion_tokens", 0)

        # Cost tracking: LiteLLM may attach provider cost metadata for paid APIs.
        cost_usd = 0.0
        if not self.model.startswith("ollama/"):
            hidden_params = getattr(response, "_hidden_params", None) or {}
            maybe_cost = hidden_params.get("response_cost", None)
            if maybe_cost is None:
                maybe_cost = getattr(response, "response_cost", None)
            if maybe_cost is not None:
                try:
                    cost_usd = float(maybe_cost)
                except (TypeError, ValueError):
                    cost_usd = 0.0
        return {
            "content": content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
            "cost_usd": cost_usd,
        }

    @staticmethod
    def _calculate_anthropic_cost(
        input_tokens: int,
        output_tokens: int,
        cache_creation_tokens: int,
        cache_read_tokens: int,
    ) -> float:
        """Calculate total API cost using Anthropic pricing."""
        cost = 0.0

        # Input tokens not in cache
        regular_input = input_tokens - cache_creation_tokens - cache_read_tokens
        cost += regular_input * (CachedLLMClient.ANTHROPIC_PRICING["input"] / 1_000_000)

        # Cache creation (input tokens used to create cache)
        cost += cache_creation_tokens * (
            CachedLLMClient.ANTHROPIC_PRICING["cache_creation"] / 1_000_000
        )

        # Cache read (input tokens read from cache)
        cost += cache_read_tokens * (
            CachedLLMClient.ANTHROPIC_PRICING["cache_read"] / 1_000_000
        )

        # Output tokens
        cost += output_tokens * (CachedLLMClient.ANTHROPIC_PRICING["output"] / 1_000_000)

        return cost

    def _log_call(
        self,
        agent_name: str,
        system_prompt_hash: str,
        user_message: str,
        assistant_response: str,
        input_tokens: int,
        output_tokens: int,
        cache_creation_tokens: int,
        cache_read_tokens: int,
        cost_usd: float,
        error_message: Optional[str],
    ) -> None:
        """Log API call to both CSV and JSONL."""
        timestamp = datetime.utcnow().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "model": self.model,
            "agent_name": agent_name,
            "system_prompt_hash": system_prompt_hash,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_tokens": cache_creation_tokens,
            "cache_read_tokens": cache_read_tokens,
            "total_cost_usd": cost_usd,
            "user_message": user_message,
            "assistant_response": assistant_response,
            "error_message": error_message,
        }
        
        # Log to CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                self.model,
                agent_name,
                system_prompt_hash,
                input_tokens,
                output_tokens,
                cache_creation_tokens,
                cache_read_tokens,
                cost_usd,
                user_message[:1000],  # Truncate for readability (increased from 500)
                assistant_response[:2000],  # Truncate for readability (increased from 500)
                error_message,
            ])
        
        # Log to JSONL (full messages)
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    

    def get_cache_stats(self) -> dict:
        """Get cache and cost statistics."""
        cache_total_input = (
            self.cache_stats["total_input_tokens"]
            + self.cache_stats["cache_created_tokens"]
            + self.cache_stats["cache_read_tokens"]
        )
        cache_usage_pct = (
            (
                self.cache_stats["cache_read_tokens"]
                / cache_total_input
                * 100
            )
            if cache_total_input > 0
            else 0.0
        )

        return {
            "total_calls": self.cache_stats["total_calls"],
            "total_input_tokens": self.cache_stats["total_input_tokens"],
            "cache_created_tokens": self.cache_stats["cache_created_tokens"],
            "cache_read_tokens": self.cache_stats["cache_read_tokens"],
            "cache_usage_percentage": round(cache_usage_pct, 1),
            "total_cost_usd": round(self.cache_stats["total_cost_usd"], 4),
            "estimated_savings_usd": self._calculate_cache_savings(),
        }

    def _calculate_cache_savings(self) -> float:
        """Estimate savings from prompt caching."""
        # Savings: difference between regular input cost and cache cost
        regular_cost = self.cache_stats["cache_read_tokens"] * (
            self.ANTHROPIC_PRICING["input"] / 1_000_000
        )
        cache_cost = self.cache_stats["cache_read_tokens"] * (
            self.ANTHROPIC_PRICING["cache_read"] / 1_000_000
        )
        savings = regular_cost - cache_cost
        return round(savings, 4)

    def get_logs(self, agent_name: Optional[str] = None, limit: int = 50) -> list[dict]:
        """Retrieve logs from JSONL file."""
        if not self.jsonl_path.exists():
            return []
        
        logs = []
        with open(self.jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    log_entry = json.loads(line)
                    if agent_name is None or log_entry.get("agent_name") == agent_name:
                        logs.append(log_entry)
        
        # Return most recent logs (limit)
        return logs[-limit:] if len(logs) > limit else logs
