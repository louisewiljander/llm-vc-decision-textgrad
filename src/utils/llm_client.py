"""
Centralized LLM client with prompt caching and JSONL response logging.

Each API call is appended as a single JSON line to:
    results/logs/api_calls.jsonl

The log is designed to be consumed by:
  - Cost and cache analysis (get_cache_stats)
  - The JudgeAgent, which reads investor decisions from the log and
    uses them as input for feedback and scoring
  - Pandas: pd.read_json("results/logs/api_calls.jsonl", lines=True)
"""
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic  # noqa: F401  (used at runtime via self.client)
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")


class CachedLLMClient:
    """
    Anthropic LLM client with prompt caching and JSONL response logging.

    Features
    --------
    - Prompt caching: marks system prompts with cache_control so Anthropic
      reuses them across calls, saving ~90% on system-prompt input tokens.
    - JSONL logging: every call (success or error) is appended to a single
      file. Each line is a self-contained JSON record — easy to tail, grep,
      and load into pandas or pass to the JudgeAgent.
    """

    # Pricing for Claude Haiku 4.5 (per 1M tokens, as of 2026-04)
    PRICES = {
        "input":          1.00,   # $/1M tokens
        "output":         5.00,
        "cache_creation": 1.25,   # 25% surcharge on first cache write
        "cache_read":     0.10,   # 90% discount on cache hits
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        log_path: Optional[str] = None,
        model: str = "claude-haiku-4-5-20251001",
    ):
        """
        Args:
            api_key:  Anthropic API key (falls back to ANTHROPIC_API_KEY env var).
            log_path: Path to the JSONL log file. Defaults to
                      results/logs/api_calls.jsonl relative to the project root.
            model:    Model identifier.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found in environment or passed as argument."
            )

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model

        if log_path is None:
            log_path = (
                Path(__file__).parent.parent.parent / "results" / "logs" / "api_calls.jsonl"
            )
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def call(
        self,
        system_prompt: str,
        user_message: str,
        agent_name: str = "default",
        use_cache: bool = True,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """
        Make a cached API call and append the result to the JSONL log.

        Args:
            system_prompt: System prompt (cached on first use per session).
            user_message:  User-turn content.
            agent_name:    Label for this agent in the log (e.g. "investor",
                           "judge"). The JudgeAgent filters by this field.
            use_cache:     Whether to send cache_control headers.
            max_tokens:    Maximum response tokens.
            temperature:   Sampling temperature.
            **kwargs:      Passed through to client.messages.create().

        Returns:
            The assistant's response text.
        """
        system = [{"type": "text", "text": system_prompt}]
        if use_cache:
            system[0]["cache_control"] = {"type": "ephemeral"}

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_message}],
                temperature=temperature,
                **kwargs,
            )
            self._append_log(agent_name, system_prompt, user_message, response)
            return response.content[0].text if response.content else ""

        except Exception as exc:
            self._append_log(
                agent_name, system_prompt, user_message,
                response=None, error_message=str(exc),
            )
            raise

    def get_logs(
        self,
        agent_name: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Read log records from the JSONL file.

        Args:
            agent_name: If provided, return only records for this agent.
            limit:      Maximum number of records to return (most recent first).

        Returns:
            List of log record dicts.
        """
        if not self.log_path.exists():
            return []

        records = []
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if agent_name is None or record.get("agent_name") == agent_name:
                    records.append(record)

        # Most-recent first, then truncate
        return records[-limit:][::-1]

    def get_cache_stats(self) -> dict:
        """
        Compute aggregate cost and cache statistics from the JSONL log.

        Returns:
            Dict with: total_calls, total_input_tokens, cache_created_tokens,
            cache_read_tokens, cache_usage_percentage, total_cost_usd,
            estimated_savings_usd.
        """
        records = self.get_logs(limit=0)  # all records

        total_calls = 0
        total_input = 0
        cache_created = 0
        cache_read = 0
        total_cost = 0.0

        for r in records:
            if r.get("error_message"):
                continue
            total_calls += 1
            total_input += r.get("input_tokens", 0)
            cache_created += r.get("cache_creation_input_tokens", 0)
            cache_read += r.get("cache_read_input_tokens", 0)
            total_cost += r.get("total_cost_usd", 0.0)

        # Savings: cache reads are charged at 10% vs full input price
        savings = (cache_read * (self.PRICES["input"] - self.PRICES["cache_read"])) / 1_000_000
        cached_tokens = cache_created + cache_read

        return {
            "total_calls": total_calls,
            "total_input_tokens": total_input,
            "cache_created_tokens": cache_created,
            "cache_read_tokens": cache_read,
            "cache_usage_percentage": (
                cached_tokens / total_input * 100 if total_input else 0.0
            ),
            "total_cost_usd": round(total_cost, 6),
            "estimated_savings_usd": round(savings, 6),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_creation_tokens: int,
        cache_read_tokens: int,
    ) -> float:
        p = self.PRICES
        return (
            input_tokens          * p["input"]          / 1_000_000
            + output_tokens       * p["output"]         / 1_000_000
            + cache_creation_tokens * p["cache_creation"] / 1_000_000
            + cache_read_tokens   * p["cache_read"]     / 1_000_000
        )

    def _append_log(
        self,
        agent_name: str,
        system_prompt: str,
        user_message: str,
        response,
        error_message: Optional[str] = None,
    ) -> None:
        """Append one JSON record to the JSONL log file."""
        input_tokens = 0
        output_tokens = 0
        cache_creation_tokens = 0
        cache_read_tokens = 0
        assistant_response = ""
        stop_reason = ""

        if response is not None:
            usage = getattr(response, "usage", None)
            if usage:
                input_tokens = getattr(usage, "input_tokens", 0) or 0
                output_tokens = getattr(usage, "output_tokens", 0) or 0
                cache_creation_tokens = (
                    getattr(usage, "cache_creation_input_tokens", 0) or 0
                )
                cache_read_tokens = (
                    getattr(usage, "cache_read_input_tokens", 0) or 0
                )
            if response.content:
                assistant_response = response.content[0].text
            stop_reason = getattr(response, "stop_reason", "")

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": self.model,
            "agent_name": agent_name,
            "system_prompt_hash": hashlib.md5(system_prompt.encode()).hexdigest()[:8],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": cache_creation_tokens,
            "cache_read_input_tokens": cache_read_tokens,
            "total_cost_usd": self._calculate_cost(
                input_tokens, output_tokens,
                cache_creation_tokens, cache_read_tokens,
            ),
            "user_message": user_message,
            "assistant_response": assistant_response,
            "stop_reason": stop_reason,
            "error_message": error_message,
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
