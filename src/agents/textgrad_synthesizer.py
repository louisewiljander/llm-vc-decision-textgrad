"""
TextGrad-optimized synthesizer agent.

This agent extends SynthesizerAgent to use the optimized prompt from TextGrad training
instead of the default prompt. The optimized prompt has been iteratively improved
through text-based gradient descent on a small validation set.
"""
import json
from pathlib import Path
from src.agents.synthesizer import SynthesizerAgent


class TextGradSynthesizer(SynthesizerAgent):
    """Synthesizer agent using TextGrad-optimized system prompt."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        """
        Initialize TextGrad synthesizer agent.
        
        Loads the optimized prompt from results/textgrad_validation/final_synthesizer_prompt.txt
        If the file doesn't exist, falls back to default SynthesizerAgent prompt.
        
        Args:
            model: LLM model to use
        """
        # Try to load optimized prompt
        optimized_prompt = self._load_optimized_prompt()
        
        # Initialize parent with optimized prompt (or default if not found)
        BaseAgent = SynthesizerAgent.__bases__[0]  # Get BaseAgent
        BaseAgent.__init__(
            self,
            system_prompt=optimized_prompt,
            agent_name="textgrad_synthesizer",
            use_cache=True,
            model=model,
        )
    
    @staticmethod
    def _load_optimized_prompt() -> str:
        """
        Load the optimized prompt from TextGrad training.
        
        Returns:
            Optimized prompt string, or default prompt if file not found
        """
        from src.agents.synthesizer import SYNTHESIZER_SYSTEM_PROMPT
        
        tg_base = Path(__file__).resolve().parents[2] / "results" / "textgrad_validation"
        latest  = tg_base / "latest"
        tg_dir  = latest.resolve() if latest.exists() else tg_base
        prompt_file = tg_dir / "final_synthesizer_prompt.txt"
        
        if prompt_file.exists():
            with open(prompt_file, 'r') as f:
                prompt = f.read()
            print(f"✓ Loaded TextGrad-optimized synthesizer prompt ({len(prompt)} chars)")
            return prompt
        else:
            print(f"⚠️  TextGrad optimized prompt not found at {prompt_file}")
            print(f"   Falling back to default synthesizer prompt")
            return SYNTHESIZER_SYSTEM_PROMPT
