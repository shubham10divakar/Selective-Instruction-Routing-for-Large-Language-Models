"""Unified LLM interface over litellm, so the same experiment code targets
gpt-4o, Claude, or a local Ollama model just by changing a model string."""
from __future__ import annotations


SYSTEM_TEMPLATE = """You are a helpful AI assistant. Follow the instructions below
when they are relevant to the user's request.

{instructions}"""


class LLMBackend:
    def __init__(self, model: str, temperature: float = 0.2, max_tokens: int = 1024):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, instructions_context: str, request: str) -> str:
        """Generate a response, injecting `instructions_context` as system content."""
        from litellm import completion  # lazy import

        system_content = (
            SYSTEM_TEMPLATE.format(instructions=instructions_context)
            if instructions_context
            else "You are a helpful AI assistant."
        )
        response = completion(
            model=self.model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": request},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content


class MockLLMBackend:
    """Deterministic, API-free stand-in for LLMBackend.

    Used by `scripts/run_benchmark.py --offline` (the default) so the full
    pipeline -- routing, composition, "generation", scoring -- is runnable
    and testable with no API key. It does not produce a real answer; it
    just echoes enough of the request/instructions to let offline metrics
    (routing precision/recall, adherence keyword coverage) exercise the
    rest of the pipeline honestly.
    """

    model = "mock"

    def generate(self, instructions_context: str, request: str) -> str:
        preview = instructions_context[:400]
        return (
            f"[mock response] Addressing: {request}\n"
            f"Applying guidance from routed instructions: {preview}"
        )
