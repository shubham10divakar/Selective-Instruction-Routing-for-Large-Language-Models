"""GPT-4o-as-judge scoring: correctness, completeness, instruction adherence."""
from __future__ import annotations

import json

JUDGE_PROMPT = """You are an expert evaluator. Score the following AI response
on three dimensions, each from 0 to 100:

1. CORRECTNESS: Is the response factually correct and logically sound?
2. COMPLETENESS: Does it fully address all aspects of the request?
3. INSTRUCTION_ADHERENCE: Does it follow the domain-specific guidelines
   that were provided in its instructions?

User Request: {request}

Instructions Given to AI: {instructions_summary}

AI Response: {response}

Reference Output: {reference}

Return a JSON object:
{{
  "correctness": <0-100>,
  "completeness": <0-100>,
  "instruction_adherence": <0-100>,
  "overall": <0-100>,
  "reasoning": "<brief explanation>"
}}"""


class LLMJudge:
    """Scores a (request, instructions, response, reference) tuple via an LLM.

    Requires litellm + a configured API key at call time; the import is
    lazy so the rest of the codebase (and tests) work without one.
    """

    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    def score(self, request: str, instructions_summary: str, response: str, reference: str) -> dict:
        from litellm import completion  # lazy import

        prompt = JUDGE_PROMPT.format(
            request=request,
            instructions_summary=instructions_summary,
            response=response,
            reference=reference,
        )
        result = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(result.choices[0].message.content)


class HeuristicJudge:
    """API-free stand-in for LLMJudge, used by `--offline` benchmark runs.

    Scores correctness/completeness as word-overlap against the reference
    output, and instruction_adherence as keyword coverage of the routed
    modules' capabilities (via AdherenceScorer). This is a much weaker
    signal than an actual LLM judge and exists purely so the metrics
    pipeline is exercisable without an API key -- do not use its numbers
    to support paper claims.
    """

    def __init__(self):
        from .adherence import AdherenceScorer

        self._adherence = AdherenceScorer()

    @staticmethod
    def _word_overlap(a: str, b: str) -> float:
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        return 100.0 * len(words_a & words_b) / len(words_a | words_b)

    def score(self, request: str, instructions_summary: str, response: str, reference: str, modules=None) -> dict:
        overlap = self._word_overlap(response, reference)
        adherence = 100.0 * self._adherence.score(response, modules or [])
        overall = (overlap + overlap + adherence) / 3  # weight correctness/completeness equally, both proxied by overlap
        return {
            "correctness": round(overlap, 1),
            "completeness": round(overlap, 1),
            "instruction_adherence": round(adherence, 1),
            "overall": round(overall, 1),
            "reasoning": "heuristic offline score (word overlap + keyword coverage), not an LLM judgment",
        }
