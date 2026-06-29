"""LLM clients backed by Ollama's OpenAI-compatible API.

Replaces the old per-machine vLLM endpoints (ports 8002/8003) with a single
Ollama endpoint serving both the generator and the judge by model name.
"""
from __future__ import annotations

from openai import OpenAI

from .config import cfg


def _client(config=cfg) -> OpenAI:
    return OpenAI(base_url=config.serving.base_url, api_key=config.serving.api_key)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) to avoid context overflow."""
    return max(1, len(text) // 4)


def safe_max_tokens(prompt: str, desired: int = 1000, ctx: int | None = None,
                    buffer: int = 300, floor: int = 128) -> int:
    ctx = ctx or cfg.serving.generator_context_limit
    available = ctx - estimate_tokens(prompt) - buffer
    if available <= 0:
        raise ValueError(f"Prompt too long: ~{estimate_tokens(prompt)} tokens vs ctx {ctx}.")
    return max(floor, min(desired, available))


class Generator:
    def __init__(self, config=cfg):
        self.client = _client(config)
        self.model = config.serving.generator_model
        self.temperature = config.serving.temperature

    def complete(self, prompt: str, desired_output_tokens: int = 1000, **kw) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=safe_max_tokens(prompt, desired_output_tokens),
            temperature=self.temperature,
            **kw,
        )
        return resp.choices[0].message.content

    def chat(self, messages, tools=None, tool_choice="auto", desired_output_tokens=1000):
        """Lower-level chat call exposing messages + tools (for agentic flows).
        Returns the raw message object (which may contain .tool_calls)."""
        kwargs = dict(
            model=self.model,
            messages=messages,
            max_tokens=desired_output_tokens,
            temperature=self.temperature,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message


class Judge:
    def __init__(self, config=cfg):
        self.client = _client(config)
        self.model = config.serving.judge_model

    def score(self, prompt: str, system: str, desired_output_tokens: int = 500) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=desired_output_tokens,
            temperature=0,
            top_p=1,
            extra_body={"response_format": {"type": "json_object"}},
        )
        return resp.choices[0].message.content.strip()
