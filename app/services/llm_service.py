from __future__ import annotations
"""
app/services/llm_service.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
LLM inference service using Groq's LLaMA3 API.

Responsibilities:
  - Single-turn completion (system prompt + user message)
  - Multi-turn chat completion (system + conversation history)
  - JSON-mode completion (structured outputs for triage scoring)
  - Async retry with exponential backoff on transient Groq errors

Usage:
    llm = LLMService()
    response = await llm.complete(system="You are...", user="Patient says...")
    reply = await llm.chat(system="...", history=[{"role":"user","content":"..."}])
"""

import asyncio
import json
from typing import Any

from groq import AsyncGroq, RateLimitError, APIStatusError
from groq.types.chat import ChatCompletion

from app.config import settings
from app.core.exceptions import LLMError
from app.core.logger import get_logger

logger = get_logger(__name__)

# Maximum times to retry on transient Groq errors (rate limit / 5xx)
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0   


class LLMService:
    """
    Async wrapper around the Groq LLaMA3 chat completion API.
    Instantiate once and inject via dependency or service layer.
    """

    def __init__(self) -> None:
        self._client = AsyncGroq(api_key=settings.groq_api_key)
        self._model = settings.groq_llm_model
        logger.info(f"LLMService initialised | model={self._model}")

    # â”€â”€ Core private call with retry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def _call(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> str:
        """
        Execute the Groq chat completion with retry on rate-limit / transient errors.

        Returns the raw text content of the first choice.
        """
        temp = temperature if temperature is not None else settings.groq_temperature
        tokens = max_tokens or settings.groq_max_tokens

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        last_exc: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                logger.debug(
                    f"LLM call attempt {attempt}/{_MAX_RETRIES} | "
                    f"messages={len(messages)} | temp={temp}"
                )
                response: ChatCompletion = await self._client.chat.completions.create(
                    **kwargs
                )
                content = response.choices[0].message.content or ""
                logger.debug(
                    f"LLM response received | tokens_used="
                    f"{response.usage.total_tokens if response.usage else 'unknown'}"
                )
                return content

            except RateLimitError as exc:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    f"Groq rate limit hit (attempt {attempt}). "
                    f"Retrying in {delay:.1f}s..."
                )
                last_exc = exc
                await asyncio.sleep(delay)

            except APIStatusError as exc:
                if exc.status_code >= 500:
                    delay = _RETRY_BASE_DELAY * attempt
                    logger.warning(
                        f"Groq 5xx error {exc.status_code} (attempt {attempt}). "
                        f"Retrying in {delay:.1f}s..."
                    )
                    last_exc = exc
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Groq API error {exc.status_code}: {exc.message}")
                    raise LLMError(message=str(exc)) from exc

            except Exception as exc:
                logger.error(f"Unexpected LLM error: {exc}")
                raise LLMError(message=str(exc)) from exc

        raise LLMError(
            message=f"LLM call failed after {_MAX_RETRIES} attempts: {last_exc}"
        )

    # â”€â”€ Public interface â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def complete(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> str:
        """
        Single-turn completion: one system prompt + one user message.

        Args:
            system:          The system instructions.
            user:            The user message / patient input.
            temperature:     Override default temperature (0.0 = deterministic).
            max_tokens:      Override default max output tokens.
            response_format: e.g. {"type": "json_object"} for structured output.

        Returns:
            The assistant's reply as a plain string.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return await self._call(messages, temperature, max_tokens, response_format)

    async def complete_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.1,
    ) -> dict:
        """
        Convenience wrapper for JSON-mode completions (e.g. triage scoring).
        Parses the response and returns a dict.

        Raises LLMError if the response is not valid JSON.
        """
        raw = await self.complete(
            system=system,
            user=user,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error(f"LLM returned invalid JSON: {raw[:200]!r}")
            raise LLMError(message=f"LLM did not return valid JSON: {exc}") from exc

    async def chat(
        self,
        system: str,
        history: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Multi-turn conversation completion.

        Args:
            system:   The persistent system prompt for the conversation.
            history:  Full list of {"role": "user"|"assistant", "content": "..."} dicts.
                      The most recent message should be the last item.

        Returns:
            The assistant's next reply as a plain string.
        """
        messages = [{"role": "system", "content": system}] + history
        return await self._call(messages, temperature, max_tokens)

    async def chat_with_context(
        self,
        system: str,
        history: list[dict],
        context_chunks: list[str],
        temperature: float | None = None,
    ) -> str:
        """
        Multi-turn chat augmented with RAG context chunks.
        Injects retrieved knowledge base excerpts into the system prompt.

        Args:
            context_chunks: List of relevant text chunks from ChromaDB.
        """
        if context_chunks:
            context_block = "\n\n---\n".join(context_chunks)
            augmented_system = (
                f"{system}\n\n"
                f"### Relevant clinical reference (from knowledge base):\n"
                f"{context_block}\n\n"
                f"Use the above reference to support your triage reasoning, "
                f"but do not quote it verbatim to the patient."
            )
        else:
            augmented_system = system

        return await self.chat(augmented_system, history, temperature)
