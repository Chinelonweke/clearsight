from __future__ import annotations
"""
app/services/llm_service.py
─────────────────────────────────────────────────────────────────
Multi-provider LLM service with automatic fallback.

Provider priority (all open-source models, all free tiers):
  1. Groq          — LLaMA-3.3-70B  (fastest, 12k tokens/min free)
  2. OpenRouter    — LLaMA/Mistral/Qwen/Gemma (4 free models, openrouter.ai)
  3. Together AI   — LLaMA-3.3-70B  (free $1 credit, api.together.xyz)
  4. Hugging Face  — Mistral-7B     (free inference, huggingface.co)

If a provider fails (rate limit, error, unavailable), the next
provider in the chain is tried automatically — zero downtime.

Configuration (add to .env):
  GROQ_API_KEY=...              # console.groq.com (free)
  OPENROUTER_API_KEY=...        # openrouter.ai (free, no credit card)
  TOGETHER_API_KEY=...          # api.together.xyz (free $1 credit)
  HUGGINGFACE_API_KEY=...       # huggingface.co/settings/tokens (free)
"""

import asyncio
import json
import os
from abc import ABC, abstractmethod
from typing import Any

import httpx
from groq import AsyncGroq, RateLimitError, APIStatusError
from groq.types.chat import ChatCompletion

from app.config import settings
from app.core.exceptions import LLMError
from app.core.logger import get_logger

logger = get_logger(__name__)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0

# ── Semaphore: max 5 concurrent LLM calls across all providers ──────────────
_llm_semaphore = asyncio.Semaphore(5)


# ── Base provider interface ──────────────────────────────────────────────────

class BaseProvider(ABC):
    """Abstract base for all LLM providers."""
    name: str

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        response_format: dict | None = None,
    ) -> str:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...


# ── Provider 1: Groq (Primary) ───────────────────────────────────────────────

class GroqProvider(BaseProvider):
    """
    Groq — fastest provider, free tier: 1000 req/min, 12k tokens/min.
    Models: LLaMA-3.3-70B, LLaMA-3.1-8B, Mixtral-8x7B
    Get key: console.groq.com
    """
    name = "Groq"

    def __init__(self) -> None:
        self._api_key = getattr(settings, "groq_api_key", "") or os.getenv("GROQ_API_KEY", "")
        self._model = getattr(settings, "groq_llm_model", "llama-3.3-70b-versatile")
        self._client = AsyncGroq(api_key=self._api_key) if self._api_key else None

    def is_available(self) -> bool:
        return bool(self._api_key and self._client)

    async def chat(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        response_format: dict | None = None,
    ) -> str:
        if not self._client:
            raise LLMError(message="Groq not configured")

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                logger.debug(f"[Groq] attempt {attempt}/{_MAX_RETRIES}")
                response: ChatCompletion = await self._client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
                logger.debug(
                    f"[Groq] OK | tokens={response.usage.total_tokens if response.usage else '?'}"
                )
                return content

            except RateLimitError as exc:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(f"[Groq] Rate limit (attempt {attempt}). Retry in {delay:.1f}s")
                last_exc = exc
                if attempt == _MAX_RETRIES:
                    raise LLMError(message="Groq rate limit exhausted") from exc
                await asyncio.sleep(delay)

            except APIStatusError as exc:
                if exc.status_code >= 500:
                    delay = _RETRY_BASE_DELAY * attempt
                    logger.warning(f"[Groq] 5xx (attempt {attempt}). Retry in {delay:.1f}s")
                    last_exc = exc
                    if attempt == _MAX_RETRIES:
                        raise LLMError(message=f"Groq 5xx after {_MAX_RETRIES} retries") from exc
                    await asyncio.sleep(delay)
                else:
                    raise LLMError(message=f"Groq API error {exc.status_code}") from exc

            except Exception as exc:
                raise LLMError(message=f"Groq unexpected error: {exc}") from exc

        raise LLMError(message=f"Groq: max retries exceeded. Last: {last_exc}")


# ── Provider 2: OpenRouter (First fallback) ───────────────────────────────────

class OpenRouterProvider(BaseProvider):
    """
    OpenRouter — unified gateway to 200+ models, many permanently free.
    Tries each free model automatically if one is rate limited.
    OpenAI-compatible API format.

    Get free key: openrouter.ai (no credit card needed)
    Free models:  openrouter.ai/models?q=free
    """
    name = "OpenRouter"
    _BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    _FREE_MODELS = [
        "meta-llama/llama-3.3-70b-instruct:free",
        "mistralai/mistral-7b-instruct:free",
        "qwen/qwen-2.5-72b-instruct:free",
        "google/gemma-3-27b-it:free",
    ]

    def __init__(self) -> None:
        self._api_key = os.getenv("OPENROUTER_API_KEY", "")

    def is_available(self) -> bool:
        return bool(self._api_key)

    async def chat(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        response_format: dict | None = None,
    ) -> str:
        if not self._api_key:
            raise LLMError(message="OpenRouter not configured")

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://clearsightclinic.online",
            "X-Title": "ClearSight Eye Clinic",
        }

        last_exc: Exception | None = None

        for model in self._FREE_MODELS:
            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_format:
                payload["response_format"] = response_format

            model_failed = False
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        resp = await client.post(
                            self._BASE_URL,
                            json=payload,
                            headers=headers,
                        )
                        if resp.status_code == 429:
                            delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                            logger.warning(
                                f"[OpenRouter:{model}] Rate limit (attempt {attempt}). "
                                f"Retry in {delay:.1f}s"
                            )
                            if attempt == _MAX_RETRIES:
                                logger.warning(f"[OpenRouter] {model} exhausted, trying next model")
                                last_exc = LLMError(message=f"{model} rate limited")
                                model_failed = True
                                break
                            await asyncio.sleep(delay)
                            continue

                        if resp.status_code == 402:
                            logger.warning(f"[OpenRouter] {model} requires payment, skipping")
                            last_exc = LLMError(message=f"{model} not free")
                            model_failed = True
                            break

                        resp.raise_for_status()
                        data = resp.json()

                        if "error" in data:
                            err_msg = data["error"].get("message", "Unknown error")
                            logger.warning(f"[OpenRouter:{model}] API error: {err_msg}")
                            last_exc = LLMError(message=err_msg)
                            model_failed = True
                            break

                        content = data["choices"][0]["message"]["content"] or ""
                        if content:
                            logger.debug(f"[OpenRouter] OK | model={model}")
                            return content

                except LLMError:
                    raise
                except Exception as exc:
                    logger.warning(f"[OpenRouter:{model}] Error: {exc}")
                    last_exc = exc
                    model_failed = True
                    break

            if model_failed:
                continue

        raise LLMError(message=f"OpenRouter: all free models failed. Last: {last_exc}")


# ── Provider 3: Together AI (Second fallback) ────────────────────────────────

class TogetherProvider(BaseProvider):
    """
    Together AI — free $1 credit on signup, then ~$0.20/1M tokens.
    Open source models: LLaMA, Mistral, Qwen, DeepSeek, etc.
    Get key: api.together.xyz
    """
    name = "Together AI"
    _BASE_URL = "https://api.together.xyz/v1/chat/completions"
    _MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

    def __init__(self) -> None:
        self._api_key = os.getenv("TOGETHER_API_KEY", "")

    def is_available(self) -> bool:
        return bool(self._api_key)

    async def chat(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        response_format: dict | None = None,
    ) -> str:
        if not self._api_key:
            raise LLMError(message="Together AI not configured")

        payload: dict[str, Any] = {
            "model": self._MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        async with httpx.AsyncClient(timeout=30.0) as client:
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    resp = await client.post(
                        self._BASE_URL,
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        },
                    )
                    if resp.status_code == 429:
                        delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                        logger.warning(
                            f"[Together] Rate limit (attempt {attempt}). Retry in {delay:.1f}s"
                        )
                        if attempt == _MAX_RETRIES:
                            raise LLMError(message="Together AI rate limit exhausted")
                        await asyncio.sleep(delay)
                        continue

                    resp.raise_for_status()
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"] or ""
                    logger.debug(f"[Together AI] OK")
                    return content

                except LLMError:
                    raise
                except Exception as exc:
                    raise LLMError(message=f"Together AI error: {exc}") from exc

        raise LLMError(message="Together AI: max retries exceeded")


# ── Provider 4: Hugging Face Inference API (Third fallback) ──────────────────

class HuggingFaceProvider(BaseProvider):
    """
    Hugging Face Inference API — free tier for smaller models.
    Model: Mistral-7B-Instruct (reliable, fast, free).
    Get key: huggingface.co/settings/tokens (Read permission)
    """
    name = "Hugging Face"
    _BASE_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

    def __init__(self) -> None:
        self._api_key = os.getenv("HUGGINGFACE_API_KEY", "")

    def is_available(self) -> bool:
        return bool(self._api_key)

    def _format_messages(self, messages: list[dict]) -> str:
        """Convert OpenAI-style messages to Mistral instruction format."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"<s>[INST] {content} [/INST]")
            elif role == "user":
                parts.append(f"[INST] {content} [/INST]")
            elif role == "assistant":
                parts.append(content)
        return "".join(parts)

    async def chat(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        response_format: dict | None = None,
    ) -> str:
        if not self._api_key:
            raise LLMError(message="HuggingFace not configured")

        prompt = self._format_messages(messages)
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": max(temperature, 0.01),
                "return_full_text": False,
            },
        }

        async with httpx.AsyncClient(timeout=45.0) as client:
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    resp = await client.post(
                        self._BASE_URL,
                        json=payload,
                        headers={"Authorization": f"Bearer {self._api_key}"},
                    )
                    if resp.status_code == 503:
                        logger.warning(f"[HuggingFace] Model loading, waiting 20s...")
                        await asyncio.sleep(20)
                        continue
                    if resp.status_code == 429:
                        delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                        logger.warning(
                            f"[HuggingFace] Rate limit (attempt {attempt}). Retry in {delay:.1f}s"
                        )
                        if attempt == _MAX_RETRIES:
                            raise LLMError(message="HuggingFace rate limit exhausted")
                        await asyncio.sleep(delay)
                        continue

                    resp.raise_for_status()
                    data = resp.json()
                    content = data[0]["generated_text"] if isinstance(data, list) else ""
                    logger.debug(f"[HuggingFace] OK")
                    return content.strip()

                except LLMError:
                    raise
                except Exception as exc:
                    raise LLMError(message=f"HuggingFace error: {exc}") from exc

        raise LLMError(message="HuggingFace: max retries exceeded")


# ── Main LLMService with provider chain ──────────────────────────────────────

class LLMService:
    """
    Multi-provider LLM service with automatic fallback.

    Provider chain: Groq → OpenRouter → Together AI → Hugging Face

    - All providers use open-source models
    - All have free tiers
    - Automatic fallback on rate limit or error
    - Max 5 concurrent calls (semaphore) to avoid overwhelming any provider
    - Zero changes needed in calling code — same public API as before
    """

    def __init__(self) -> None:
        all_providers: list[BaseProvider] = [
            GroqProvider(),
            OpenRouterProvider(),
            TogetherProvider(),
            HuggingFaceProvider(),
        ]
        self._providers = [p for p in all_providers if p.is_available()]
        self._model = getattr(settings, "groq_llm_model", "llama-3.3-70b-versatile")

        provider_names = [p.name for p in self._providers]
        logger.info(f"LLMService initialised | providers={provider_names} | model={self._model}")

        if len(self._providers) == 0:
            logger.error("No LLM providers configured! Set at least GROQ_API_KEY in .env")
        elif len(self._providers) == 1:
            logger.warning(
                "Only 1 LLM provider configured. Add OPENROUTER_API_KEY, "
                "TOGETHER_API_KEY, HUGGINGFACE_API_KEY for resilience."
            )

    async def _call(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> str:
        """
        Try each provider in order. Fall back on failure.
        Respects global concurrency semaphore (max 5 simultaneous calls).
        """
        temp = temperature if temperature is not None else getattr(settings, "groq_temperature", 0.7)
        tokens = max_tokens or getattr(settings, "groq_max_tokens", 1024)

        if not self._providers:
            raise LLMError(message="No LLM providers configured")

        async with _llm_semaphore:
            last_exc: Exception | None = None

            for provider in self._providers:
                try:
                    logger.debug(f"Trying provider: {provider.name}")
                    result = await provider.chat(
                        messages=messages,
                        temperature=temp,
                        max_tokens=tokens,
                        response_format=response_format,
                    )
                    if result:
                        return result
                except LLMError as exc:
                    logger.warning(f"Provider [{provider.name}] failed: {exc}. Trying next...")
                    last_exc = exc
                    continue
                except Exception as exc:
                    logger.warning(f"Provider [{provider.name}] unexpected error: {exc}. Trying next...")
                    last_exc = exc
                    continue

            raise LLMError(
                message=f"All LLM providers failed. Last error: {last_exc}"
            )

    # ── Public interface (identical API to original — no changes elsewhere) ──

    async def complete(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> str:
        """Single-turn: system prompt + one user message."""
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
        """JSON-mode completion for structured outputs (e.g. triage scoring)."""
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
        """Multi-turn conversation completion."""
        messages = [{"role": "system", "content": system}] + history
        return await self._call(messages, temperature, max_tokens)

    async def chat_with_context(
        self,
        system: str,
        history: list[dict],
        context_chunks: list[str],
        temperature: float | None = None,
    ) -> str:
        """Multi-turn chat augmented with RAG context chunks."""
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