"""One tiny, swappable LLM interface.

    llm = make_llm({"provider": "gemini", "model": "gemini-3.8-flash"})
    text = llm.complete(system, user, task="episode")

Providers: gemini | openai (any OpenAI-compatible endpoint: OpenAI, Groq,
OpenRouter, Together, Ollama, vLLM, an internal gateway) | anthropic | mock.
Every response is cached on disk, keyed by (provider, model, prompt), so reruns
are free and a run can be resumed after a rate-limit failure.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import requests


class LLMError(RuntimeError):
    pass


def extract_json(text: str):
    """Parse JSON from a model reply, tolerating ```json fences and chatter."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = t.find(open_c), t.rfind(close_c)
        if i != -1 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError(f"model did not return valid JSON: {text[:300]!r}")


class BaseLLM:
    name = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = cfg.get("model", "")
        self.temperature = float(cfg.get("temperature", 0.3))
        self.max_tokens = int(cfg.get("max_tokens", 4096))
        self.min_interval = float(cfg.get("min_interval_sec", 0))
        self.retries = int(cfg.get("retries", 4))
        self.cache_dir = Path(cfg.get("cache_dir", ".cache/llm"))
        self.use_cache = bool(cfg.get("cache", True))
        self.log: list[dict] = []
        self._last = 0.0

    # subclasses implement _call(system, user) -> (text, usage_dict)
    def _call(self, system: str, user: str):
        raise NotImplementedError

    def complete(self, system: str, user: str, task: str = "", mock_hint=None) -> str:
        key = hashlib.sha256(json.dumps([self.name, self.model, system, user]).encode()).hexdigest()[:24]
        path = self.cache_dir / f"{key}.json"
        if self.use_cache and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.log.append({"task": task, "cached": True, "usage": data.get("usage", {})})
            return data["text"]
        if self.name == "mock":
            text, usage = self._mock(mock_hint)
        else:
            text, usage = self._with_retries(system, user)
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"task": task, "model": self.model, "text": text, "usage": usage}), encoding="utf-8")
        self.log.append({"task": task, "cached": False, "usage": usage})
        return text

    def _with_retries(self, system, user):
        err = None
        for attempt in range(self.retries + 1):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                return self._call(system, user)
            except LLMError as e:
                err = e
                msg = str(e)
                retryable = any(c in msg for c in ("429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED", "overloaded", "timeout"))
                if not retryable or attempt == self.retries:
                    break
                time.sleep(min(60, 5 * 2 ** attempt))
        raise err

    def _post(self, url, headers, body):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=float(self.cfg.get("timeout_sec", 120)))
        except requests.RequestException as e:
            raise LLMError(f"timeout/network error calling {self.name}: {e}")
        if r.status_code != 200:
            raise LLMError(f"{self.name} HTTP {r.status_code}: {r.text[:400]}")
        return r.json()


class GeminiLLM(BaseLLM):
    name = "gemini"

    def _call(self, system, user):
        key = os.environ.get(self.cfg.get("api_key_env", "GEMINI_API_KEY"), "")
        if not key:
            raise LLMError("set GEMINI_API_KEY (free key: https://aistudio.google.com/apikey)")
        base = self.cfg.get("base_url", "https://generativelanguage.googleapis.com/v1beta")
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": self.temperature, "maxOutputTokens": self.max_tokens,
                                 "responseMimeType": "application/json"},
        }
        data = self._post(f"{base}/models/{self.model}:generateContent",
                          {"x-goog-api-key": key, "Content-Type": "application/json"}, body)
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        except (KeyError, IndexError):
            raise LLMError(f"gemini returned no text: {json.dumps(data)[:400]}")
        u = data.get("usageMetadata", {})
        return text, {"in": u.get("promptTokenCount", 0), "out": u.get("candidatesTokenCount", 0)}


class OpenAICompatLLM(BaseLLM):
    name = "openai"

    def _call(self, system, user):
        key = os.environ.get(self.cfg.get("api_key_env", "OPENAI_API_KEY"), "")
        base = self.cfg.get("base_url", "https://api.openai.com/v1").rstrip("/")
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        body = {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if self.cfg.get("json_mode", True):
            body["response_format"] = {"type": "json_object"}
        data = self._post(f"{base}/chat/completions", headers, body)
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise LLMError(f"openai-compatible endpoint returned no text: {json.dumps(data)[:400]}")
        u = data.get("usage", {})
        return text, {"in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0)}


class AnthropicLLM(BaseLLM):
    name = "anthropic"

    def _call(self, system, user):
        key = os.environ.get(self.cfg.get("api_key_env", "ANTHROPIC_API_KEY"), "")
        if not key:
            raise LLMError("set ANTHROPIC_API_KEY")
        base = self.cfg.get("base_url", "https://api.anthropic.com").rstrip("/")
        body = {"model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
                "system": system, "messages": [{"role": "user", "content": user}]}
        data = self._post(f"{base}/v1/messages",
                          {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}, body)
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        u = data.get("usage", {})
        return text, {"in": u.get("input_tokens", 0), "out": u.get("output_tokens", 0)}


class MockLLM(BaseLLM):
    """Offline stand-in for tests: echoes a hint the pipeline derives from the rules.
    It exercises parsing/validation/ranking plumbing; it is NOT a quality signal."""
    name = "mock"

    def _mock(self, hint):
        return json.dumps(hint if hint is not None else {}), {"in": 0, "out": 0}


PROVIDERS = {"gemini": GeminiLLM, "openai": OpenAICompatLLM, "anthropic": AnthropicLLM, "mock": MockLLM}


def make_llm(cfg: dict) -> BaseLLM:
    p = cfg.get("provider", "gemini")
    if p not in PROVIDERS:
        raise ValueError(f"unknown provider {p!r}; choose from {sorted(PROVIDERS)}")
    return PROVIDERS[p](cfg)
