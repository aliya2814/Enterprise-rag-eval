"""Pluggable LLM wrapper. All three backends have a genuinely free path.

- ollama : fully local/offline, no key (recommended default)
- groq   : free API tier, very fast
- gemini : free API tier
"""
from __future__ import annotations
import os
from .config import settings


def complete(prompt: str, system: str | None = None, temperature: float = 0.0,
             model: str | None = None) -> str:
    backend = settings.llm_backend.lower()
    if backend == "ollama":
        return _ollama(prompt, system, temperature, model)
    if backend == "groq":
        return _groq(prompt, system, temperature, model)
    if backend == "gemini":
        return _gemini(prompt, system, temperature, model)
    raise ValueError(f"Unknown LLM_BACKEND: {backend}")


def _ollama(prompt, system, temperature, model=None):
    import ollama
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = ollama.chat(model=model or settings.ollama_model, messages=messages,
                       options={"temperature": temperature})
    return resp["message"]["content"]


def _groq(prompt, system, temperature, model=None):
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=model or settings.groq_model, messages=messages, temperature=temperature)
    return resp.choices[0].message.content


def _gemini(prompt, system, temperature, model=None):
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(model or settings.gemini_model,
                                  system_instruction=system or None)
    resp = model.generate_content(
        prompt, generation_config={"temperature": temperature})
    return resp.text
