from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List

from groq import Groq

from rag.text_utils import normalize_text, safe_join_snippets


DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"


SYSTEM_PROMPT = """You are a helpful assistant for question-answering over user-uploaded documents.

Rules:
- If the user message is a greeting/thanks/bye (e.g., "hi", "hello", "hey", "hii", "good morning", "good evening", "thanks", "thank you", "bye", "goodbye", "see you"):
  - Respond politely and friendly.
  - Do NOT mention uploaded documents or CONTEXT.
  - End by asking one short follow-up question such as: "How can I help you?" or "Do you have a question?"
- Otherwise:
- Use ONLY the provided CONTEXT to answer.
- If the answer is not in the CONTEXT, say you cannot find it in the uploaded documents.
- Be concise, clear, and human-friendly.
- When you make factual claims, ensure they are supported by the CONTEXT.
"""


@dataclass(frozen=True)
class LLMAnswer:
    answer: str


def _build_prompt(question: str, context_chunks: Iterable[str]) -> str:
    context = safe_join_snippets(context_chunks, max_chars=9000)
    question = normalize_text(question)
    # Chat-model friendly prompt with explicit context section.
    return f"""CONTEXT:
{context if context else "(no context found)"}

QUESTION:
{question}
"""


def _get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Missing GROQ_API_KEY. Set it in your environment, then restart Streamlit.\n"
            "PowerShell:\n"
            "  setx GROQ_API_KEY \"YOUR_KEY_HERE\"\n"
            "Then reopen the terminal / restart the app."
        )
    return Groq(api_key=api_key)


def generate_answer(
    *,
    question: str,
    context_chunks: List[str],
    model: str = DEFAULT_GROQ_MODEL,
    temperature: float = 0.2,
) -> LLMAnswer:
    prompt = _build_prompt(question, context_chunks)
    client = _get_client()

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    text = ""
    if resp.choices and resp.choices[0].message and resp.choices[0].message.content:
        text = resp.choices[0].message.content
    return LLMAnswer(answer=normalize_text(text))

