from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List
from functools import lru_cache
import hashlib
from groq import Groq
from rag.data_state import get_data_hash
from rag.text_utils import normalize_text, safe_join_snippets
from rag.semantic_cache import find_similar, store
from rag.logger import get_logger
logger = get_logger()

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"


SYSTEM_PROMPT = """You are a helpful assistant for question-answering over user-uploaded documents.

Rules:
<<<<<<< HEAD
- If the user message is a greeting/thanks/bye (e.g., "hi", "hello", "hey", "hii", "good morning", "good evening", "thanks", "thank you", "bye", "goodbye", "see you"):
  - Respond politely and friendly.
  - Do NOT mention uploaded documents or CONTEXT.
  - End by asking one short follow-up question such as: "How can I help you?" or "Do you have a question?"
=======
If the user message is a greeting (e.g., "hi", "hello", "hey", "good morning", "good evening"):
  - Respond politely and friendly.
  - End with a short follow-up question like "How can I help you?"

- If the user message is a gratitude or closing message (e.g., "thanks", "thank you", "bye", "goodbye", "see you"):
  - Respond politely and friendly.
  - Do NOT ask any follow-up question.
  - End naturally (e.g., "You're welcome!", "Glad I could help!", "Take care!" "I'm happy to assist you what can i further help you" "I'm always here to help you").
>>>>>>> 1fa4a3cc2d7120b31147cf4cd0822279629d4e9b
- Otherwise:
- Use ONLY the provided CONTEXT to answer.
- If the answer is not in the CONTEXT, say you cannot find it in the uploaded documents.
- Be concise, clear, and human-friendly.
- When you make factual claims, ensure they are supported by the CONTEXT.
- Never hallucinate.
- Do NOT mention:
   - tools
   
"""
@dataclass(frozen=True)
class LLMAnswer:
    answer: str
    cache_status: str = "MISS"


def _build_prompt(question: str, context_chunks: Iterable[str]) -> str:
    context = safe_join_snippets(context_chunks, max_chars=9000)
    question = normalize_text(question).lower()
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"


def _get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY")
    return Groq(api_key=api_key)


# -------------------------------
# 🔹 CACHE KEY BUILDER
# -------------------------------
def _hash_key(question: str, context: List[str]) -> str:
    data_hash = get_data_hash()   # 🔥 NEW

    raw = question + "||" + "||".join(context) + "||" + data_hash
    return hashlib.sha256(raw.encode()).hexdigest()


# -------------------------------
# 🔹 CACHE: LLM RESPONSE
# -------------------------------
@lru_cache(maxsize=50)
def _cached_llm(
    cache_key: str,
    question: str,
    context_tuple: tuple,
    model: str,
    temperature: float,
) -> str:

    context_chunks = list(context_tuple)
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
    if resp.choices and resp.choices[0].message:
        text = resp.choices[0].message.content or ""

    return normalize_text(text)

def generate_answer(
    *,
    question: str,
    context_chunks: List[str],
    model: str = DEFAULT_GROQ_MODEL,
    temperature: float = 0.2,
) -> LLMAnswer:

    if not context_chunks:
        return LLMAnswer(
            answer="I cannot find this information in the uploaded documents.",
            cache_status="NO_CONTEXT"
        )

    # Semantic cache
    found, cached_answer = find_similar(question)

    if found:
        logger.info("Cache SEMANTIC HIT")
        return LLMAnswer(answer=cached_answer, cache_status="SEMANTIC_HIT")
    logger.info("Cache MISS → calling LLM")
    
    # LLM call
    key = _hash_key(question, context_chunks)

    

    answer = _cached_llm(
        key,
        question,
        tuple(context_chunks),
        model,
        temperature,
    )

    store(question, answer)

    return LLMAnswer(answer=answer, cache_status="MISS")
    
#new ai based chattitle
def generate_chat_title(question: str, model: str = DEFAULT_GROQ_MODEL) -> str:
    client = _get_client()

    prompt = f"""
Generate a short chat title (max 4 words) based on this question.
Do NOT add quotes.

Question:
{question}
"""

    resp = client.chat.completions.create(
        model=model,
        temperature=0.3,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    title = resp.choices[0].message.content.strip()
    return title  