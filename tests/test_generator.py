import os
from unittest.mock import patch

from app.rag.generator import (
    SYSTEM_PROMPT,
    RESPONSE_INSTRUCTIONS,
    build_context,
    build_user_prompt,
    get_active_llm_mode,
    get_active_llm_label,
)


def test_build_context_empty_no_general_knowledge():
    ctx = build_context([])
    assert "SIN EVIDENCIA RAG" in ctx
    assert "Responde con conocimiento general" not in ctx


def test_build_user_prompt_includes_rejection_rules():
    prompt = build_user_prompt("¿Síntomas de roya?", "evidencia", "hist", "pred")
    assert "PREGUNTA DEL USUARIO" in prompt
    assert "recházala educadamente" in prompt
    assert SYSTEM_PROMPT not in prompt
    assert RESPONSE_INSTRUCTIONS in prompt


def test_get_active_llm_mode_openai_when_key_set():
    with patch("app.rag.generator.OPENAI_API_KEY", "sk-test"), patch(
        "app.rag.generator.OPENAI_MODEL", "gpt-4o-mini"
    ):
        assert get_active_llm_mode().startswith("openai")


def test_get_active_llm_mode_local_without_key():
    with patch("app.rag.generator.OPENAI_API_KEY", ""):
        assert get_active_llm_mode() == "local"


def test_get_active_llm_label():
    with patch("app.rag.generator.OPENAI_API_KEY", "sk-test"), patch(
        "app.rag.generator.OPENAI_MODEL", "gpt-4o-mini"
    ):
        assert "OpenAI" in get_active_llm_label()

    with patch("app.rag.generator.OPENAI_API_KEY", ""):
        assert "RAG extractivo local" in get_active_llm_label()
