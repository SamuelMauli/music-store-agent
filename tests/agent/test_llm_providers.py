"""Registro de provedores: spec, credenciais, validação de boot e montagem do chat model."""

from __future__ import annotations

import importlib
from typing import Any, get_args

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrockConverse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import SecretStr

from emporio.agent.llm import providers
from emporio.agent.llm.providers import LLMConfigError, MissingExtraError
from emporio.agent.llm.spec import (
    ModelSpec,
    fallback_specs,
    parse_spec,
    provider_label,
    role_specs,
)
from emporio.settings import LLMProviderName, Settings

SDK_ENVS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
    "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
    "MISTRAL_API_KEY",
    "TOGETHER_API_KEY",
    "XAI_API_KEY",
    "VLLM_API_KEY",
    "OLLAMA_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "OPENAI_API_VERSION",
    "AWS_REGION",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for env in SDK_ENVS:
        monkeypatch.delenv(env, raising=False)


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def _build(spec: str, settings: Settings | None = None) -> Any:
    settings = settings or _settings()
    return providers.build_chat(
        parse_spec(spec, settings.llm_provider), settings, temperature=0.1, max_tokens=321
    )


# --- spec ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("anthropic:claude-sonnet-5-5", ModelSpec("anthropic", "claude-sonnet-5-5")),
        ("openai:gpt-5-mini", ModelSpec("openai", "gpt-5-mini")),
        ("google:gemini-2.5-flash", ModelSpec("google", "gemini-2.5-flash")),
        # Groq: nome com "/" fica inteiro no provedor padrão (retrocompatível com o .env)
        ("openai/gpt-oss-120b", ModelSpec("openai_compatible", "openai/gpt-oss-120b")),
        # Prefixo desconhecido não é provedor: "qwen2.5:7b" é nome de modelo do Ollama
        ("qwen2.5:7b-instruct", ModelSpec("openai_compatible", "qwen2.5:7b-instruct")),
        ("ollama:qwen2.5:7b", ModelSpec("ollama", "qwen2.5:7b")),
        ("groq:openai/gpt-oss-20b", ModelSpec("groq", "openai/gpt-oss-20b")),
        (
            "bedrock:anthropic.claude-3-haiku-20240307-v1:0",
            ModelSpec("bedrock", "anthropic.claude-3-haiku-20240307-v1:0"),
        ),
        ("  openai:gpt-5  ", ModelSpec("openai", "gpt-5")),
    ],
)
def test_parse_spec(raw: str, expected: ModelSpec) -> None:
    assert parse_spec(raw, "openai_compatible") == expected


def test_parse_spec_without_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="sem nome de modelo"):
        parse_spec("anthropic:", "openai")


def test_spec_label_round_trips() -> None:
    spec = ModelSpec("groq", "openai/gpt-oss-20b")
    assert parse_spec(spec.label, "openai") == spec


def test_roles_mix_providers_and_chain_keeps_legacy_alias_last() -> None:
    settings = _settings(
        llm_provider="openai",
        llm_model_small="google:gemini-2.5-flash",
        llm_model_medium="anthropic:claude-sonnet-5-5",
        llm_fallbacks=["openai:gpt-5-mini", "gpt-5-mini", "groq:llama-3.3-70b"],
        llm_model_fallback="mistral:mistral-small",
    )
    roles = role_specs(settings)
    assert roles["small"] == ModelSpec("google", "gemini-2.5-flash")
    assert roles["medium"] == ModelSpec("anthropic", "claude-sonnet-5-5")
    assert [s.label for s in fallback_specs(settings)] == [
        "openai:gpt-5-mini",  # duplicata ("gpt-5-mini" no provedor padrão) removida
        "groq:llama-3.3-70b",
        "mistral:mistral-small",
    ]
    assert list(roles) == ["small", "medium", "fallback[0]", "fallback[1]", "fallback[2]"]


def test_fallbacks_are_read_as_json_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMPORIO_LLM_FALLBACKS", '["openai:gpt-5-mini","google:gemini-2.5-flash"]')
    monkeypatch.setenv("EMPORIO_LLM_API_KEYS", '{"anthropic": "segredo-a"}')
    settings = _settings()
    assert settings.llm_fallbacks == ["openai:gpt-5-mini", "google:gemini-2.5-flash"]
    assert "segredo-a" not in repr(settings)


# --- montagem por provedor ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "env", "url"),
    [
        ("openai", "OPENAI_API_KEY", None),
        ("groq", "GROQ_API_KEY", "https://api.groq.com/openai/v1"),
        ("openrouter", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
        ("deepseek", "DEEPSEEK_API_KEY", "https://api.deepseek.com/v1"),
        ("mistral", "MISTRAL_API_KEY", "https://api.mistral.ai/v1"),
        ("together", "TOGETHER_API_KEY", "https://api.together.xyz/v1"),
        ("xai", "XAI_API_KEY", "https://api.x.ai/v1"),
        ("vllm", "VLLM_API_KEY", "http://localhost:8000/v1"),
        ("ollama", "OLLAMA_API_KEY", "http://localhost:11434/v1"),
    ],
)
def test_openai_family_builds_chat_openai(
    monkeypatch: pytest.MonkeyPatch, provider: str, env: str, url: str | None
) -> None:
    monkeypatch.setenv(env, f"chave-{provider}")
    chat = _build(f"{provider}:modelo-x")
    assert type(chat) is ChatOpenAI
    assert chat.model_name == "modelo-x"
    assert chat.openai_api_base == url
    assert chat.openai_api_key.get_secret_value() == f"chave-{provider}"
    assert (chat.max_tokens, chat.max_retries, chat.temperature) == (321, 0, 0.1)
    assert chat.request_timeout == 20.0


def test_local_providers_build_without_key() -> None:
    chat = _build("ollama:qwen2.5:7b")
    assert chat.model_name == "qwen2.5:7b"
    assert chat.openai_api_key.get_secret_value() == "sem-chave"


def test_openai_compatible_keeps_legacy_endpoint_and_key() -> None:
    settings = _settings(llm_api_key="legado", llm_base_url="http://localhost:11434/v1")
    chat = _build("openai/gpt-oss-120b", settings)
    assert type(chat) is ChatOpenAI
    assert chat.model_name == "openai/gpt-oss-120b"
    assert chat.openai_api_base == "http://localhost:11434/v1"
    assert chat.openai_api_key.get_secret_value() == "legado"


def test_azure_builds_with_endpoint_deployment_and_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "chave-az")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://loja.openai.azure.com")
    chat = _build("azure_openai:gpt-5-prod", _settings(llm_azure_api_version="2025-04-01"))
    assert type(chat) is AzureChatOpenAI
    assert chat.azure_endpoint == "https://loja.openai.azure.com"
    assert chat.deployment_name == "gpt-5-prod"
    assert chat.openai_api_version == "2025-04-01"
    assert chat.max_retries == 0


def test_azure_version_falls_back_to_sdk_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "chave-az")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://loja.openai.azure.com")
    monkeypatch.setenv("OPENAI_API_VERSION", "2024-06-01")
    assert _build("azure_openai:dep").openai_api_version == "2024-06-01"


def test_anthropic_builds_chat_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "chave-ant")
    chat = _build("anthropic:claude-sonnet-5-5")
    assert type(chat) is ChatAnthropic
    assert chat.model == "claude-sonnet-5-5"
    assert chat.anthropic_api_key.get_secret_value() == "chave-ant"
    assert (chat.max_tokens, chat.max_retries, chat.default_request_timeout) == (321, 0, 20.0)
    # sonnet-5-5 recusa temperatura fora do padrão: o adaptador manda a padrão
    assert chat.temperature is None
    assert _build("anthropic:claude-haiku-4-5").temperature == 0.1
    # haiku-5-5 recusa qualquer temperature (HTTP 400 medido ao vivo): não é enviada
    assert _build("anthropic:claude-haiku-5-5").temperature is None


def test_anthropic_as_default_provider_honours_base_url_and_key() -> None:
    settings = _settings(
        llm_provider="anthropic", llm_api_key="k-padrao", llm_base_url="https://proxy.interno"
    )
    chat = _build("claude-haiku-5", settings)
    assert chat.anthropic_api_url == "https://proxy.interno"
    assert chat.anthropic_api_key.get_secret_value() == "k-padrao"


@pytest.mark.parametrize("env", ["GOOGLE_API_KEY", "GEMINI_API_KEY"])
def test_google_builds_gemini(monkeypatch: pytest.MonkeyPatch, env: str) -> None:
    monkeypatch.setenv(env, "chave-g")
    chat = _build("google:gemini-2.5-flash")
    assert type(chat) is ChatGoogleGenerativeAI
    assert chat.model.endswith("gemini-2.5-flash")
    assert chat.google_api_key is not None
    assert chat.google_api_key.get_secret_value() == "chave-g"
    assert (chat.max_output_tokens, chat.max_retries, chat.timeout) == (321, 0, 20.0)


def test_bedrock_builds_converse_without_key() -> None:
    chat = _build(
        "bedrock:anthropic.claude-3-haiku-20240307-v1:0", _settings(llm_aws_region="us-east-1")
    )
    assert type(chat) is ChatBedrockConverse
    assert chat.model_id == "anthropic.claude-3-haiku-20240307-v1:0"
    assert chat.region_name == "us-east-1"
    assert (chat.max_tokens, chat.max_retries) == (321, 0)


def test_bedrock_honours_base_url_when_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_REGION", "sa-east-1")
    settings = _settings(llm_provider="bedrock", llm_base_url="https://vpce.bedrock.local")
    chat = _build("meta.llama3-70b-instruct-v1:0", settings)
    assert chat.region_name == "sa-east-1"
    assert chat.endpoint_url == "https://vpce.bedrock.local"


def test_fake_has_no_chat_model() -> None:
    with pytest.raises(LLMConfigError, match="FakeLLM"):
        _build("fake:x")


def test_registry_covers_every_configurable_provider() -> None:
    assert set(providers.REGISTRY) == set(get_args(LLMProviderName))


# --- credenciais e endpoint ---------------------------------------------------------------


def test_base_url_override_only_for_the_default_provider_when_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "g")
    implicit = _settings(llm_provider="groq")
    assert _build("groq:m", implicit).openai_api_base == "https://api.groq.com/openai/v1"
    explicit = _settings(llm_provider="groq", llm_base_url="https://gw.interno/v1")
    assert _build("groq:m", explicit).openai_api_base == "https://gw.interno/v1"
    # o endpoint do provedor padrão não vaza para outro provedor da cadeia
    monkeypatch.setenv("OPENROUTER_API_KEY", "o")
    assert _build("openrouter:m", explicit).openai_api_base == "https://openrouter.ai/api/v1"


def test_credential_precedence_never_exposes_the_value(monkeypatch: pytest.MonkeyPatch) -> None:
    info = providers.REGISTRY["anthropic"]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "do-sdk")
    assert providers.credential(info, _settings()).source == "ANTHROPIC_API_KEY"
    per_provider = _settings(llm_api_keys={"anthropic": SecretStr("por-provedor")})
    assert providers.credential(info, per_provider).source == "EMPORIO_LLM_API_KEYS[anthropic]"
    primary = _settings(llm_provider="anthropic", llm_api_key="principal")
    found = providers.credential(info, primary)
    assert found.source == "EMPORIO_LLM_API_KEY"
    assert "principal" not in repr(found)
    # a chave principal não vale para os outros provedores
    assert providers.credential(providers.REGISTRY["openai"], primary).value is None


def test_validate_requires_key_with_actionable_message() -> None:
    with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY") as caught:
        providers.validate(ModelSpec("anthropic", "claude"), _settings())
    assert "EMPORIO_LLM_API_KEYS[anthropic]" in str(caught.value)
    with pytest.raises(LLMConfigError, match="EMPORIO_LLM_API_KEY"):
        providers.validate(ModelSpec("openai_compatible", "m"), _settings())


@pytest.mark.parametrize("spec", ["bedrock:meta.llama3", "ollama:qwen", "vllm:m"])
def test_validate_does_not_require_key_for_iam_or_local(spec: str) -> None:
    providers.validate(parse_spec(spec, "openai"), _settings())


def test_validate_accepts_local_openai_compatible_without_key() -> None:
    settings = _settings(llm_base_url="http://127.0.0.1:8080/v1")
    providers.validate(ModelSpec("openai_compatible", "m"), settings)


def test_validate_azure_requires_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    with pytest.raises(LLMConfigError, match="AZURE_OPENAI_ENDPOINT"):
        providers.validate(ModelSpec("azure_openai", "dep"), _settings())


def test_validate_rejects_fake_and_unknown_providers() -> None:
    with pytest.raises(LLMConfigError, match="fake"):
        providers.validate(ModelSpec("fake", "x"), _settings())
    with pytest.raises(LLMConfigError, match="desconhecido"):
        providers.validate(ModelSpec("cohere", "x"), _settings())


# --- extra opcional ausente ---------------------------------------------------------------


def test_missing_extra_is_reported_at_boot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    real = importlib.util.find_spec
    monkeypatch.setattr(
        providers.importlib.util,
        "find_spec",
        lambda name, *a: None if name == "langchain_anthropic" else real(name, *a),
    )
    with pytest.raises(MissingExtraError, match=r"instale `emporio\[anthropic\]`"):
        providers.validate(ModelSpec("anthropic", "claude"), _settings())


@pytest.mark.parametrize(
    ("spec", "module", "extra"),
    [
        ("anthropic:claude", "langchain_anthropic", "anthropic"),
        ("google:gemini", "langchain_google_genai", "google"),
        ("bedrock:meta.llama3", "langchain_aws", "bedrock"),
    ],
)
def test_missing_extra_on_import_is_a_clear_error(
    monkeypatch: pytest.MonkeyPatch, spec: str, module: str, extra: str
) -> None:
    real = importlib.import_module

    def fake_import(name: str, *args: Any) -> Any:
        if name == module:
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real(name, *args)

    monkeypatch.setattr(providers.importlib, "import_module", fake_import)
    with pytest.raises(MissingExtraError, match=rf"emporio\[{extra}\]"):
        _build(spec)


@pytest.mark.parametrize(
    ("overrides", "tier", "expected"),
    [
        ({"llm_provider": "fake"}, None, "fake"),
        (
            {"llm_model_small": "google:gemini-3.1-flash-lite", "llm_model_medium": "google:g"},
            None,
            "google",
        ),
        ({"llm_model_small": "groq:a", "llm_model_medium": "anthropic:b"}, None, "groq+anthropic"),
        ({"llm_model_small": "groq:a", "llm_model_medium": "anthropic:b"}, "medium", "anthropic"),
        (
            {"llm_model_small": "openai/gpt-oss-20b", "llm_model_medium": "openai/gpt-oss-120b"},
            "small",
            "openai_compatible(api.groq.com)",
        ),
    ],
)
def test_provider_label_comes_from_spec_not_base_url(
    overrides: dict[str, Any], tier: str | None, expected: str
) -> None:
    """Regressão: o relatório dizia ``api.groq.com`` numa rodada 100% Gemini."""
    assert provider_label(_settings(**overrides), tier) == expected
