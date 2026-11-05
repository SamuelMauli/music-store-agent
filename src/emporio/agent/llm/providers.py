"""Registro de provedores de LLM: classe do LangChain, extra opcional, credencial e endpoint.

Trocar de provedor é configuração (``EMPORIO_LLM_*``), não código. Os SDKs opcionais
(Anthropic, Gemini, Bedrock) são importados só quando o provedor é usado; sem o extra
instalado a mensagem diz o que instalar.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr

from emporio.agent.llm.spec import ModelSpec
from emporio.settings import Settings

Family = Literal["openai", "azure", "anthropic", "google", "bedrock", "fake"]
StructuredMethod = Literal["function_calling", "json_schema", "json_mode"]

_LOCAL_PREFIXES = ("http://localhost", "http://127.0.0.1", "http://0.0.0.0", "http://[::1]")
_DEFAULT_AZURE_API_VERSION = "2024-10-21"
_NO_TEMPERATURE_PREFIXES = ("claude-haiku-5",)


class LLMConfigError(ValueError):
    """Configuração de LLM inválida ou incompleta (detectada no boot)."""


class MissingExtraError(LLMConfigError):
    """O SDK do provedor não está instalado (extra opcional ausente)."""


@dataclass(frozen=True, slots=True)
class ProviderInfo:
    """Metadados de um provedor registrado.

    Attributes:
        name: Nome usado na spec (``anthropic:claude-...``).
        family: Qual adaptador do LangChain monta o chat model.
        module: Módulo Python do adaptador (importado sob demanda).
        extra: Extra opcional do pacote que traz o módulo (``None`` = dependência base).
        key_envs: Variáveis padrão do SDK com a chave, em ordem de preferência.
        base_url: Endpoint padrão (provedores OpenAI-compatíveis).
        requires_key: Se a chave é obrigatória no boot (Bedrock usa role IAM; locais não).
        structured_method: ``method`` do ``with_structured_output`` mais confiável ali.
        hoist_system: Se mensagens de sistema precisam ir todas para o topo.
    """

    name: str
    family: Family
    module: str = "langchain_openai"
    extra: str | None = None
    key_envs: tuple[str, ...] = ()
    base_url: str | None = None
    requires_key: bool = True
    structured_method: StructuredMethod = "function_calling"
    hoist_system: bool = False


def _compat(name: str, env: str, url: str, *, requires_key: bool = True) -> ProviderInfo:
    return ProviderInfo(name, "openai", key_envs=(env,), base_url=url, requires_key=requires_key)


REGISTRY: dict[str, ProviderInfo] = {
    info.name: info
    for info in (
        ProviderInfo("openai", "openai", key_envs=("OPENAI_API_KEY",)),
        ProviderInfo("azure_openai", "azure", key_envs=("AZURE_OPENAI_API_KEY",)),
        ProviderInfo(
            "anthropic",
            "anthropic",
            module="langchain_anthropic",
            extra="anthropic",
            key_envs=("ANTHROPIC_API_KEY",),
            # Claude 5.x recusa tool forçada para saída estruturada (HTTP 400): usa o
            # structured output nativo da API.
            structured_method="json_schema",
            hoist_system=True,
        ),
        ProviderInfo(
            "google",
            "google",
            module="langchain_google_genai",
            extra="google",
            key_envs=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
            structured_method="json_schema",
            hoist_system=True,
        ),
        ProviderInfo(
            "bedrock",
            "bedrock",
            module="langchain_aws",
            extra="bedrock",
            requires_key=False,
            hoist_system=True,
        ),
        _compat("groq", "GROQ_API_KEY", "https://api.groq.com/openai/v1"),
        _compat("ollama", "OLLAMA_API_KEY", "http://localhost:11434/v1", requires_key=False),
        _compat("openrouter", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
        _compat("deepseek", "DEEPSEEK_API_KEY", "https://api.deepseek.com/v1"),
        _compat("mistral", "MISTRAL_API_KEY", "https://api.mistral.ai/v1"),
        _compat("together", "TOGETHER_API_KEY", "https://api.together.xyz/v1"),
        _compat("xai", "XAI_API_KEY", "https://api.x.ai/v1"),
        _compat("vllm", "VLLM_API_KEY", "http://localhost:8000/v1", requires_key=False),
        ProviderInfo("openai_compatible", "openai"),  # chave/endpoint: EMPORIO_LLM_*
        ProviderInfo("fake", "fake", requires_key=False),
    )
}


def info_for(provider: str) -> ProviderInfo:
    """Metadados do provedor.

    Raises:
        LLMConfigError: Se o provedor não estiver registrado.
    """
    try:
        return REGISTRY[provider]
    except KeyError as exc:
        known = ", ".join(sorted(REGISTRY))
        raise LLMConfigError(
            f"provedor de LLM desconhecido: {provider!r} (conhecidos: {known})"
        ) from exc


def _is_default(info: ProviderInfo, settings: Settings) -> bool:
    return info.name in {settings.llm_provider, "openai_compatible"}


def base_url(info: ProviderInfo, settings: Settings) -> str | None:
    """Endpoint efetivo: ``EMPORIO_LLM_BASE_URL`` vale para o provedor padrão se definido.

    Para ``openai_compatible`` vale sempre (inclusive o default, Groq), como antes.
    """
    if info.name == "openai_compatible":
        return settings.llm_base_url
    if info.name == settings.llm_provider and "llm_base_url" in settings.model_fields_set:
        return settings.llm_base_url
    if info.family == "azure":
        return os.environ.get("AZURE_OPENAI_ENDPOINT") or None
    return info.base_url


@dataclass(frozen=True, slots=True)
class Credential:
    """Chave resolvida e de onde veio (o nome da fonte, nunca o valor)."""

    value: SecretStr | None
    source: str | None


def credential(info: ProviderInfo, settings: Settings) -> Credential:
    """Resolve a chave: ``EMPORIO_LLM_API_KEY`` (padrão), ``EMPORIO_LLM_API_KEYS`` e o SDK.

    Args:
        info: Provedor.
        settings: Configuração.

    Returns:
        A chave (``SecretStr``) e o nome da fonte, ou ``Credential(None, None)``.
    """
    if _is_default(info, settings) and settings.llm_api_key is not None:
        return Credential(settings.llm_api_key, "EMPORIO_LLM_API_KEY")
    if (key := settings.llm_api_keys.get(info.name)) is not None:
        return Credential(key, f"EMPORIO_LLM_API_KEYS[{info.name}]")
    for env in info.key_envs:
        if value := os.environ.get(env):
            return Credential(SecretStr(value), env)
    return Credential(None, None)


def _needs_key(info: ProviderInfo, settings: Settings) -> bool:
    if not info.requires_key:
        return False
    url = base_url(info, settings) or ""
    return not url.startswith(_LOCAL_PREFIXES)


def validate(spec: ModelSpec, settings: Settings) -> None:
    """Checagem de boot: provedor conhecido, extra instalado, credencial e endpoint presentes.

    Args:
        spec: Modelo a validar.
        settings: Configuração.

    Raises:
        LLMConfigError: Com a correção necessária (nunca com o valor de segredo algum).
    """
    info = info_for(spec.provider)
    if info.family == "fake":
        raise LLMConfigError("'fake' não entra numa cadeia real: use EMPORIO_LLM_PROVIDER=fake.")
    if importlib.util.find_spec(info.module) is None:
        raise _missing(info)
    if _needs_key(info, settings) and credential(info, settings).value is None:
        envs = " ou ".join(("EMPORIO_LLM_API_KEY", *info.key_envs))
        if not _is_default(info, settings):
            envs = " ou ".join((*info.key_envs, f"EMPORIO_LLM_API_KEYS[{info.name}]"))
        raise LLMConfigError(f"provedor {info.name!r} sem credencial: defina {envs}.")
    if info.family == "azure" and base_url(info, settings) is None:
        raise LLMConfigError(
            "azure_openai sem endpoint: defina AZURE_OPENAI_ENDPOINT ou EMPORIO_LLM_BASE_URL."
        )


def _missing(info: ProviderInfo) -> MissingExtraError:
    return MissingExtraError(
        f"o provedor {info.name!r} exige o extra opcional: instale `emporio[{info.extra}]` "
        f"(uv sync --extra {info.extra})."
    )


def _load(info: ProviderInfo, attr: str) -> Any:
    try:
        module = importlib.import_module(info.module)
    except ImportError as exc:
        raise _missing(info) from exc
    return getattr(module, attr)


@dataclass(frozen=True, slots=True)
class ChatParams:
    """Parâmetros comuns de uma chamada."""

    temperature: float
    max_tokens: int
    timeout: float


Builder = Callable[[ProviderInfo, ModelSpec, Settings, ChatParams], BaseChatModel]


def _key(info: ProviderInfo, settings: Settings, placeholder: str | None = None) -> Any:
    value = credential(info, settings).value
    return value if value is not None else (SecretStr(placeholder) if placeholder else None)


def _openai(info: ProviderInfo, spec: ModelSpec, settings: Settings, p: ChatParams) -> Any:
    cls = _load(info, "ChatOpenAI")
    return cls(
        model=spec.model,
        base_url=base_url(info, settings),
        api_key=_key(info, settings, placeholder="sem-chave"),  # locais ignoram a chave
        temperature=p.temperature,
        max_completion_tokens=p.max_tokens,
        timeout=p.timeout,
        max_retries=0,  # retentativa é do ResilientLLM, com circuit breaker
    )


def _azure(info: ProviderInfo, spec: ModelSpec, settings: Settings, p: ChatParams) -> Any:
    cls = _load(info, "AzureChatOpenAI")
    version = (
        settings.llm_azure_api_version
        or os.environ.get("AZURE_OPENAI_API_VERSION")
        or os.environ.get("OPENAI_API_VERSION")
        or _DEFAULT_AZURE_API_VERSION
    )
    return cls(
        azure_endpoint=base_url(info, settings),
        azure_deployment=spec.model,
        api_version=version,
        api_key=_key(info, settings),
        temperature=p.temperature,
        max_completion_tokens=p.max_tokens,
        timeout=p.timeout,
        max_retries=0,
    )


def _anthropic(info: ProviderInfo, spec: ModelSpec, settings: Settings, p: ChatParams) -> Any:
    cls = _load(info, "ChatAnthropic")
    extra = {"base_url": url} if (url := base_url(info, settings)) else {}

    def make(temperature: float | None) -> Any:
        return cls(
            model=spec.model,
            api_key=_key(info, settings),
            temperature=temperature,
            max_tokens=p.max_tokens,
            timeout=p.timeout,
            max_retries=0,
            **extra,
        )

    # Claude Haiku 5.x recusa qualquer ``temperature`` (HTTP 400 "Extra inputs are not
    # permitted", medido ao vivo) e o adaptador ainda não sabe disso.
    if spec.model.startswith(_NO_TEMPERATURE_PREFIXES):
        return make(None)
    chat = make(p.temperature)
    # Modelos recentes (ex.: claude-sonnet-5-5) recusam temperatura fora do padrão; o
    # próprio adaptador sabe quais: se a checagem dele reprovar, usa a temperatura padrão.
    check = getattr(chat, "_assert_valid_model_configuration", None)
    try:
        if callable(check):
            check({})
    except ValueError:
        return make(None)
    return chat


def _google(info: ProviderInfo, spec: ModelSpec, settings: Settings, p: ChatParams) -> Any:
    cls = _load(info, "ChatGoogleGenerativeAI")
    return cls(
        model=spec.model,
        api_key=_key(info, settings),
        temperature=p.temperature,
        max_tokens=p.max_tokens,
        request_timeout=p.timeout,
        retries=0,
    )


def _bedrock(info: ProviderInfo, spec: ModelSpec, settings: Settings, p: ChatParams) -> Any:
    cls = _load(info, "ChatBedrockConverse")
    region = settings.llm_aws_region or os.environ.get("AWS_REGION")
    extra: dict[str, Any] = {"region_name": region} if region else {}
    if (url := base_url(info, settings)) is not None:
        extra["base_url"] = url
    # Credenciais: cadeia padrão do boto3 (env, perfil, role IAM da instância/tarefa).
    return cls(
        model=spec.model,
        temperature=p.temperature,
        max_tokens=p.max_tokens,
        max_retries=0,
        **extra,
    )


def _fake(info: ProviderInfo, spec: ModelSpec, settings: Settings, p: ChatParams) -> Any:
    raise LLMConfigError("o provedor 'fake' não tem chat model: use FakeLLM.")


BUILDERS: dict[Family, Builder] = {
    "openai": _openai,
    "azure": _azure,
    "anthropic": _anthropic,
    "google": _google,
    "bedrock": _bedrock,
    "fake": _fake,
}


def build_chat(
    spec: ModelSpec, settings: Settings, *, temperature: float, max_tokens: int
) -> BaseChatModel:
    """Monta o chat model nativo do LangChain para a spec.

    Args:
        spec: Provedor e modelo.
        settings: Configuração (credenciais, endpoint, timeout).
        temperature: Temperatura da chamada.
        max_tokens: Teto de tokens de saída.

    Returns:
        Uma instância de ``BaseChatModel`` com retentativa interna desligada.

    Raises:
        LLMConfigError: Provedor desconhecido ou extra ausente.
    """
    info = info_for(spec.provider)
    params = ChatParams(temperature, max_tokens, settings.llm_timeout_seconds)
    chat: BaseChatModel = BUILDERS[info.family](info, spec, settings, params)
    return chat
