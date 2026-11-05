"""Perfil de produção (``EMPORIO_ENV=prod``): recusa subir com configuração insegura."""

from __future__ import annotations

from pydantic import SecretStr

from emporio.settings import Settings


class ProductionConfigError(RuntimeError):
    """Configuração inválida para produção; ``problems`` lista TODAS as faltas."""

    def __init__(self, problems: list[str]) -> None:
        """Monta a mensagem com um problema por linha."""
        self.problems = problems
        super().__init__(
            "Configuração inválida para EMPORIO_ENV=prod:\n" + "\n".join(f"- {p}" for p in problems)
        )


def _filled(secret: SecretStr | None) -> bool:
    return secret is not None and bool(secret.get_secret_value().strip())


def production_problems(settings: Settings) -> list[str]:
    """Faltas da configuração para produção (vazio fora de ``prod`` ou se tudo está certo)."""
    if settings.env != "prod":
        return []
    problems: list[str] = []
    if not _filled(settings.session_secret):
        problems.append("EMPORIO_SESSION_SECRET ausente (sessões precisam valer entre réplicas)")
    if not any(_filled(key) for key in settings.api_keys):
        problems.append("EMPORIO_API_KEYS vazio (a API não pode ficar aberta)")
    if settings.llm_provider == "fake":
        problems.append("EMPORIO_LLM_PROVIDER=fake (use um provedor real)")
    if settings.state_backend != "postgres":
        problems.append("EMPORIO_STATE_BACKEND deve ser postgres (sqlite não é compartilhado)")
    elif not _filled(settings.database_url):
        problems.append("EMPORIO_DATABASE_URL ausente com state_backend=postgres")
    if not _filled(settings.redis_url):
        problems.append("EMPORIO_REDIS_URL ausente (limites de taxa precisam valer entre réplicas)")
    return problems


def ensure_production_ready(settings: Settings) -> None:
    """Levanta ``ProductionConfigError`` se ``env=prod`` e algo falta."""
    if problems := production_problems(settings):
        raise ProductionConfigError(problems)
