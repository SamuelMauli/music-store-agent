"""Sinônimos de domínio por seção, para que busca lexical e densa achem o vocabulário do cliente.

O manual fala "direito de arrependimento"; o cliente fala "quero devolver" ou "pedir estorno".
Cada lista só repete termos que o próprio manual cobre naquela seção.
"""

from __future__ import annotations

from typing import Final

SECTION_KEYWORDS: Final[dict[str, tuple[str, ...]]] = {
    "1": ("loja", "empresa", "quem somos", "fundação", "acessórios", "escopo"),
    "1.2": ("cnpj", "razão social", "endereço", "telefone", "whatsapp", "e-mail", "contato"),
    "2": ("horário", "funcionamento", "abre", "fecha", "aberto", "sábado", "domingo", "feriado"),
    "3": ("pagamento", "pagar", "pix", "cartão", "crédito", "débito", "boleto", "à vista"),
    "3.1": (
        "parcelamento",
        "parcelar",
        "parcela mínima",
        "sem juros",
        "dividir",
        "combinar formas de pagamento",
        "pix mais cartão",
    ),
    "4": ("troca", "devolução", "devolver", "reembolso", "estorno"),
    "4.1": (
        "arrependimento",
        "me arrependi",
        "desistir",
        "devolver",
        "devolução",
        "estorno",
        "reembolso",
        "cancelar compra",
        "compra online",
    ),
    "4.2": (
        "defeito",
        "defeituoso",
        "quebrou",
        "não funciona",
        "problema de fabricação",
        "mau uso",
    ),
    "4.3": ("preferência", "trocar cor", "trocar modelo", "tamanho", "outro modelo"),
    "4.4": ("não troca", "personalizado", "venda final", "liquidação", "boquilha", "setup"),
    "5": ("frete", "entrega", "envio", "enviar"),
    "5.1": ("frete grátis", "taxa de entrega", "motoboy", "região metropolitana", "entrega local"),
    "5.2": (
        "outras cidades",
        "outro estado",
        "correios",
        "pac",
        "sedex",
        "jadlog",
        "transportadora",
        "frete especial",
        "avaria",
        "seguro",
        "cep",
    ),
    "5.3": ("rastreamento", "rastrear", "código de rastreio", "onde está meu pedido"),
    "6": ("promoção", "desconto", "oferta"),
    "6.1": (
        "black friday",
        "aniversário da loja",
        "volta às aulas",
        "queima de estoque",
        "campanha",
    ),
    "6.2": ("cumulativa", "acumular", "pix em promoção", "rain check", "preço promocional"),
    "7": ("whatsapp", "atendimento", "canal", "assistente virtual"),
    "7.1": (
        "tom de voz",
        "tempo de resposta",
        "acessórios",
        "cordas",
        "palhetas",
        "cabos",
        "pedais",
    ),
    "7.2": ("fluxo de atendimento", "saudação", "etapas"),
    "7.3": (
        "fora de estoque",
        "esgotado",
        "descontinuado",
        "promoção expirada",
        "reclamação",
        "prazo de retorno",
    ),
    "8": ("garantia",),
    "8.1": ("garantia legal", "defeito de fabricação", "código de defesa do consumidor"),
    "8.2": ("garantia do fabricante", "certificado de garantia"),
    "8.3": ("não cobre", "desgaste", "mau uso", "queda", "reparo de terceiros"),
    "9": ("privacidade", "lgpd", "dados pessoais", "excluir dados", "consentimento", "marketing"),
    "10": ("disposições finais", "dúvidas", "gerência"),
}
