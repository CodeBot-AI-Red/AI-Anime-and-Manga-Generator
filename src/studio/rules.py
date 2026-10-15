"""Regras de estúdio extraídas dos guias "Criar" e "Atualizar Imagens de Anime".

Aplicadas automaticamente pelo gerador:
- ``build_prompt`` acrescenta os 7 pilares do look profissional ao prompt;
- ``NEGATIVE_PROMPT`` alimenta o ramo negativo da guidance;
- ``PALETTES`` são as fichas de cor (settei) com valores reais de produção;
- ``CORRECTIONS`` liga cada problema detectado pela visão ao prompt de
  correção do guia (Parte 5 — Problema → Solução).
"""

from __future__ import annotations

from dataclasses import dataclass

LINEART_RGB = (61, 43, 40)
"""Cor padrão de linha em produção real — nunca preto puro #000000."""
DARK_LINEART_RGB = (40, 30, 25)
SKIN_SHADOW_HEX = "#C8A0B0"

STUDIO_PILLARS: tuple[str, ...] = (
    "layered anime eyes: pure white sclera, BL shadow inside iris, iris gradient dark at top light at bottom, "
    "centered pupil with depth, large primary catchlight plus small secondary catchlight contained within iris",
    "variable-weight lineart, thick outer silhouette, thin interior lines, tapered stroke ends, "
    "lineart color RGB 61 43 40 not pure black",
    "cel shading with crisp shadow edge, no gradient blending, base tone plus one hard shadow, white highlight",
    "hair in 3-5 defined chunk groups, curved white highlight streak on the top curve of each chunk, roots in shadow color",
    "consistent character color settei palette, saturated vivid colors",
    "single clear light source, warm key light with cool fill, colored rim light, rule of thirds composition",
    "official anime studio production art, clean flat color surfaces, painterly anime background with three depth planes",
)

NEGATIVE_PROMPT = (
    "pure black lineart, uniform line weight, double lines, scratchy sketch lines, gradient soft shading, airbrushed skin, "
    "skin texture, pores, grain, photorealistic, 3d render, photo, grey sclera, catchlight outside iris, blurry eyes, "
    "individual messy hair strands, highlight on hair roots, rounded fabric folds, desaturated washed out colors, "
    "perfectly centered symmetrical composition, wrong perspective, merged background elements, small eyes big nose, "
    "fused fingers, extra fingers, missing fingers, jpeg artifacts, watermark, text, signature"
)


@dataclass(frozen=True)
class Palette:
    name: str
    reference: str
    swatches: dict[str, str]
    prompt: str


PALETTES: dict[str, Palette] = {
    "fantasia": Palette("Fantasia histórica / oriental", "Akatsuki no Yona", {
        "pele": "#F5CBA7", "pele_sombra_1": "#D4A5A0", "pele_sombra_2": "#B07060", "esclerotica": "#FFFFFF",
        "iris_feminina": "#8870C0", "iris_masculina": "#4060A0", "BL": "#1A1A1A", "Hi": "#FFFFFF",
        "cabelo": "#C03040", "cabelo_sombra": "#6A1020", "cabelo_Hi": "#FF8090", "linha": "#3D2B28",
        "roupa_clara": "#F0D0D0", "roupa_escura": "#1E2D6E",
    }, "historical fantasy anime color palette, warm rosewood tones, traditional East Asian costume palette, flat cel shading, "
       "deep crimson hair with pink highlight streak, violet iris with white catchlight, Akatsuki no Yona art style"),
    "scifi": Palette("Ação / sci-fi", "Aquarion EVOL", {
        "pele": "#FFCC80", "pele_sombra_1": "#E08050", "esclerotica": "#F0F4FF", "iris": "#20A0FF",
        "cabelo": "#1A1A2E", "cabelo_sombra": "#0D0D1A", "cabelo_Hi": "#C0D8FF", "linha": "#3D2B28",
        "roupa": "#006064", "acento": "#E91E8C", "energia": "#00E5FF",
    }, "sci-fi mecha anime color palette, dark navy and magenta accents, electric blue energy effects, cel shading, "
       "dramatic lighting, Aquarion EVOL visual style, saturated complementary colors"),
    "shonen": Palette("Esportivo / shonen", "Diamond no Ace", {
        "pele": "#F0C080", "pele_sombra_1": "#D08050", "pele_sombra_2": "#A06030", "esclerotica": "#FFFFFF",
        "iris": "#804020", "cabelo": "#3A2010", "cabelo_Hi": "#C08040", "uniforme": "#FFFFFF",
        "uniforme_sombra": "#B0B8C8", "acento": "#1A3080", "acento_dourado": "#C08000",
    }, "sports anime color palette, warm natural skin tones, athletic uniform design with navy and white, "
       "realistic but stylized cel shading, Diamond no Ace visual quality, expressive face with determined look"),
    "idol": Palette("Idol / mahou shoujo", "AKB0048", {
        "pele": "#FFEECC", "pele_sombra_1": "#F0C0B0", "esclerotica": "#FFFFFF", "figurino_rosa": "#FF69B4",
        "figurino_dourado": "#FFD700", "iris_rosa": "#FF6080", "iris_roxa": "#8040C0", "iris_verde": "#208040", "iris_azul": "#2060C0",
    }, "idol anime color palette, pastel and vivid pink tones, sparkle effects, layered eye catchlight structure, "
       "heart-shaped inner eye reflection, AKB0048 visual style, soft cel shading with vibrant saturated colors"),
    "dark": Palette("Dark / seinen / guerreiro", "Berserk, Vinland Saga", {
        "pele": "#D7C8B8", "pele_sombra_1": "#A09888", "esclerotica": "#F8F0E0", "linha": "#281E19",
    }, "dark fantasy anime color palette, muted earth tones, gritty warrior aesthetic, minimal hair highlight, "
       "battle-worn clothing details, high contrast shadows, Berserk or Vinland Saga visual grit, desaturated palette"),
}

CORRECTIONS: dict[str, str] = {
    "lineart_preto": "change all lineart color to dark brown RGB(61,43,40), not pure black #000000",
    "sombra_gradiente": "replace all gradient shading with hard-edge cel shading, shadow edges must be crisp with zero blur",
    "textura": "remove all skin texture, apply flat anime cel surface, single hard-edge shadow in cool lavender-pink, no pores, no photorealistic detail",
    "cores_apagadas": "boost saturation 20-25%, apply warm-cool light contrast, add colored rim light on character silhouette edges",
    "esclerotica_cinza": "rebuild eyes with proper anime layer structure: BL shadow base, gradient iris, oval catchlight inside iris, thick upper eyelash, pure white sclera",
    "composicao_centralizada": "move the subject off-center following the rule of thirds, character facing 3/4 angle looking into frame space",
    "cores_excessivas": "reduce to a flat cel palette: base tone, one shadow and one highlight per surface, consistent character settei colors",
    "contraste_baixo": "apply a single clear light source with warm key light and cool fill, defined shadow shapes",
    "inconsistencia_referencia": "match exactly the reference character colors: hair, skin, outfit and eyes must keep the same hex values across scenes",
}


FORBIDDEN_DOMAIN_KEYWORDS: tuple[str, ...] = (
    "photorealistic",
    "real photo",
    "photography",
    "3d render",
    "3d cgi",
    "hyperrealistic",
    "realistic photo",
    "stock photo",
    "real human",
    "live action",
    "realistic portrait",
    "unreal engine",
    "octane render",
    "corporate vector",
)

ANIME_MANGA_DOMAIN_DIRECTIVE = "pure japanese anime and manga art style, official studio production illustration"


def validate_manga_anime_domain(prompt: str) -> tuple[bool, str]:
    """Verifica se o prompt está estritamente restrito ao domínio de anime e mangá."""
    lowered = prompt.lower()
    violations = [keyword for keyword in FORBIDDEN_DOMAIN_KEYWORDS if keyword in lowered]
    if violations:
        return False, f"O prompt contém termos fora do domínio de anime/mangá: {', '.join(violations)}"
    return True, "Prompt válido dentro do domínio estrito de anime e mangá."


def enforce_manga_anime_style(prompt: str) -> str:
    """Garante a conversão e purificação de um prompt para o estilo exclusivo de anime e mangá."""
    cleaned = prompt
    for keyword in FORBIDDEN_DOMAIN_KEYWORDS:
        import re
        cleaned = re.sub(re.escape(keyword), "", cleaned, flags=re.IGNORECASE)

    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    cleaned_prompt = ", ".join(parts)
    if not cleaned_prompt:
        cleaned_prompt = "anime character"

    if "anime" not in cleaned_prompt.lower() and "manga" not in cleaned_prompt.lower():
        cleaned_prompt = f"{cleaned_prompt}, {ANIME_MANGA_DOMAIN_DIRECTIVE}"

    return cleaned_prompt


PALETTES.update({
    "manga_shonen": Palette("Mangá Shonen", "Shonen Jump (Dragon Ball, Naruto)", {
        "linha": "#1A1A1A", "fundo": "#FFFFFF", "sombra_1": "#505050", "sombra_2": "#202020",
    }, "shonen action manga style, dynamic ink lineart, high impact speed lines, bold black and white shading with screentone details"),
    "manga_shojo": Palette("Mangá Shojo", "Shojo Beat (Sailor Moon, Fruits Basket)", {
        "linha": "#2A2028", "fundo": "#FFFFFF", "sombra_1": "#8A7888", "brilho": "#FFFFFF",
    }, "shojo romance manga style, delicate fine lineart, sparkling eyes, flower frame accents, soft halftone screentone shading"),
    "manga_seinen": Palette("Mangá Seinen", "Young Animal (Berserk, Vagabond)", {
        "linha": "#0F0D0E", "fundo": "#F8F6F0", "sombra_1": "#3A3438", "sombra_2": "#1A1618",
    }, "seinen dark fantasy manga style, detailed cross-hatching lineart, heavy dramatic shadows, dark ink texture, intense narrative atmosphere"),
    "manga_monochrome": Palette("Mangá Monocromático", "Página de Mangá Oficial", {
        "linha": "#000000", "fundo": "#FFFFFF", "reticula": "#808080",
    }, "monochrome manga page style, pure black ink lineart, white background, halftone screentone patterns, professional manga panel layout"),
    "anime_modern": Palette("Anime Moderno", "ufotable / MAPPA / Makoto Shinkai", {
        "pele": "#FFE5D9", "pele_sombra_1": "#D0A0B0", "esclerotica": "#FFFFFF", "iris": "#2080FF",
        "cabelo": "#8020E0", "cabelo_Hi": "#E0A0FF", "linha": "#3D2B28",
    }, "modern digital anime art style, ultra clean lineart, luminous vibrant coloring, subtle volumetric atmosphere, crisp cel shading"),
    "anime_retro_90s": Palette("Anime Retro Anos 90", "Evangelion / Cowboy Bebop", {
        "pele": "#F8C8A0", "pele_sombra_1": "#C08070", "esclerotica": "#F0E8E0", "iris": "#306090",
        "cabelo": "#A02020", "cabelo_Hi": "#F08070", "linha": "#402820",
    }, "90s retro anime aesthetic, cel painted style, soft analog bloom, detailed hand-drawn lineart, iconic TV anime look"),
})


def palette_prompt(name: str | None) -> str:
    if not name:
        return ""
    if name not in PALETTES:
        raise ValueError(f"Paleta desconhecida '{name}'. Opções: {', '.join(PALETTES)}")
    return PALETTES[name].prompt


def build_prompt(prompt: str, palette: str | None = None, apply_rules: bool = True, enforce_domain: bool = True) -> str:
    """Prompt final = pedido do usuário + paleta + os 7 pilares do estúdio."""
    user_prompt = enforce_manga_anime_style(prompt) if enforce_domain else prompt.strip()
    parts = [user_prompt, palette_prompt(palette)]
    if apply_rules:
        parts.extend(STUDIO_PILLARS)
    return ", ".join(part for part in parts if part)


def build_negative_prompt(extra: str | None = None, apply_rules: bool = True) -> str | None:
    parts = ([NEGATIVE_PROMPT] if apply_rules else []) + ([extra.strip()] if extra else [])
    return ", ".join(parts) or None
