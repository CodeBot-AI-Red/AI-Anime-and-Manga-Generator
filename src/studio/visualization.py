"""Sistema de visualização de imagens para terminal (ANSI TrueColor) e galeria HTML.

Permite ao usuário e à IA visualizar imagens de anime/mangá diretamente no terminal
ou gerar um relatório visual HTML interativo com paletas, notas de estúdio e diagnósticos.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import torch
from torch.nn import functional as functional

from .critic import StudioReport
from .image_io import encode_png_bytes, save_png


def render_terminal_image(image: torch.Tensor, max_cols: int = 60) -> str:
    """Renderiza uma imagem ``(3, H, W)`` em ANSI TrueColor de 24 bits usando o caractere half-block '▀'.

    Cada caractere exibe 2 pixels verticais simultaneamente (pixel superior no foreground e inferior no background).
    """
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("A imagem deve ter a forma (3, H, W).")

    image = image.float().clamp(0, 1)
    channels, orig_h, orig_w = image.shape

    cols = max(8, min(max_cols, orig_w))
    # Cada caractere half-block tem proporção ~1:2 (largura:altura) no terminal.
    rows = max(2, int(orig_h * (cols / orig_w) * 0.5))
    if rows % 2 != 0:
        rows += 1

    resized = functional.interpolate(image[None], size=(rows, cols), mode="bilinear", align_corners=False)[0]
    rgb = (resized * 255).byte()

    lines = []
    for y in range(0, rows, 2):
        top_y = y
        bot_y = min(y + 1, rows - 1)
        line_chars = []
        for x in range(cols):
            r_top, g_top, b_top = rgb[0, top_y, x].item(), rgb[1, top_y, x].item(), rgb[2, top_y, x].item()
            r_bot, g_bot, b_bot = rgb[0, bot_y, x].item(), rgb[1, bot_y, x].item(), rgb[2, bot_y, x].item()
            # \033[38;2;R;G;Bm -> Foreground (Top) | \033[48;2;R;G;Bm -> Background (Bottom)
            line_chars.append(f"\033[38;2;{r_top};{g_top};{b_top}m\033[48;2;{r_bot};{g_bot};{b_bot}m▀\033[0m")
        lines.append("".join(line_chars))

    return "\n".join(lines)


def _tensor_to_base64_png(image: torch.Tensor) -> str:
    return base64.b64encode(encode_png_bytes(image)).decode("utf-8")


def save_html_gallery(
    image: torch.Tensor,
    output_path: Path,
    report: StudioReport | None = None,
    reference: torch.Tensor | None = None,
    prompt: str = "",
) -> Path:
    """Gera um arquivo HTML interativo com tema dark para visualização completa da imagem e métricas de estúdio."""
    img_b64 = _tensor_to_base64_png(image)
    ref_b64 = _tensor_to_base64_png(reference) if reference is not None else None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    score_pct = f"{report.overall_score:.0%}" if report else "N/A"
    score_val = report.overall_score if report else 0.0
    score_color = "#4caf50" if score_val >= 0.75 else ("#ff9800" if score_val >= 0.5 else "#f44336")

    checks_html = ""
    if report:
        for check in report.checks:
            badge_class = "pass" if check.passed else "fail"
            badge_text = "APROVADO" if check.passed else "CORRIGIR"
            checks_html += f"""
            <div class="check-card {badge_class}">
                <div class="check-header">
                    <span class="check-title">{check.label}</span>
                    <span class="badge {badge_class}">{badge_text} ({check.score:.0%})</span>
                </div>
                <p class="check-detail">{check.detail}</p>
                {f'<p class="check-correction"><strong>Ação:</strong> {check.correction}</p>' if check.correction else ''}
            </div>
            """

    colors_html = ""
    if report and report.dominant_colors:
        colors_html = '<div class="swatches">'
        for hex_color in report.dominant_colors:
            colors_html += f'<div class="swatch" style="background-color: {hex_color}" title="{hex_color}"><span>{hex_color}</span></div>'
        colors_html += "</div>"

    html_content = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visualizador de Estúdio AI Anime & Manga</title>
    <style>
        :root {{
            --bg: #121218;
            --card-bg: #1e1e28;
            --text: #e2e8f0;
            --text-dim: #94a3b8;
            --accent: #e91e63;
            --pass: #22c55e;
            --fail: #ef4444;
            --border: #2d2d3d;
        }}
        body {{
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 2rem;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        header {{
            text-align: center;
            margin-bottom: 2rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 1.5rem;
        }}
        h1 {{
            color: #ff79c6;
            margin: 0 0 0.5rem 0;
            font-size: 2.2rem;
        }}
        .prompt-box {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 2rem;
            font-style: italic;
            color: #8be9fd;
        }}
        .main-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }}
        @media (max-width: 768px) {{
            .main-grid {{ grid-template-columns: 1fr; }}
        }}
        .image-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
        }}
        .image-card img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
            image-rendering: pixelated;
        }}
        .score-box {{
            background: var(--card-bg);
            border: 2px solid {score_color};
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            text-align: center;
        }}
        .score-value {{
            font-size: 3rem;
            font-weight: bold;
            color: {score_color};
        }}
        .check-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-left: 4px solid var(--border);
            border-radius: 6px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.8rem;
        }}
        .check-card.pass {{ border-left-color: var(--pass); }}
        .check-card.fail {{ border-left-color: var(--fail); }}
        .check-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: bold;
        }}
        .badge {{
            font-size: 0.75rem;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-weight: bold;
        }}
        .badge.pass {{ background: rgba(34, 197, 94, 0.2); color: var(--pass); }}
        .badge.fail {{ background: rgba(239, 68, 68, 0.2); color: var(--fail); }}
        .check-detail {{
            font-size: 0.9rem;
            color: var(--text-dim);
            margin: 0.4rem 0 0 0;
        }}
        .check-correction {{
            font-size: 0.85rem;
            color: #ffb86c;
            margin: 0.4rem 0 0 0;
        }}
        .swatches {{
            display: flex;
            gap: 0.5rem;
            justify-content: center;
            margin-top: 1rem;
            flex-wrap: wrap;
        }}
        .swatch {{
            width: 60px;
            height: 40px;
            border-radius: 6px;
            display: flex;
            align-items: flex-end;
            justify-content: center;
            font-size: 0.65rem;
            font-weight: bold;
            color: #fff;
            text-shadow: 0 1px 2px #000;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>✨ Visualizador Estúdio AI Manga & Anime</h1>
            <p>Análise de Visão Neural e Crítico de Estúdio</p>
        </header>

        {f'<div class="prompt-box"><strong>Prompt:</strong> "{prompt}"</div>' if prompt else ''}

        <div class="main-grid">
            <div class="left-col">
                <div class="image-card">
                    <h3>Imagem Gerada</h3>
                    <img src="data:image/png;base64,{img_b64}" alt="Arte Gerada">
                    {colors_html}
                </div>
                {f'''
                <div class="image-card" style="margin-top: 1.5rem;">
                    <h3>Imagem de Referência</h3>
                    <img src="data:image/png;base64,{ref_b64}" alt="Referência">
                </div>
                ''' if ref_b64 else ''}
            </div>

            <div class="right-col">
                <div class="score-box">
                    <div>Pontuação de Qualidade Estúdio</div>
                    <div class="score-value">{score_pct}</div>
                    {f'<small>{report.width} × {report.height} px</small>' if report else ''}
                </div>

                <h3>Checklist de Qualidade do Guia de Estúdio</h3>
                {checks_html}
            </div>
        </div>
    </div>
</body>
</html>
"""

    output_path.write_text(html_content, encoding="utf-8")
    return output_path
