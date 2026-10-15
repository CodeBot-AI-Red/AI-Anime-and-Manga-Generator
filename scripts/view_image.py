"""Visualiza qualquer imagem no terminal em ANSI TrueColor e analisa seu padrão de estúdio.

Exemplos:
    python scripts/view_image.py gerada.png
    python scripts/view_image.py gerada.png --cols 80
    python scripts/view_image.py gerada.png --reference ficha.png --html relatorio.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.studio import StudioStyleCritic, render_terminal_image, save_html_gallery
from src.studio.image_io import load_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Caminho da imagem a ser visualizada.")
    parser.add_argument("--cols", type=int, default=60, help="Número de colunas do terminal para exibição.")
    parser.add_argument("--reference", type=Path, help="Imagem de referência para consistência de cor.")
    parser.add_argument("--html", type=Path, help="Gera um relatório galeria em HTML interativo.")
    parser.add_argument("--no-critique", action="store_true", help="Oculta o relatório do crítico de estúdio.")
    args = parser.parse_args()

    image = load_image(args.image)
    reference = load_image(args.reference) if args.reference else None

    print(f"\n Visualizando {args.image} ({image.shape[2]}×{image.shape[1]} px)\n")
    print(render_terminal_image(image, max_cols=args.cols))
    print()

    if not args.no_critique:
        critic = StudioStyleCritic()
        report = critic.analyze(image, reference)
        print(report.summary())
        print()

    if args.html:
        report = StudioStyleCritic().analyze(image, reference) if 'report' not in locals() else report
        html_path = save_html_gallery(image, args.html, report, reference=reference)
        print(f"Galeria HTML salva em: {html_path}")


if __name__ == "__main__":
    main()
