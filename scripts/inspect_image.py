"""Faz a IA "ver" uma imagem (gerada ou enviada) e listar o que corrigir segundo os guias.

Exemplos:
    python scripts/inspect_image.py minha_arte.png
    python scripts/inspect_image.py cena_02.png --reference ficha_personagem.png --fix cena_02_corrigida.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.studio import StudioStyleCritic, apply_studio_finish, render_terminal_image, save_html_gallery
from src.studio.image_io import load_image, save_png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--reference", type=Path, help="Ficha/cena anterior para checar consistência de cores.")
    parser.add_argument("--fix", type=Path, help="Salva uma versão com o acabamento de estúdio aplicado.")
    parser.add_argument("--view", "--show", action="store_true", dest="view", help="Exibe a imagem no terminal em ANSI TrueColor.")
    parser.add_argument("--cols", type=int, default=60, help="Largura em colunas para exibição no terminal.")
    parser.add_argument("--html", type=Path, help="Salva o relatório em HTML interativo com a imagem.")
    parser.add_argument("--json", action="store_true", help="Imprime o relatório em JSON.")
    args = parser.parse_args()

    critic = StudioStyleCritic()
    reference = load_image(args.reference) if args.reference else None
    image = load_image(args.image)
    report = critic.analyze(image, reference)

    if args.view:
        print("\n--- VISUALIZAÇÃO NO TERMINAL ---")
        print(render_terminal_image(image, max_cols=args.cols))

    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) if args.json else report.summary())

    if args.html:
        html_path = save_html_gallery(image, args.html, report, reference=reference)
        print(f"\nRelatório visual HTML salvo em {html_path}")

    if args.fix:
        fixed = apply_studio_finish(image)
        save_png(fixed, args.fix)
        print(f"\nVersão corrigida salva em {args.fix}")
        print(critic.analyze(fixed, reference).summary())


if __name__ == "__main__":
    main()
