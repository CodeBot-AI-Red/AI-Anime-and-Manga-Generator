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

from src.studio import StudioStyleCritic, apply_studio_finish
from src.studio.image_io import load_image, save_png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--reference", type=Path, help="Ficha/cena anterior para checar consistência de cores.")
    parser.add_argument("--fix", type=Path, help="Salva uma versão com o acabamento de estúdio aplicado.")
    parser.add_argument("--json", action="store_true", help="Imprime o relatório em JSON.")
    args = parser.parse_args()

    critic = StudioStyleCritic()
    reference = load_image(args.reference) if args.reference else None
    image = load_image(args.image)
    report = critic.analyze(image, reference)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) if args.json else report.summary())
    if args.fix:
        fixed = apply_studio_finish(image)
        save_png(fixed, args.fix)
        print(f"\nVersão corrigida salva em {args.fix}")
        print(critic.analyze(fixed, reference).summary())


if __name__ == "__main__":
    main()
