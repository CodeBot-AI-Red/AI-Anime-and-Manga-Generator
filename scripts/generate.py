"""Gera imagens de anime seguindo as regras de estúdio, vê o resultado e se autocorrige.

Fluxo: prompt + regras do estúdio → difusão (com imagens de referência opcionais
e/ou imagem inicial do usuário) → acabamento de estúdio → análise visual →
refinamento img2img automático enquanto a pontuação estiver abaixo da meta.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.models.config import ModelConfig
from src.studio import StudioStyleCritic, apply_studio_finish, build_negative_prompt, build_prompt
from src.studio.image_io import load_image, load_images, save_png
from src.studio.rules import PALETTES
from src.training.diffusion_model import DiffusionTrainingModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="guerreira anime em uma cidade ao entardecer")
    parser.add_argument("--negative-prompt", default=None, help="O que evitar, somado ao prompt negativo do estúdio.")
    parser.add_argument("--palette", choices=sorted(PALETTES), help="Ficha de cores (settei) do guia.")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint .pt; a arquitetura é lida dele.")
    parser.add_argument("--preset", choices=["large", "lite"], default="large")
    parser.add_argument("--output", type=Path, default=Path("generated.png"))
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--reference", type=Path, action="append", default=[], help="Imagem que a IA deve ver (repetível).")
    parser.add_argument("--init-image", type=Path, help="Imagem do usuário para corrigir/refazer (img2img).")
    parser.add_argument("--strength", type=float, default=0.6)
    parser.add_argument("--refine-rounds", type=int, default=1, help="Rodadas de autocorreção após ver a imagem.")
    parser.add_argument("--target-score", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-studio-rules", action="store_true", help="Não injeta as regras dos guias no prompt.")
    parser.add_argument("--no-finish", action="store_true", help="Desliga o acabamento de estúdio.")
    args = parser.parse_args()

    if args.checkpoint:
        model = DiffusionTrainingModel.from_checkpoint(args.checkpoint, device=args.device)
    else:
        overrides = {"image_resolution": (args.resolution, args.resolution)} if args.resolution else {}
        config = ModelConfig.large(**overrides) if args.preset == "large" else ModelConfig.lite(**overrides)
        model = DiffusionTrainingModel(config, device=args.device, seed=args.seed)
        print("Aviso: sem --checkpoint o modelo está com pesos aleatórios; treine antes com scripts/train.py.")

    rules = not args.no_studio_rules
    prompt = build_prompt(args.prompt, args.palette, rules)
    negative = build_negative_prompt(args.negative_prompt, rules)
    negatives = [negative] if negative else None
    generator = torch.Generator(device=model.device).manual_seed(args.seed)
    references = load_images(args.reference).unsqueeze(0) if args.reference else None
    init = load_image(args.init_image).unsqueeze(0) if args.init_image else None
    critic = StudioStyleCritic()
    reference_view = references[0, 0] if references is not None else None

    image = model.sample([prompt], args.steps, args.guidance_scale, negative_prompts=negatives,
                         reference_images=references, init_images=init, strength=args.strength, generator=generator)[0]
    if not args.no_finish:
        image = apply_studio_finish(image)
    report = critic.analyze(image, reference_view)

    for round_index in range(args.refine_rounds):
        if report.overall_score >= args.target_score or not report.issues:
            break
        print(f"Rodada {round_index + 1}: a IA viu a imagem ({report.overall_score:.0%}) e vai corrigir: "
              + ", ".join(check.label for check in report.issues))
        # A imagem gerada vira referência visual + imagem inicial, com o prompt de correção do guia.
        seen = torch.nn.functional.interpolate(image[None], size=(224, 224), mode="bilinear", align_corners=False)[None]
        refs = seen if references is None else torch.cat((references, seen), dim=1)[:, -model.config.max_reference_images:]
        image = model.sample([f"{prompt}, {report.correction_prompt}"], args.steps, args.guidance_scale,
                             negative_prompts=negatives, reference_images=refs,
                             init_images=image.unsqueeze(0), strength=0.35, generator=generator)[0]
        if not args.no_finish:
            image = apply_studio_finish(image)
        report = critic.analyze(image, reference_view)

    save_png(image, args.output)
    args.output.with_suffix(".report.json").write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Geração concluída: {args.output} ({tuple(image.shape)})")
    print(report.summary())


if __name__ == "__main__":
    main()
