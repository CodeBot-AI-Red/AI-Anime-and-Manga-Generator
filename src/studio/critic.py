"""Visão analítica: inspeciona imagens geradas ou enviadas pelo usuário.

Implementa de forma mensurável o checklist "Como identificar que a imagem
parece feita por IA" dos guias. Cada verificação gera pontuação 0–1
(1 = padrão de estúdio), diagnóstico em português e o prompt de correção.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from torch.nn import functional as functional

from .rules import CORRECTIONS, LINEART_RGB


@dataclass
class Check:
    key: str
    label: str
    score: float
    passed: bool
    detail: str
    correction: str | None = None


@dataclass
class StudioReport:
    width: int
    height: int
    overall_score: float
    checks: list[Check] = field(default_factory=list)
    dominant_colors: list[str] = field(default_factory=list)

    @property
    def issues(self) -> list[Check]:
        return [check for check in self.checks if not check.passed]

    @property
    def correction_prompt(self) -> str:
        return ", ".join(check.correction for check in self.issues if check.correction)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["correction_prompt"] = self.correction_prompt
        return data

    def summary(self) -> str:
        lines = [f"Pontuação de estúdio: {self.overall_score:.0%} ({self.width}×{self.height})"]
        for check in self.checks:
            lines.append(f"  [{'OK' if check.passed else '!!'}] {check.label}: {check.score:.0%} — {check.detail}")
        lines.append("  Cores dominantes: " + " ".join(self.dominant_colors))
        if self.issues:
            lines.append("  Prompt de correção: " + self.correction_prompt)
        return "\n".join(lines)


def _luminance(image: torch.Tensor) -> torch.Tensor:
    return 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]


def _saturation(image: torch.Tensor) -> torch.Tensor:
    high, low = image.max(dim=0).values, image.min(dim=0).values
    return (high - low) / high.clamp_min(1e-6)


def _gradient(luma: torch.Tensor) -> torch.Tensor:
    kernel_x = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    grads = functional.conv2d(luma[None, None], torch.stack((kernel_x, kernel_x.T)).unsqueeze(1), padding=1)[0]
    return grads.pow(2).sum(dim=0).sqrt()


class StudioStyleCritic:
    """Analisa uma imagem ``(3, H, W)`` em [0, 1] segundo as regras de estúdio."""

    def __init__(self, analysis_size: int = 256, threshold: float = 0.6) -> None:
        self.analysis_size, self.threshold = analysis_size, threshold

    def _prepare(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("A imagem deve ter a forma (3, altura, largura).")
        image = image.float().clamp(0, 1)
        longest = max(image.shape[1:])
        if longest > self.analysis_size:
            scale = self.analysis_size / longest
            size = (max(8, int(image.shape[1] * scale)), max(8, int(image.shape[2] * scale)))
            image = functional.interpolate(image[None], size=size, mode="area")[0]
        return image

    def _check(self, key: str, label: str, score: float, detail: str) -> Check:
        score = float(max(0.0, min(1.0, score)))
        passed = score >= self.threshold
        return Check(key, label, score, passed, detail, None if passed else CORRECTIONS.get(key))

    def analyze(self, image: torch.Tensor, reference: torch.Tensor | None = None) -> StudioReport:
        original_h, original_w = image.shape[1:]
        image = self._prepare(image)
        luma, sat = _luminance(image), _saturation(image)
        grad = _gradient(luma)
        checks: list[Check] = []

        # LINEART: linhas escuras devem ser marrom RGB(61,43,40), não preto puro.
        dark = luma < 0.22
        if dark.any():
            line_pixels = image[:, dark]
            pure_black = (line_pixels.max(dim=0).values < 0.06).float().mean().item()
            target = torch.tensor(LINEART_RGB, dtype=torch.float32)[:, None] / 255
            warm = ((line_pixels[0] - line_pixels[2]) > 0.02).float().mean().item()
            closeness = 1 - (line_pixels - target).abs().mean().item() * 3
            score = 0.5 * (1 - pure_black) + 0.3 * warm + 0.2 * closeness
            detail = f"{pure_black:.0%} das linhas em preto puro; {warm:.0%} em tom quente"
        else:
            score, detail = 0.7, "nenhuma linha escura detectada"
        checks.append(self._check("lineart_preto", "Cor da lineart", score, detail))

        # SOMBRA: cel shading = transições abruptas; gradiente suave = muitas bordas fracas.
        weak = ((grad > 0.008) & (grad < 0.05)).float().mean().item()
        strong = (grad >= 0.05).float().mean().item()
        soft_ratio = weak / max(weak + strong, 1e-6)
        checks.append(self._check("sombra_gradiente", "Sombras de borda dura", 1 - max(0.0, soft_ratio - 0.35) / 0.55,
                                  f"{soft_ratio:.0%} das transições são suaves (gradiente)"))

        # TEXTURA: ruído de alta frequência em áreas que deveriam ser chapadas.
        blurred = functional.avg_pool2d(luma[None, None], 3, stride=1, padding=1, count_include_pad=False)[0, 0]
        flat = grad < 0.05
        noise = (luma - blurred).abs()[flat].mean().item() if flat.any() else 0.0
        checks.append(self._check("textura", "Superfícies chapadas (sem granulado)", 1 - noise / 0.03,
                                  f"ruído médio em áreas planas = {noise:.4f}"))

        # SATURAÇÃO.
        colored = sat[(luma > 0.15) & (luma < 0.95)]
        mean_sat = colored.mean().item() if colored.numel() else 0.0
        checks.append(self._check("cores_apagadas", "Saturação", mean_sat / 0.45, f"saturação média = {mean_sat:.0%}"))

        # BRANCOS (esclerótica / Hi): branco limpo, não cinza.
        bright = (luma > 0.82) & (sat < 0.12)
        if bright.sum() > 16:
            whiteness = image[:, bright].min(dim=0).values.mean().item()
            score, detail = (whiteness - 0.82) / 0.15, f"brancos médios em {whiteness * 255:.0f}/255"
        else:
            score, detail = 0.7, "poucas áreas brancas para avaliar"
        checks.append(self._check("esclerotica_cinza", "Brancos puros (esclerótica/Hi)", score, detail))

        # COMPOSIÇÃO: evitar sujeito perfeitamente centralizado e simétrico.
        weights = grad / grad.sum().clamp_min(1e-6)
        ys, xs = torch.meshgrid(torch.linspace(0, 1, grad.shape[0]), torch.linspace(0, 1, grad.shape[1]), indexing="ij")
        cx, cy = (weights * xs).sum().item(), (weights * ys).sum().item()
        symmetry = max(0.0, 1 - (luma - luma.flip(-1)).abs().mean().item() * 4)
        offset = min(1.0, (abs(cx - 0.5) + abs(cy - 0.5)) / 0.12)
        checks.append(self._check("composicao_centralizada", "Composição (regra dos terços)",
                                  0.6 * offset + 0.4 * (1 - symmetry),
                                  f"centro visual em ({cx:.2f}, {cy:.2f}); simetria {symmetry:.0%}"))

        # PALETA: cel shading usa poucas cores chapadas.
        quantized = (image * 15).round().to(torch.int64)
        codes = quantized[0] * 256 + quantized[1] * 16 + quantized[2]
        counts = torch.bincount(codes.flatten(), minlength=4096).float()
        coverage = (counts.sort(descending=True).values[:24].sum() / counts.sum()).item()
        checks.append(self._check("cores_excessivas", "Paleta chapada (settei)", (coverage - 0.35) / 0.45,
                                  f"24 cores cobrem {coverage:.0%} da imagem"))

        # CONTRASTE / LUZ.
        spread = (luma.quantile(0.95) - luma.quantile(0.05)).item()
        checks.append(self._check("contraste_baixo", "Contraste e luz definida", spread / 0.6, f"faixa tonal = {spread:.0%}"))

        # CONSISTÊNCIA com uma imagem de referência (ficha do personagem / cena anterior).
        if reference is not None:
            ref = self._prepare(reference)

            def hist(img: torch.Tensor) -> torch.Tensor:
                bins = (img.flatten(1) * 7).round().T @ torch.tensor([64.0, 8.0, 1.0])
                values = torch.bincount(bins.long(), minlength=512).float()
                return values / values.sum()

            similarity = torch.minimum(hist(image), hist(ref)).sum().item()
            checks.append(self._check("inconsistencia_referencia", "Consistência com a referência",
                                      similarity / 0.55, f"interseção de paleta = {similarity:.0%}"))

        dominant = [f"#{int(code) // 256 * 17:02X}{int(code) // 16 % 16 * 17:02X}{int(code) % 16 * 17:02X}"
                    for code in counts.argsort(descending=True)[:6] if counts[code] > 0]
        overall = sum(check.score for check in checks) / len(checks)
        return StudioReport(original_w, original_h, overall, checks, dominant)
