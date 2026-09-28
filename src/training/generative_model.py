"""Adaptador de treinamento para o primeiro gerador experimental."""

from __future__ import annotations

from src.models import IterativeImageGenerator, ModelConfig, Tensor


class GenerativeTrainingModel:
    """Conecta dados pré-processados ao gerador sem criar outro dataloader.

    O parâmetro escalar é propositalmente pequeno: valida o contrato do
    ``Trainer`` atual enquanto camadas treináveis maiores não são introduzidas.
    """

    def __init__(self, model_config: ModelConfig | None = None) -> None:
        self.generator = IterativeImageGenerator(model_config)
        self.scale, self.bias = 0.5, 0.0

    def forward(self, inputs: Tensor, metadata: tuple[object, ...]) -> Tensor:
        prompts = [getattr(item, "caption", None) or "imagem de anime" for item in metadata]
        # O encoder usa imagens do dataset; o gerador exercita texto, etapa e
        # decoder, então ambos os caminhos compartilham a mesma representação.
        latents = self.generator.latent_representation.encode(inputs)
        embeddings = self.generator.text_conditioner.encode(prompts)
        update = self.generator.model.forward(latents, embeddings, 0)
        refined = Tensor(latents.shape, tuple(value - delta for value, delta in zip(latents.values, update.values)))
        decoded = self.generator.latent_representation.decode(refined)
        return Tensor(decoded.shape, tuple(self.scale * value + self.bias for value in decoded.values))

    def backward(self, inputs: Tensor, targets: Tensor) -> dict[str, float]:
        predictions = self.forward(inputs, (object(),) * inputs.shape[0])
        errors = [prediction - target for prediction, target in zip(predictions.values, targets.values)]
        coefficient = 2 / len(errors)
        # A saída pré-escala pode ser obtida de p = scale * value + bias.
        source = [(value - self.bias) / self.scale for value in predictions.values]
        return {"scale": coefficient * sum(error * value for error, value in zip(errors, source)), "bias": coefficient * sum(errors)}

    def state_dict(self) -> dict[str, float]:
        return {"scale": self.scale, "bias": self.bias}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.scale, self.bias = float(state["scale"]), float(state["bias"])
