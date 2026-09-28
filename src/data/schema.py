"""Estruturas e validação de metadados para exemplos do dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


STRUCTURED_FIELDS = (
    "character",
    "pose",
    "expression",
    "camera",
    "setting",
    "lighting",
    "style",
    "scene_description",
)


class MetadataValidationError(ValueError):
    """Indica que uma anotação não segue o formato esperado."""


@dataclass(frozen=True)
class ImageMetadata:
    """Descrição textual estruturada de uma imagem local do dataset.

    Todos os campos são opcionais para permitir que anotações sejam feitas de
    forma incremental. Ao menos um campo, incluindo ``caption``, deve existir.
    """

    character: str | None = None
    pose: str | None = None
    expression: str | None = None
    camera: str | None = None
    setting: str | None = None
    lighting: str | None = None
    style: str | None = None
    scene_description: str | None = None
    caption: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ImageMetadata":
        """Cria metadados validados a partir de um objeto JSON."""
        allowed_fields = set(STRUCTURED_FIELDS) | {"caption"}
        unknown_fields = set(data) - allowed_fields
        if unknown_fields:
            raise MetadataValidationError(
                "Campos de metadados desconhecidos: " + ", ".join(sorted(unknown_fields))
            )

        values: dict[str, str | None] = {}
        for field in allowed_fields:
            value = data.get(field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise MetadataValidationError(f"O campo '{field}' deve ser uma string não vazia.")
            values[field] = value.strip() if isinstance(value, str) else None

        if not any(values.values()):
            raise MetadataValidationError("Metadados precisam conter ao menos um campo preenchido.")
        return cls(**values)
