"""Leitura de anotações JSON e legendas TXT associadas a imagens."""

from __future__ import annotations

import json
from pathlib import Path

from .schema import ImageMetadata, MetadataValidationError


class MetadataReader:
    """Lê ``.json`` e ``.txt`` com o mesmo caminho relativo da imagem."""

    def read(self, image_path: Path, image_root: Path, annotation_root: Path) -> ImageMetadata:
        relative_stem = image_path.relative_to(image_root).with_suffix("")
        json_path = annotation_root / relative_stem.with_suffix(".json")
        text_path = annotation_root / relative_stem.with_suffix(".txt")
        data: dict[str, object] = {}

        if json_path.exists():
            try:
                parsed = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise MetadataValidationError(f"JSON inválido em {json_path}: {error}") from error
            if not isinstance(parsed, dict):
                raise MetadataValidationError(f"O JSON em {json_path} deve ser um objeto.")
            data.update(parsed)

        if text_path.exists():
            try:
                caption = text_path.read_text(encoding="utf-8").strip()
            except OSError as error:
                raise MetadataValidationError(f"Não foi possível ler legenda: {text_path}") from error
            if not caption:
                raise MetadataValidationError(f"Legenda vazia: {text_path}")
            data.setdefault("caption", caption)

        if not data:
            raise MetadataValidationError(f"Metadados ausentes para a imagem: {image_path}")
        return ImageMetadata.from_mapping(data)
