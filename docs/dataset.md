# Dataset local: primeira versão

Este projeto ainda **não treina modelos**, não baixa dados e não usa APIs externas. Os componentes em `src/data/` apenas descobrem imagens que você colocou localmente, leem as respectivas anotações e separam exemplos para treinamento e validação futuros.

## Estrutura de diretórios

Organize um dataset local com duas árvores que tenham os mesmos caminhos relativos:

```text
meu-dataset/
├── images/
│   └── chapter-01/
│       └── page-001.png
└── annotations/
    └── chapter-01/
        ├── page-001.json
        └── page-001.txt          # opcional
```

As extensões aceitas são `.png`, `.jpg`, `.jpeg`, `.gif` e `.webp`. O carregador verifica a assinatura do arquivo; PNG e GIF também têm largura e altura identificadas sem instalar bibliotecas de imagem. Não adicione imagens de terceiros sem confirmar que você tem os direitos e as permissões necessários.

## Metadados e legendas

Para uma imagem como `images/chapter-01/page-001.png`, a anotação JSON correspondente é `annotations/chapter-01/page-001.json`. O arquivo deve conter um objeto JSON e pode usar estes campos de texto:

```json
{
  "character": "Aiko, estudante de cabelo azul",
  "pose": "correndo",
  "expression": "determinada",
  "camera": "plano médio em ângulo baixo",
  "setting": "rua urbana à noite",
  "lighting": "letreiros neon e chuva refletida",
  "style": "anime com cores vibrantes",
  "scene_description": "Aiko corre para alcançar o trem."
}
```

Os campos representam, respectivamente: personagem, pose, expressão, câmera, cenário, iluminação, estilo e descrição da cena. Eles são opcionais individualmente, mas o JSON precisa ter ao menos um campo preenchido. Campos desconhecidos, valores que não sejam texto ou textos vazios são inválidos para evitar anotações ambíguas.

Também é possível incluir uma legenda livre em `annotations/chapter-01/page-001.txt`. Ela é carregada como `caption`; se houver JSON e TXT, os dois são combinados. Para cada imagem deve existir pelo menos um dos dois arquivos.

## Validação e divisão

Use a API local para validar antes de qualquer etapa futura de treinamento:

```python
from pathlib import Path
from src.data import DatasetValidator, split_dataset

samples, errors = DatasetValidator().validate(
    Path("meu-dataset/images"),
    Path("meu-dataset/annotations"),
)
if errors:
    raise ValueError("\\n".join(errors))

split = split_dataset(samples, validation_fraction=0.1, seed="dataset-v1")
print(len(split.train), len(split.validation))
```

`DatasetValidator` continua examinando os demais arquivos e devolve uma lista de erros para imagens inválidas, imagens sem metadados e anotações inválidas. `split_dataset` usa uma ordenação por hash estável e uma semente explícita, portanto a mesma coleção de arquivos e a mesma semente produzem a mesma divisão. A fração de validação deve estar entre `0` (inclusive) e `1` (exclusivo).

## Próximos passos

Quando o pipeline de treinamento existir, ele deverá consumir somente `DatasetSample` validados e a divisão produzida por `split_dataset`. Esta camada não redimensiona, transforma, baixa, nem treina imagens: essas decisões ficam deliberadamente para módulos futuros.
