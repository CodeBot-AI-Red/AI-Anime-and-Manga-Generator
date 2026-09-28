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

## Pré-processamento para o treinamento futuro

Após a validação, use `DatasetPreprocessor` para preparar os exemplos. A configuração padrão é pequena (`32x32` e lote de `2`) para testes locais e pode ser ajustada sem alterar o dataset:

```python
from src.data import DatasetPreprocessor, PreprocessingConfig

preprocessor = DatasetPreprocessor(
    PreprocessingConfig(resolution=(64, 64), batch_size=4)
)
prepared = preprocessor.process(split.train[0])
batch = preprocessor.process_batch(split.train[:4])
```

O pré-processador atual decodifica PNGs RGB/RGBA de 8 bits não entrelaçados, redimensiona por vizinho mais próximo e normaliza cada componente de pixel de `0..255` para `0.0..1.0`. A imagem individual resulta em um `Tensor` no formato `(C, H, W)`; um lote resulta em `(N, C, H, W)`. Os metadados `ImageMetadata` são preservados, alinhados à imagem processada e não são modificados.

Arquivos corrompidos, PNGs com uma codificação não compatível, amostras sem imagem ou metadados, dimensões divergentes e lotes vazios ou acima de `batch_size` geram `PreprocessingError`. Outros formatos ainda podem ser validados na etapa anterior, mas somente PNG é decodificado neste estágio inicial sem dependências externas.

## Fluxo atual e próximos passos

```text
Dataset local
    → Validação (DatasetValidator)
    → Pré-processamento (DatasetPreprocessor)
    → Tensor normalizado (C, H, W) ou (N, C, H, W)
    → Futuro treinamento
```

O projeto continua sem treinar modelos, baixar dados ou usar APIs externas. Quando o sistema de treinamento existir, ele deverá consumir somente `DatasetSample` validados e os tensores preparados por esta etapa.
