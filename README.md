# ✨ AI Anime & Manga Generator

> Uma fundação **local**, **treinada do zero** e orientada a experimentação para geração de imagens de anime e mangá por difusão.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-2ea44f)
![Local first](https://img.shields.io/badge/execu%C3%A7%C3%A3o-local%20first-6f42c1)

Este repositório implementa uma U-Net condicional em PyTorch, um encoder de prompt *byte-level*, agendamento de ruído e um pipeline de preparação de dados. Não há download automático de modelos, pesos pré-treinados, dataset, API externa ou telemetria. Você fornece imagens e anotações **sobre as quais tem direitos de uso**, treina localmente e gera um PNG localmente.

> [!IMPORTANT]
> É uma base pequena para pesquisa e aprendizado — não um gerador de qualidade profissional pronto para produção. Os padrões usam imagens de **32×32**, poucos passos e CPU para manter a execução acessível. Qualidade visual exige um dataset bem curado, resolução/capacidade maiores e treinamento suficiente.

## Índice

- [O que existe hoje](#o-que-existe-hoje)
- [Capacidade do modelo](#capacidade-do-modelo)
- [Início rápido](#início-rápido)
- [Dataset e anotações](#dataset-e-anotações)
- [Treinar](#treinar)
- [Gerar uma imagem](#gerar-uma-imagem)
- [Parâmetros](#parâmetros)
- [Arquitetura e fluxo](#arquitetura-e-fluxo)
- [Limitações e escopo](#limitações-e-escopo)
- [Testes e documentação](#testes-e-documentação)

## O que existe hoje

| Área | Implementação atual |
| --- | --- |
| **Dados** | Descoberta e validação de PNG, JPEG, GIF e WebP; metadados JSON/TXT; divisão determinística de treino/validação. |
| **Pré-processamento** | Decodificação local de PNG RGB/RGBA de 8 bits, redimensionamento por vizinho mais próximo e tensores normalizados em `[0, 1]`. |
| **Modelo principal** | U-Net condicional de dois níveis, blocos residuais FiLM, *skip connections*, embedding temporal senoidal e atenção cruzada texto–imagem. |
| **Texto** | Transformer treinável do zero que tokeniza os bytes UTF-8 do prompt; não usa encoder de texto externo. |
| **Treinamento** | Predição de ruído com MSE, AdamW, *classifier-free guidance* e checkpoints `.pt` retomáveis. |
| **Amostragem** | Processo reverso DDIM determinístico, inclusive quando poucos passos saltam timesteps; PNG salvo sem Pillow. |

## Capacidade do modelo

### **542.515 parâmetros treináveis** na configuração padrão

Essa é a capacidade do `DiffusionDenoiser` que os scripts criam por padrão: imagem RGB de 32×32, `base_channels=16`, condicionamento de 64 dimensões, encoder de texto de 2 camadas, 4 cabeças de atenção e 2 blocos residuais por estágio. Todos os parâmetros são treináveis; portanto, o total de parâmetros e o total de parâmetros treináveis são iguais.

| Componente | Parâmetros | Participação |
| --- | ---: | ---: |
| Embedding temporal (`time_embedding`) | 8.320 | 1,5% |
| Encoder de prompt (`prompt_encoder`) | 124.672 | 23,0% |
| Fusão do condicionamento (`condition`) | 12.416 | 2,3% |
| Entrada, reduções e reconstruções da U-Net | 82.512 | 15,2% |
| Blocos residuais da U-Net | 297.360 | 54,8% |
| Atenção cruzada texto–imagem (`cross_attention`) | 16.768 | 3,1% |
| Saída da U-Net (`output`) | 467 | 0,1% |
| **Total — `DiffusionDenoiser`** | **542.515** | **100%** |

> [!NOTE]
> A resolução, o número de passos de geração e o número de timesteps mudam o custo de memória/execução, mas não adicionam pesos a esta arquitetura. Aumentar principalmente `base_channels`, `time_embedding_dim`, `text_encoder_layers` ou `unet_res_blocks` aumenta a quantidade de parâmetros. Uma alteração na configuração também exige recriar a mesma arquitetura ao carregar um checkpoint.

Para medir exatamente uma configuração customizada, execute:

```bash
python - <<'PY'
from src.models.config import ModelConfig
from src.models.diffusion import DiffusionDenoiser, count_trainable_parameters

config = ModelConfig(base_channels=16, image_resolution=(32, 32))
model = DiffusionDenoiser(config)
print(f"{count_trainable_parameters(model):,} parâmetros treináveis")
PY
```

## Início rápido

### 1. Preparar o ambiente

```bash
git clone <URL-DO-SEU-FORK>
cd AI-Anime-and-Manga-Generator

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

O único requisito de runtime declarado é `torch>=2.2`. O código usa anotações de tipo modernas; use Python 3.10 ou superior.

### 2. Organizar pelo menos duas imagens anotadas

```text
data/
├── raw/
│   └── chapter-01/
│       ├── page-001.png
│       └── page-002.png
└── annotations/
    └── chapter-01/
        ├── page-001.json
        ├── page-001.txt       # opcional se houver JSON
        └── page-002.txt
```

### 3. Treinar e gerar

```bash
python scripts/train.py \
  --images data/raw \
  --annotations data/annotations \
  --resolution 32 \
  --epochs 2 \
  --batch-size 2 \
  --checkpoint-dir checkpoints

python scripts/generate.py \
  --checkpoint checkpoints/epoch-2.pt \
  --prompt "guerreira anime em uma cidade ao entardecer" \
  --output generated.png \
  --resolution 32 \
  --steps 8
```

O primeiro comando imprime amostras ignoradas, caso existam, e salva `epoch-<n>.pt` ao fim de cada época. O segundo cria `generated.png` no caminho informado.

## Dataset e anotações

As duas árvores devem ter os **mesmos caminhos relativos**. Para `data/raw/chapter-01/page-001.png`, o leitor procura `data/annotations/chapter-01/page-001.json` e/ou `data/annotations/chapter-01/page-001.txt`.

### Imagens aceitas

- A validação reconhece extensões `.png`, `.jpg`, `.jpeg`, `.gif` e `.webp` e confere a assinatura do arquivo.
- O pré-processador do treinamento decodifica atualmente apenas PNG RGB/RGBA de 8 bits, não entrelaçado. Portanto, para treinar use **PNG**.
- Cada imagem precisa de ao menos uma anotação válida. O script requer no mínimo **duas** imagens válidas para separar treino e validação.

### Exemplo de JSON

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

Todos os campos são opcionais individualmente, mas é obrigatório preencher pelo menos um. Os campos permitidos são `character`, `pose`, `expression`, `camera`, `setting`, `lighting`, `style`, `scene_description` e `caption`. Valores precisam ser strings não vazias; chaves desconhecidas tornam a anotação inválida. Uma legenda livre em `.txt` é lida como `caption`; JSON e TXT são combinados quando ambos existem.

No treino, o prompt vem de `caption`; na ausência dele, vem de `scene_description`; se ambos faltarem, usa-se `imagem de anime`. Prefira uma legenda `.txt` descritiva para condicionar o modelo de forma direta.

## Treinar

O pipeline é inteiramente local:

```text
PNG + JSON/TXT
      │
      ▼
validação → divisão determinística (20% validação) → pré-processamento
      │
      ▼
imagem [0, 1] → imagem [-1, 1] + ruído + timestep
      │
      ▼
U-Net condicional prevê o ruído → MSE → AdamW → checkpoint .pt
```

Por padrão, o treinamento é em `cpu`, usa taxa de aprendizado `0.001` e salva a cada época. Os artefatos de `data/` e `checkpoints/` são ignorados pelo Git para evitar o versionamento acidental de dados e pesos.

## Gerar uma imagem

O gerador começa de ruído gaussiano, aplica o denoiser condicional e incondicional e combina ambos por *classifier-free guidance*. `--steps` controla o número de passos de amostragem, não o número de timesteps do scheduler. Mais passos normalmente aumentam custo e podem melhorar a trajetória de amostragem, mas não compensam um checkpoint pouco treinado.

> [!WARNING]
> `scripts/generate.py` recria a arquitetura padrão e só expõe resolução e passos. Para carregar um checkpoint, mantenha a mesma arquitetura usada no treino — em especial `base_channels`, dimensões, camadas e resolução. Os scripts padrão de treino/geração são compatíveis entre si quando usados com os valores padrão. Checkpoints treinados com uma `ModelConfig` customizada devem ser carregados por código Python que recrie essa mesma configuração.

Exemplo de uso programático com `guidance_scale`:

```python
from pathlib import Path

from src.models.config import ModelConfig
from src.training.diffusion_model import DiffusionTrainingModel

config = ModelConfig(image_resolution=(32, 32), guidance_scale=3.0)
model = DiffusionTrainingModel(config)
model.load_checkpoint(Path("checkpoints/epoch-2.pt"))
image = model.sample(["heroína anime, chuva neon, plano médio"], sampling_steps=8)
```

O tensor retornado tem formato `(lote, 3, altura, largura)`, está na CPU e tem valores em `[0, 1]`.

## Parâmetros

### CLI de treinamento — `scripts/train.py`

| Argumento | Padrão | Descrição |
| --- | ---: | --- |
| `--images` | **obrigatório** | Diretório raiz das imagens locais. |
| `--annotations` | **obrigatório** | Diretório raiz dos `.json` e `.txt` correspondentes. |
| `--resolution` | `32` | Largura e altura quadradas usadas no pré-processamento e na U-Net. Deve ser divisível por 4 na arquitetura atual. |
| `--epochs` | `2` | Número de épocas de treinamento. Deve ser positivo. |
| `--batch-size` | `2` | Número de imagens por lote. Deve ser positivo. |
| `--checkpoint-dir` | `checkpoints` | Diretório dos checkpoints `epoch-<época>.pt`. |

O script fixa `learning_rate=0.001`, `device="cpu"`, `seed=42`, `save_every=1`, `num_timesteps=100` e a arquitetura padrão de `ModelConfig`.

### CLI de geração — `scripts/generate.py`

| Argumento | Padrão | Descrição |
| --- | ---: | --- |
| `--prompt` | `guerreira anime em uma cidade ao entardecer` | Texto que condiciona a geração. |
| `--checkpoint` | nenhum | Caminho opcional para checkpoint `.pt`; sem ele, a rede aleatória é usada. |
| `--output` | `generated.png` | Caminho do PNG RGB de saída. Diretórios pais são criados automaticamente. |
| `--resolution` | `32` | Resolução quadrada da imagem; deve coincidir com o checkpoint. |
| `--steps` | `8` | Passos de amostragem. É limitado internamente a `num_timesteps` (100). |

### `ModelConfig` — arquitetura e difusão

Estes são os parâmetros da configuração Python em `src/models/config.py`. Os valores abaixo são os padrões efetivos da classe.

| Parâmetro | Padrão | Efeito / restrição |
| --- | ---: | --- |
| `image_channels` | `3` | Canais de entrada e saída da imagem. |
| `image_resolution` | `(32, 32)` | Tupla `(largura, altura)`. Cada dimensão deve ser positiva e divisível por `downsample_factor²` (4 no padrão da U-Net). |
| `base_channels` | `16` | Largura inicial da U-Net; os níveis seguintes usam 2× e 4× esse valor. |
| `latent_channels` | `16` | Dimensão usada pelos componentes experimentais de representação latente; não é o espaço latente da U-Net de difusão atual. |
| `downsample_factor` | `2` | Fator de cada redução espacial; a U-Net tem duas reduções. |
| `unet_res_blocks` | `2` | Blocos residuais por estágio da U-Net. |
| `num_timesteps` | `100` | Número de passos do processo de ruído linear. Deve ser ao menos 2 para o scheduler. |
| `generation_steps` | `4` | Passos usados por `sample()` quando `sampling_steps` não é informado. |
| `time_embedding_dim` | `64` | Dimensão do embedding temporal e do condicionamento interno; deve ser divisível por `attention_heads` e ser ao menos 4. |
| `text_vocab_size` | `256` | Tamanho do vocabulário do encoder byte-level. |
| `max_prompt_length` | `128` | Máximo de bytes UTF-8 do prompt considerados. |
| `text_encoder_layers` | `2` | Camadas do Transformer de texto treinado do zero. |
| `attention_heads` | `4` | Cabeças do Transformer e da atenção cruzada; deve dividir as dimensões usadas pela atenção. |
| `text_embedding_dim` | `12` | Dimensão reservada aos componentes leves/experimentais de texto; o denoiser usa `time_embedding_dim`. |
| `scene_embedding_dim` | `12` | Dimensão do condicionador de cena experimental. |
| `condition_dropout` | `0.1` | Probabilidade de remover o texto por amostra durante o treino para ensinar a rota incondicional. Intervalo: `[0, 1)`. |
| `guidance_scale` | `3.0` | Intensidade da guidance na amostragem. `0` desliga a combinação incondicional; valores negativos são inválidos. |

O `NoiseScheduler` usa betas lineares de `0.0001` a `0.02`. Esses limites são argumentos da classe (`beta_start` e `beta_end`), mas não são expostos pela CLI.

### `TrainingConfig` — loop de treino

| Parâmetro | Padrão da classe | Observação |
| --- | ---: | --- |
| `epochs` | `2` | Positivo. |
| `learning_rate` | `0.01` | Positivo. O script sobrescreve para `0.001`. |
| `batch_size` | `2` | Positivo. |
| `checkpoint_dir` | `checkpoints` | Diretório de saída. |
| `save_every` | `1` | Salva em épocas múltiplas deste valor. |
| `device` | `auto` | Nesta implementação, resolve para `cpu`; o script usa `cpu`. |
| `seed` | `42` | Semente do carregador e da inicialização do modelo. |

### Arquivos YAML em `configs/`

Os arquivos `configs/model.yaml`, `configs/training.yaml` e `configs/generation.yaml` servem como **presets de referência** dos hiperparâmetros. Nesta versão, os scripts não fazem leitura automática de YAML; para mudar a execução pela CLI, use os argumentos listados acima, ou instancie `ModelConfig`/`TrainingConfig` no seu código. Alguns valores desses arquivos não correspondem exatamente aos padrões da classe, portanto trate-os como ponto de partida explícito, não como configuração ativa.

## Arquitetura e fluxo

```text
prompt UTF-8 ──► Transformer byte-level ──┐
timestep ────► embedding senoidal ────────┼─► condicionamento FiLM
                                           │
imagem ruidosa ─► U-Net (2 downsamples) ─► gargalo + atenção cruzada
                  │              │                         │
                  └──── conexões de skip ◄─────────────────┘
                                                    │
                                             ruído previsto
                                                    │
                                         passo DDIM até a imagem
```

- Durante o treino, imagens normalizadas são convertidas de `[0, 1]` para `[-1, 1]`, recebem ruído e a U-Net aprende a prever esse ruído com MSE.
- A U-Net usa dois níveis de redução espacial, blocos residuais com modulação FiLM e atenção cruzada no gargalo para relacionar posições da imagem aos tokens do prompt.
- O *classifier-free guidance* remove captions independentemente por amostra no treino. Na geração, as previsões condicional e incondicional são combinadas por `guidance_scale`.
- O sampler faz transições DDIM diretamente entre timesteps selecionados; isso permite usar menos passos que os 100 timesteps do scheduler sem aplicar uma posterior DDPM adjacente de forma incorreta.

## Limitações e escopo

- Não há pesos distribuídos, treino em GPU configurável pela CLI, download de dados/modelos nem integração com serviços externos.
- O pipeline de treino atual aceita PNG no estágio de decodificação, embora outros formatos sejam reconhecidos na validação.
- Controles espaciais de pose, composição, câmera, identidade persistente, páginas completas de mangá e consistência entre cenas/quadrinhos **ainda não estão implementados** no denoiser principal.
- Os módulos em `src/models/` também preservam um núcleo leve/experimental de tensores e geração iterativa para testes de contratos; o caminho recomendado para treinar e amostrar imagens é `DiffusionTrainingModel` usado pelos scripts.

## Testes e documentação

Execute a suíte completa:

```bash
python -m unittest discover -s tests -v
```

Documentos complementares:

- [Arquitetura de difusão](docs/model-architecture.md)
- [Treinamento local](docs/training.md)
- [Formato e preparação do dataset](docs/dataset.md)
- [Arquitetura do núcleo experimental](docs/architecture.md)
- [Roadmap](docs/roadmap.md)

## Licença

Distribuído sob a [licença MIT](LICENSE).
