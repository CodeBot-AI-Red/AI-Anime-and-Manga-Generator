# ✨ AI Anime & Manga Generator

> Gerador local de imagens de anime e mangá por difusão, **treinado do zero** e **estritamente dedicado ao estilo anime/mangá**. Suporta perfis de **grande porte (182M)** e **ultra grande porte (440M)**, com **visualizador de imagens ANSI TrueColor no terminal**, **visão neural ViT**, **relatórios interativos em HTML** e **regras de estúdio de animação** integradas.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)
![Params](https://img.shields.io/badge/par%C3%A2metros-656K%20%7C%20182M%20%7C%20440M-ff69b4)
![Domain](https://img.shields.io/badge/dom%C3%ADnio-Anime%20%26%20Mang%C3%A1-violet)
![License](https://img.shields.io/badge/license-MIT-2ea44f)

---

Não utiliza pesos pré-treinados externos, APIs de terceiros ou coleta de dados. Você fornece suas imagens e anotações **com direitos de uso**, treina 100% localmente e gera suas ilustrações e páginas de mangá de forma privada.

> [!IMPORTANT]
> - **Perfil Padrão:** O perfil padrão é **`large` (182,4 milhões de parâmetros, 256×256)**, recomendado para GPU.
> - **Perfil Ultra:** O perfil **`xlarge` (439,9 milhões de parâmetros, 512×512)** é voltado para alta resolução e exige GPU com mais VRAM.
> - **Perfil Leve:** Para CPU, testes rápidos e desenvolvimento, utilize o perfil **`lite` (656 mil parâmetros, 32×32)**.
> - *Sem um checkpoint treinado localmente, os pesos permanecem aleatórios e a amostragem gera ruído.*

---

## 📋 Índice

- [Destaques e Funcionalidades](#-destaques-e-funcionalidades)
- [Capacidade e Perfis do Modelo](#-capacidade-e-perfis-do-modelo)
- [Início Rápido](#-início-rápido)
- [Visão Neural e Crítico de Estúdio](#-visão-neural-e-crítico-de-estúdio)
- [Regras de Estúdio Aplicadas](#-regras-de-estúdio-aplicadas)
- [Guia Completo de Parâmetros](#-guia-completo-de-parâmetros)
  - [Arquitetura (`ModelConfig`)](#1-configuração-da-arquitetura-modelconfig)
  - [CLI de Geração (`scripts/generate.py`)](#2-cli-de-geração-scriptsgeneratepy)
  - [CLI de Treinamento (`scripts/train.py`)](#3-cli-de-treinamento-scriptstrainpy)
  - [CLIs de Inspeção e Visualização](#4-clis-de-inspeção-e-visualização)
- [Arquitetura e Fluxo do Sistema](#-arquitetura-e-fluxo-do-sistema)
- [Limitações](#-limitações)
- [Testes e Verificação](#-testes-e-verificação)
- [Licença](#-licença)

---

## 🚀 Destaques e Funcionalidades

| Recurso | Descrição |
| --- | --- |
| **Domínio Estrito Anime & Mangá** | Sanitização e validação automática de prompts (`validate_manga_anime_domain`, `enforce_manga_anime_style`), garantindo fidelidade técnica às estética japonesa tradicional. |
| **Perfis Escalonáveis (`lite`, `large`, `xlarge`)** | Suporte nativo desde o perfil **`lite` (32×32, 656K params)** para testes de CPU, até **`large` (256×256, 182M params)** e **`xlarge` (512×512, 440M params)**. |
| **Visualizador de Imagens no Terminal** | Exibe as artes geradas ou inspecionadas diretamente no terminal usando caracteres de bloco e cores **ANSI TrueColor 24-bit** (`--view` / `--show`). |
| **Relatórios Interativos em HTML** | Gera galerias e relatórios visuais ricos em HTML (`--html`), permitindo comparar a imagem gerada, imagem de referência e os diagnósticos do crítico. |
| **Modo Mangá / Screentone** | Opção `--manga` com conversão monocromática de alta fidelidade e padrão de retículas estilizadas (*halftone screentones*). |
| **Encoder de Texto Transformer** | Processa até **512 bytes** de prompt com **12 a 16 camadas Transformer**, suportando prompts ricos e regras de estúdio completas. |
| **Visão Neural (`VisionEncoder`)** | Vision Transformer (**ViT de 8 a 12 camadas**, patches 16×16) que injeta até **4 imagens de referência** via atenção cruzada para manter consistência de personagem e estilo. |
| **Auto-revisão / Autocorreção** | O `StudioStyleCritic` analisa a imagem gerada e, se a pontuação estiver abaixo da meta (`--target-score`), aciona rodadas automáticas de refinamento img2img (`--refine-rounds`). |
| **Aprimoramento / Corregedoria (`img2img`)** | Permite refazer ou corrigir artes existentes com `--init-image` e controle de força (`--strength`). |
| **Regras dos Guias Settei** | Incorpora 7 pilares do visual de estúdio, 5 paletas *settei* com valores HEX reais, lineart em tom marrom quente RGB(61,43,40) e sombreamento *cel shading*. |

---

## 📊 Capacidade e Perfis do Modelo

A arquitetura e a contagem exata de parâmetros de cada perfil do modelo são detalhadas abaixo:

### Contagem de Parâmetros Treináveis por Componente

| Componente da Arquitetura | `xlarge` (512×512) | `large` (Padrão, 256×256) | `lite` (CPU / Teste, 32×32) |
| --- | ---: | ---: | ---: |
| **Embedding temporal** (`time_embedding`) | 2.099.200 | 1.181.184 | 8.320 |
| **Encoder de prompt** (`prompt_encoder`) | 202.328.064 | 85.645.824 | 124.672 |
| **Encoder visual** (`vision_encoder`, ViT) | 152.212.480 | 57.450.240 | 113.856 |
| **Fusão de condicionamento** (`condition`) | 3.147.776 | 1.771.008 | 12.416 |
| **Reduções e projeções espacial** (In/Down/Up Proj) | 11.803.584 | 5.247.616 | 82.512 |
| **Blocos residuais da U-Net** (ResBlocks) | 65.628.864 | 29.834.368 | 297.360 |
| **Atenção cruzada** (`cross_attention`) | 2.757.120 | 1.313.792 | 16.768 |
| **Camada de saída** (`output`) | 5.571 | 3.715 | 467 |
| **Total de Parâmetros Treináveis** | **439.982.659 (~440M)** | **182.447.747 (~182M)** | **656.371 (~656K)** |

Para medir a contagem de parâmetros de qualquer configuração personalizada via Python:

```python
from src.models.config import ModelConfig
from src.models.diffusion import DiffusionDenoiser, count_trainable_parameters

config = ModelConfig.large()  # ou ModelConfig.xlarge(), ModelConfig.lite()
denoiser = DiffusionDenoiser(config)
print(f"Parâmetros treináveis: {count_trainable_parameters(denoiser):,}")
```

---

## ⚡ Início Rápido

### 1. Instalação do Ambiente

```bash
git clone <URL-DO-SEU-REPOSITORIO> && cd AI-Anime-and-Manga-Generator
python -m venv .venv
source .venv/bin/activate        # No Windows: .venv\Scripts\activate
pip install -r requirements.txt  # Instala PyTorch 2.2+
```

### 2. Estrutura do Dataset Local

Organize suas imagens locais e anotações (arquivos `.json` ou `.txt` com mesmo nome base):

```text
data/
├── raw/
│   ├── chapter-01/
│   │   ├── page-001.png
│   │   └── page-002.png
└── annotations/
    └── chapter-01/
        ├── page-001.json
        └── page-002.json
```

### 3. Treinar o Modelo

```bash
# Treinamento com o perfil padrão (large, 256x256, GPU)
python scripts/train.py \
  --images data/raw \
  --annotations data/annotations \
  --epochs 100 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --device auto

# Para ultra resolução (xlarge, 512x512)
python scripts/train.py --images data/raw --annotations data/annotations --preset xlarge

# Para testes rápidos em CPU
python scripts/train.py --images data/raw --annotations data/annotations --preset lite
```

### 4. Gerar Imagens

```bash
# Geração padrão com checkpoint
python scripts/generate.py \
  --checkpoint checkpoints/epoch-100.pt \
  --prompt "guerreira de cabelo vermelho em uma cidade ao entardecer" \
  --palette fantasia \
  --output saida/cena01.png \
  --view
```

---

## 👁️ Visão Neural e Crítico de Estúdio

O modelo integra duas abordagens complementares de visão:

1. **Visão Neural (`VisionEncoder`)**: Um Vision Transformer (ViT) treinado do zero extrai vetores de características de até 4 imagens de referência enviadas pelo usuário ou geradas anteriormente. Esses vetores são injetados na U-Net por atenção cruzada, garantindo consistência visual.
2. **Crítico de Estúdio (`StudioStyleCritic`)**: Avalia a qualidade da imagem gerada medindo características de cel shading, ruído em superfícies chapadas, cor de lineart, contraste, saturação e composição (regra dos terços).

```bash
# Visualizar a imagem no terminal e gerar relatório HTML
python scripts/generate.py --checkpoint ck.pt \
  --prompt "heroína anime em cena de ação" \
  --view --cols 70 --html relatorio.html

# Gerar em estilo mangá monocromático com retículas (screentone)
python scripts/generate.py --checkpoint ck.pt \
  --prompt "samurai em duelo sob chuva" \
  --manga --view

# Usar imagem de referência (ex: ficha de personagem / settei)
python scripts/generate.py --checkpoint ck.pt \
  --prompt "mesma heroína sorrindo no festival" \
  --reference ficha_aiko.png --view

# Aprimorar/Refazer uma arte sua (img2img)
python scripts/generate.py --checkpoint ck.pt \
  --prompt "retrato anime, iluminação dramática" \
  --init-image meu_esboco.png --strength 0.5 --view

# Apenas inspecionar uma imagem existente
python scripts/inspect_image.py minha_arte.png --reference ficha_aiko.png --view --html analise.html
```

---

## 🎨 Regras de Estúdio Aplicadas

Implementadas no módulo `src/studio/`, com base em folhas de modelo (*settei*) e diretrizes de colorização industrial de anime:

| Regra de Estúdio | Aplicação no Sistema |
| --- | --- |
| **Olhos em Camadas** | Esclerótica branca pura, pupila escura, íris com gradiente de cor e 2 destaques de luz (*catchlights*). Aplicado via prompt, checagens do crítico e acabamento. |
| **Lineart Quente RGB(61,43,40)** | Linhas contornadas nunca em preto absoluto `(0,0,0)`, mas sim em tom marrom escuro quente. Corrigido automaticamente no acabamento. |
| **Cel Shading Limpo** | Sombreamento em blocos com bordas nítidas quantizadas em 3 níveis (Destaque/Normal/Sombra). |
| **Pele Chapada de Anime** | Eliminação de texturas de poros ou granulados fotográficos em áreas de pele através de suavização direcionada. |
| **Realce de Saturação** | Elevação automática de +20% a +25% na saturação de cores para um visual vibrante típico de estúdio. |
| **Paletas Settei Pré-definidas** | Fichas de cores selecionáveis via `--palette` (`fantasia`, `scifi`, `shonen`, `idol`, `dark`, `manga_monochrome`). |

---

## ⚙️ Guia Completo de Parâmetros

### 1. Configuração da Arquitetura (`ModelConfig`)

Hiperparâmetros configuráveis da U-Net, encoders e difusão:

| Parâmetro | `xlarge` | `large` (Padrão) | `lite` | Descrição |
| --- | ---: | ---: | ---: | --- |
| `image_resolution` | `(512, 512)` | `(256, 256)` | `(32, 32)` | Resolução espacial de treino e geração (largura, altura) |
| `image_channels` | `3` | `3` | `3` | Número de canais de cor (RGB) |
| `base_channels` | `192` | `128` | `16` | Canais base do 1º estágio da U-Net (escala com profundidade) |
| `latent_channels` | `16` | `16` | `16` | Canais latentes internos do modelo |
| `unet_res_blocks` | `3` | `3` | `2` | Número de blocos residuais por nível na U-Net |
| `downsample_factor` | `2` | `2` | `2` | Fator de redução de resolução espacial entre níveis |
| `num_timesteps` | `1000` | `1000` | `100` | Passos de adpção de ruído no processo de difusão |
| `generation_steps` | `50` | `50` | `4` | Passos de amostragem durante a geração DDIM |
| `time_embedding_dim` | `1024` | `768` | `64` | Dimensão do vetor de condicionamento temporal/texto/visão |
| `guidance_scale` | `8.5` | `7.5` | `3.0` | Força do Classifier-Free Guidance (CFG) |
| `condition_dropout` | `0.1` | `0.1` | `0.1` | Probabilidade de descartar o texto no treino |
| `text_vocab_size` | `256` | `256` | `256` | Tamanho do vocabulário byte-level |
| `max_prompt_length` | `512` | `512` | `128` | Comprimento máximo do prompt em bytes |
| `text_encoder_layers` | `16` | `12` | `2` | Camadas Transformer do encoder de texto |
| `attention_heads` | `16` | `16` | `4` | Número de cabeças de atenção na U-Net e encoders |
| `vision_enabled` | `True` | `True` | `True` | Ativa o encoder visual Vision Transformer (ViT) |
| `vision_patch_size` | `16` | `16` | `8` | Tamanho dos patches (px) divididos pelo ViT |
| `vision_layers` | `12` | `8` | `2` | Número de camadas do Transformer visual |
| `vision_input_resolution` | `256` | `224` | `32` | Resolução de redimensionamento para imagens de referência |
| `max_reference_images` | `4` | `4` | `4` | Máximo de imagens de referência injetadas por geração |
| `reference_dropout` | `0.5` | `0.5` | `0.5` | Taxa de dropout de imagens de referência no treino |
| `generation_profile` | `"xlarge"` | `"large"` | `"lite"` | Nome do perfil de geração utilizado |

---

### 2. CLI de Geração (`scripts/generate.py`)

Gera imagens, aplica acabamento de estúdio, executa auto-revisão e visualização:

| Opção | Valor Padrão | Descrição |
| --- | --- | --- |
| `--prompt` | `"guerreira anime em uma cidade ao entardecer"` | Descrição textual da cena desejada |
| `--negative-prompt` | `None` | Elementos adicionais a evitar (combinados ao negativo de estúdio) |
| `--palette` | `None` | Escolha de paleta settei: `fantasia`, `scifi`, `shonen`, `idol`, `dark`, `manga_monochrome` |
| `--checkpoint` | `None` | Caminho para o arquivo `.pt` de checkpoint treinado |
| `--preset` | `"large"` | Perfil do modelo se sem checkpoint: `large` (256x256), `xlarge` (512x512), `lite` (32x32) |
| `--manga` | `False` | Ativa o modo de acabamento mangá monocromático com retículas (screentone) |
| `--output` | `generated.png` | Caminho do arquivo PNG de saída (gera também `.report.json`) |
| `--resolution` | `None` | Sobrescreve a resolução de geração (largura/altura em pixels) |
| `--steps` | `None` (50) | Sobrescreve a quantidade de passos de amostragem DDIM |
| `--guidance-scale` | `None` (7.5/8.5) | Sobrescreve o multiplicador de CFG |
| `--reference` | `[]` | Caminho de imagem de referência (pode ser usado até 4 vezes) |
| `--init-image` | `None` | Imagem inicial para processo img2img de correção |
| `--strength` | `0.6` | Força de modificação na imagem inicial (0.0 a 1.0) |
| `--refine-rounds` | `1` | Número máximo de rodadas de autocorreção caso a nota fique abaixo da meta |
| `--target-score` | `0.75` | Nota do crítico (0.0 a 1.0) para considerar a imagem dentro dos padrões |
| `--seed` | `42` | Semente aleatória para reprodutibilidade |
| `--device` | `cuda` (se disponível) | Dispositivo de processamento (`cuda`, `mps`, `cpu`) |
| `--view`, `--show` | `False` | Renderiza a imagem diretamente no terminal usando ANSI TrueColor |
| `--cols` | `60` | Largura em colunas de texto para a exibição no terminal |
| `--html` | `None` | Salva um relatório e galeria visual interativa em HTML no caminho informado |
| `--no-studio-rules` | `False` | Desativa a injeção de regras de estúdio automáticas no prompt |
| `--no-finish` | `False` | Desativa o pós-processamento de acabamento técnico de estúdio |

---

### 3. CLI de Treinamento (`scripts/train.py`)

Treina o denoiser a partir do dataset local de imagens e anotações:

| Opção | Valor Padrão | Descrição |
| --- | --- | --- |
| `--images` | **Obrigatório** | Diretório contendo as imagens locais de treino |
| `--annotations` | **Obrigatório** | Diretório contendo as anotações JSON ou TXT correspondentes |
| `--preset` | `"large"` | Perfil da arquitetura a treinar: `large`, `xlarge` ou `lite` |
| `--resolution` | `None` | Sobrescreve a resolução de treino (ex: `256` ou `512`) |
| `--learning-rate` | `1e-4` | Taxa de aprendizado do otimizador AdamW |
| `--device` | `"auto"` | Seleção de dispositivo (`auto` detecta CUDA -> MPS -> CPU) |
| `--epochs` | `2` | Quantidade total de épocas de treinamento |
| `--batch-size` | `2` | Tamanho do lote de dados por passo |
| `--checkpoint-dir` | `checkpoints` | Diretório onde os checkpoints `.pt` serão salvos |

---

### 4. CLIs de Inspeção e Visualização

#### `scripts/inspect_image.py` — Análise Detalhada de Imagem
- **`image`** *(Obrigatório)*: Caminho para a imagem a ser analisada.
- **`--reference`**: Imagem de referência para comparar consistência de cores e estilo.
- **`--fix`**: Caminho para salvar uma versão da imagem com acabamento de estúdio aplicado.
- **`--view` / `--show`**: Exibe a imagem no terminal via ANSI TrueColor.
- **`--cols`** *(Padrão: 60)*: Resolução em colunas para a exibição no terminal.
- **`--html`**: Salva o relatório de inspeção em formato HTML interativo.
- **`--json`**: Imprime o diagnóstico técnico no formato JSON.

#### `scripts/view_image.py` — Visualizador de Imagens
- **`image`** *(Obrigatório)*: Imagem a ser exibida.
- **`--cols`** *(Padrão: 60)*: Largura da exibição no terminal.
- **`--reference`**: Imagem de referência para cálculo de consistência.
- **`--html`**: Salva o relatório em formato galeria HTML.
- **`--no-critique`**: Oculta as métricas do crítico e exibe somente a imagem.

---

## 🏗️ Arquitetura e Fluxo do Sistema

```text
 Prompt do Usuário ──┐
                     ├─► build_prompt() ──► Regras de Estúdio + Paleta Settei
 Negativo do Usuário ┘

                            ┌─────────────────────────────────────────┐
                            │           PromptEncoder (12-16 layers)  │ ──► Tokens de Texto
                            └─────────────────────────────────────────┘          │
                                                                                 ▼
 Referências Visuais ──────► ┌─────────────────────────────────────────┐     Atenção Cruzada
 (Até 4 imagens)            │           VisionEncoder (ViT 8-12 L)    │ ──► Tokens Visuais
                            └─────────────────────────────────────────┘          │
                                                                                 ▼
 Ruído Inicial (BCHW) ──────► ┌─────────────────────────────────────────┐    U-Net Denoiser
 Imagem Init (img2img) ────►  │  DownBlocks ─► Middle ─► UpBlocks    │ ◄── (182M a 440M)
                              └─────────────────────────────────────────┘
                                                 │
                                                 ▼
                                        Amostragem DDIM (50 Passos)
                                                 │
                                                 ▼
                                       apply_studio_finish()
                               (Lineart Marrom · Cel Shading · Saturação)
                                                 │
                                                 ▼
                                       StudioStyleCritic.analyze()
                                                 │
                                ┌────────────────┴────────────────┐
                         Nota >= Meta?                     Nota < Meta?
                                │                                 │
                                ▼                                 ▼
                         Geração Final               Refinamento Autocorreção
                        (PNG + HTML + ANSI)         (Nova rodada com prompt ajustado)
```

---

## ⚠️ Limitações

1. **Treinamento do Zero**: O modelo não utiliza pesos pré-treinados externos. A qualidade e o fidedignismo anatômico dependem diretamente do tamanho e qualidade do seu conjunto de dados local.
2. **Análise do Crítico**: O `StudioStyleCritic` realiza análise estatística de pixels (distribuição de cores, bordas, ruído e regra dos terços); ele não realiza reconhecimento anatômico complexo (ex: contagem precisa de dedos). Anatomia correta é reforçada através dos prompts negativos de estúdio.

---

## 🧪 Testes e Verificação

A suíte de testes unitários cobre validação de dados, amostragem por difusão, encoders visuais, regras de acabamento e salvamento de checkpoints:

```bash
python3 -m unittest discover -s tests
```

---

## 📜 Licença

Distribuído sob a licença **MIT**. Veja [LICENSE](LICENSE) para mais detalhes.
