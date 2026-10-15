# 🎨 AI Anime & Manga Generator (Grande Porte Estúdio)

> **IA Geradora de Imagens de Alta Capacidade para Anime e Mangá**, treinada 100% do zero e **estritamente dedicada à estética visual japonesa (Anime & Mangá)**. Projetada para modelos de grande porte com até **1,43 Bilhão de Parâmetros**, oferecendo suporte a **Flash Attention**, **Precisão Mista (AMP FP16/BF16)**, **Rastreamento EMA**, **Vision Transformer (ViT)**, **Auto-revisão por Visão Neural**, **Retículas Halftone de Mangá** e **19 Paletas Industriais Settei**.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Params](https://img.shields.io/badge/par%C3%A2metros-669K%20%7C%20183.5M%20%7C%20442.1M%20%7C%201.43B%2B-ff69b4?style=for-the-badge)
![Domain](https://img.shields.io/badge/dom%C3%ADnio-Exclusivo%20Anime%20%26%20Mang%C3%A1-violet?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-2ea44f?style=for-the-badge)

---

## 🌸 Sobre o Projeto

Este repositório contém uma arquitetura de IA generativa de **grande e supremo porte**, desenvolvida para o domínio **exclusivo de ilustrações de anime e páginas de mangá**. O sistema não utiliza APIs externas, pesos pré-treinados ocidentais ou dados fotorrealistas. Todo o treinamento e geração ocorrem localmente com total controle sobre estilo, paletas de cores, *lineart* e regras de produção de estúdios japoneses (*settei*).

> [!IMPORTANT]
> ### 👑 Perfis de Capacidade do Modelo
> - **Perfil Estúdio Supremo (`xxlarge`):** **1.430.660.035 (~1,43 Bilhão de Parâmetros, 1024×1024)** — Arquitetura de grande porte para ilustrações de alta fidelidade e riqueza de detalhes profissionais.
> - **Perfil Ultra (`xlarge`):** **442.149.955 (~442,1 Milhões de Parâmetros, 512×512)** — Alta resolução balanceada para GPUs com VRAM elevada.
> - **Perfil Padrão (`large`):** **183.499.395 (~183,5 Milhões de Parâmetros, 256×256)** — Recomendado para treino e amostragem diários em GPUs intermediárias.
> - **Perfil Leve (`lite`):** **669.043 (~669 Mil Parâmetros, 32×32)** — Ideal para testes unitários, validação em CPU e depuração rápida.

---

## 📋 Índice

- [✨ Destaques e Funcionalidades](#-destaques-e-funcionalidades)
- [📊 Capacidade e Parâmetros Atualizados](#-capacidade-e-parâmetros-atualizados)
- [⚡ Início Rápido](#-início-rápido)
- [🎨 Regras de Estúdio e 19 Paletas Settei](#-regras-de-estúdio-e-19-paletas-settei)
- [👁️ Visão Neural e Crítico de Estúdio](#-visão-neural-e-crítico-de-estúdio)
- [⚙️ Guia Completo de Parâmetros e CLI](#️-guia-completo-de-parâmetros-e-cli)
  - [1. Configuração da Arquitetura (`ModelConfig`)](#1-configuração-da-arquitetura-modelconfig)
  - [2. CLI de Geração (`scripts/generate.py`)](#2-cli-de-geração-scriptsgeneratepy)
  - [3. CLI de Treinamento (`scripts/train.py`)](#3-cli-de-treinamento-scriptstrainpy)
  - [4. CLIs de Inspeção e Visualização](#4-clis-de-inspeção-e-visualização)
- [🏗️ Arquitetura e Fluxo do Sistema](#️-arquitetura-e-fluxo-do-sistema)
- [🧪 Testes e Verificação](#-testes-e-verificação)
- [📜 Licença](#-licença)

---

## ✨ Destaques e Funcionalidades

| Funcionalidade | Descrição |
| --- | --- |
| **Domínio Estrito Anime & Mangá** | Validação e sanitização automática de prompts (`validate_manga_anime_domain`, `enforce_manga_anime_style`), bloqueando contaminação por fotorrealismo, 3D ou estilos ocidentais. |
| **Arquitetura de Grande Porte (Até 1.43B)** | Suporte nativo a 4 perfis escalonáveis: `lite` (32×32), `large` (256×256), `xlarge` (512×512) e `xxlarge` (1024×1024, 1.43B+ params). |
| **Atenção Otimizada (Flash Attention)** | Implementação de `F.scaled_dot_product_attention` para processamento acelerado e menor consumo de VRAM em resoluções até 1024×1024. |
| **Treinamento de Alta Eficiência** | Suporte a Precisão Mista (AMP FP16/BF16), Acúmulo de Gradientes e rastreamento Exponential Moving Average (EMA). |
| **Modo Mangá & Retículas Screentone** | Renderização monocromática estilo mangá com padrões de retícula *halftone* (`--manga`) e paletas dedicadas (`manga_shonen`, `manga_shojo`, `manga_seinen`, `manga_monochrome`). |
| **Encoder Visual ViT (`VisionEncoder`)** | Vision Transformer (8 a 20 camadas) para injeção de até 4 imagens de referência via atenção cruzada, garantindo consistência de personagens. |
| **Autocorreção e Refinamento de Estúdio** | O `StudioStyleCritic` inspeciona a arte e, se a nota estiver abaixo da meta (`--target-score`), executa rodadas automáticas de refinamento img2img (`--refine-rounds`). |
| **Terminal TrueColor & Relatórios HTML** | Exibição de imagens no terminal em ANSI TrueColor (`--view`) e exportação de galerias e laudos diagnósticos em HTML (`--html`). |

---

## 📊 Capacidade e Parâmetros Atualizados

Abaixo está a contagem **exata e atualizada** de parâmetros treináveis calculada diretamente na arquitetura PyTorch do projeto:

### 🔬 Tabela de Parâmetros por Componente

| Componente da U-Net / Encoders | `xxlarge` (1024×1024) | `xlarge` (512×512) | `large` (256×256) | `lite` (32×32) |
| --- | ---: | ---: | ---: | ---: |
| **Embedding Temporal** (`time_embedding`) | 4.721.664 | 2.099.200 | 1.181.184 | 8.320 |
| **Encoder de Prompt** (`prompt_encoder`) | 567.813.120 | 202.328.064 | 85.645.824 | 124.672 |
| **Encoder Visual ViT** (`vision_encoder`) | 568.708.608 | 152.212.480 | 57.450.240 | 113.856 |
| **Fusão de Condição** (`condition`) | 7.080.960 | 3.147.776 | 1.771.008 | 12.416 |
| **Projeção de Entrada** (`input`) | 8.960 | 5.376 | 3.584 | 448 |
| **Blocos Iniciais U-Net** (`down_blocks`) | 11.315,200 | 3.174.912 | 1.477.632 | 13.568 |
| **Amostragem Down 1** (`downsample`) | 3.277.440 | 1.180.032 | 524.544 | 8.224 |
| **Blocos Profundos U-Net** (`down_blocks_deep`) | 37.376.000 | 10.331.136 | 4.724.736 | 45.568 |
| **Atenção Cruzada Down** (`down_cross_attn`) | 2.789.120 | 1.083.648 | 525.824 | 6.336 |
| **Amostragem Down 2** (`downsample_deep`) | 13.108.480 | 4.719.360 | 2.097.664 | 32.832 |
| **Bloco Central U-Net** (`middle`) | 133.734.400 | 36.587.520 | 16.527.360 | 164.864 |
| **Atenção Cruzada Central** (`cross_attention`) | 7.216.640 | 2.757.120 | 1.313.792 | 16.768 |
| **Amostragem Up 1** (`upsample_deep`) | 13.107.840 | 4.718.976 | 2.097.408 | 32.800 |
| **Blocos de Subida** (`up_blocks_deep`) | 41.883.520 | 11.954.304 | 5.446.400 | 56.928 |
| **Atenção Cruzada Up** (`up_cross_attn`) | 2.789.120 | 1.083.648 | 525.824 | 6.336 |
| **Amostragem Up 2** (`upsample`) | 3.277.120 | 1.179.840 | 524.416 | 8.208 |
| **Blocos Finais U-Net** (`up_blocks`) | 12.442.560 | 3.580.992 | 1.658.240 | 16.432 |
| **Camada de Saída** (`output`) | 9.283 | 5.571 | 3.715 | 467 |
| **TOTAL DE PARÂMETROS TREINÁVEIS** | **1.430.660.035 (~1.43B)** | **442.149.955 (~442.1M)** | **183.499.395 (~183.5M)** | **669.043 (~669K)** |

Para checar os parâmetros em tempo de execução Python:

```python
from src.models.config import ModelConfig
from src.models.diffusion import DiffusionDenoiser, count_trainable_parameters

config = ModelConfig.xxlarge()
model = DiffusionDenoiser(config)
print(f"Parâmetros do modelo supremo: {count_trainable_parameters(model):,}")
```

---

## ⚡ Início Rápido

### 1. Instalação e Ambiente

```bash
git clone <URL-DO-REPOSITORIO>
cd AI-Anime-and-Manga-Generator
python -m venv .venv
source .venv/bin/activate        # No Windows: .venv\Scripts\activate
pip install -r requirements.txt  # Instala PyTorch 2.2+
```

### 2. Treinar o Modelo Localmente

```bash
# Treino do Perfil Supremo (xxlarge, 1024x1024, 1.43B params, GPU)
python scripts/train.py \
  --images data/raw \
  --annotations data/annotations \
  --preset xxlarge \
  --epochs 100 \
  --batch-size 4 \
  --gradient-accumulation-steps 2 \
  --mixed-precision \
  --learning-rate 1e-4

# Treino no Perfil Padrão (large, 256x256)
python scripts/train.py --images data/raw --annotations data/annotations --preset large
```

### 3. Gerar Ilustrações e Páginas de Mangá

```bash
# Geração no perfil supremo (1024x1024) com visualização no terminal
python scripts/generate.py \
  --checkpoint checkpoints/epoch-100.pt \
  --prompt "guerreira de cabelo vermelho em uma cidade ao entardecer" \
  --palette fantasia \
  --output saida/ilustracao.png \
  --view

# Geração de Mangá Monocromático com Retículas (screentone)
python scripts/generate.py \
  --checkpoint checkpoints/epoch-100.pt \
  --prompt "samurai em duelo sob chuva intensa" \
  --palette manga_shonen \
  --manga \
  --view
```

---

## 🎨 Regras de Estúdio e 19 Paletas Settei

O módulo `src/studio/` incorpora as diretrizes de colorização e traço dos principais estúdios de animação e editoras de mangá:

| Regra de Produção | Aplicação no Sistema |
| --- | --- |
| **Olhos em Camadas** | Esclerótica branca pura, pupila com profundidade, íris em gradiente e múltiplos destaques de luz (*catchlights*). |
| **Lineart Quente RGB(61,43,40)** | Linhas contornadas em tom marrom escuro quente, evitando a rigidez do preto puro `(0,0,0)`. |
| **Cel Shading Limpo** | Sombreamento quantizado em blocos definidos (Base, Sombra e Destaque), eliminando gradientes indesejados. |
| **19 Paletas Settei** | Opções prontas via `--palette`: `fantasia`, `scifi`, `shonen`, `idol`, `dark`, `manga_shonen`, `manga_shojo`, `manga_seinen`, `manga_monochrome`, `anime_modern`, `anime_retro_90s`, `cyberpunk_anime`, `fantasy_isekai`, `slice_of_life` e mais. |

---

## 👁️ Visão Neural e Crítico de Estúdio

- **Visão Neural (`VisionEncoder`):** Extrai características de até 4 imagens de referência (fichas de personagens, mapas ou esboços) e injeta na U-Net via atenção cruzada.
- **Crítico de Estúdio (`StudioStyleCritic`):** Mede a qualidade técnica da imagem gerada (cel shading, saturação, lineart, regra dos terços) e calcula uma pontuação de 0% a 100%.

```bash
# Geração com imagem de referência e relatório HTML
python scripts/generate.py \
  --prompt "heroína anime no festival de verão" \
  --reference referencia_personagem.png \
  --preset xxlarge \
  --view --html relatorio.html
```

---

## ⚙️ Guia Completo de Parâmetros e CLI

### 1. Configuração da Arquitetura (`ModelConfig`)

| Parâmetro | `xxlarge` (Estúdio) | `xlarge` (Ultra) | `large` (Padrão) | `lite` (CPU) | Descrição |
| --- | ---: | ---: | ---: | ---: | --- |
| `image_resolution` | `(1024, 1024)` | `(512, 512)` | `(256, 256)` | `(32, 32)` | Resolução espacial de treino e amostragem |
| `base_channels` | `320` | `192` | `128` | `16` | Canais base do 1º estágio da U-Net |
| `unet_res_blocks` | `4` | `3` | `3` | `2` | Blocos residuais por nível da U-Net |
| `num_timesteps` | `1000` | `1000` | `1000` | `100` | Passos do processo de difusão |
| `generation_steps` | `50` | `50` | `50` | `4` | Passos de amostragem DDIM |
| `time_embedding_dim` | `1536` | `1024` | `768` | `64` | Dimensão do vetor de tempo e contexto |
| `guidance_scale` | `9.0` | `8.5` | `7.5` | `3.0` | Multiplicador de Classifier-Free Guidance (CFG) |
| `text_encoder_layers` | `20` | `16` | `12` | `2` | Camadas Transformer do encoder de prompt |
| `attention_heads` | `16` | `16` | `16` | `4` | Cabeças de atenção na U-Net e encoders |
| `vision_layers` | `20` | `12` | `8` | `2` | Camadas do Vision Transformer (ViT) |
| `vision_input_resolution` | `384` | `256` | `224` | `32` | Resolução do ViT para referências visuais |

---

### 2. CLI de Geração (`scripts/generate.py`)

| Argumento | Padrão | Descrição |
| --- | --- | --- |
| `--prompt` | `"guerreira anime em uma cidade ao entardecer"` | Descrição em texto da ilustração ou cena de mangá |
| `--negative-prompt` | `None` | Elementos adicionais a evitar |
| `--palette` | `None` | Escolha da ficha de cores settei (19 opções disponíveis) |
| `--checkpoint` | `None` | Caminho do arquivo `.pt` de modelo treinado |
| `--preset` | `"large"` | Perfil da arquitetura: `large`, `xlarge`, `xxlarge` ou `lite` |
| `--manga` | `False` | Ativa o modo mangá monocromático com retículas (screentone) |
| `--output` | `generated.png` | Caminho do arquivo PNG de saída |
| `--steps` | `50` | Quantidade de passos de amostragem DDIM |
| `--guidance-scale` | `7.5`–`9.0` | Força de alinhamento com o prompt (CFG) |
| `--reference` | `[]` | Imagens de referência para o VisionEncoder (até 4) |
| `--init-image` | `None` | Imagem inicial para processo img2img de correção |
| `--strength` | `0.6` | Força de alteração na imagem inicial em img2img |
| `--refine-rounds` | `1` | Rodadas de autocorreção caso a pontuação do crítico fique abaixo do alvo |
| `--target-score` | `0.75` | Pontuação mínima desejada do crítico (0.0 a 1.0) |
| `--view`, `--show` | `False` | Exibe a imagem gerada diretamente no terminal via ANSI TrueColor |
| `--cols` | `60` | Largura em colunas da exibição no terminal |
| `--html` | `None` | Exporta relatório visual interativo em galeria HTML |

---

### 3. CLI de Treinamento (`scripts/train.py`)

| Argumento | Padrão | Descrição |
| --- | --- | --- |
| `--images` | **Obrigatório** | Diretório das imagens de treino |
| `--annotations` | **Obrigatório** | Diretório das anotações JSON/TXT |
| `--preset` | `"large"` | Perfil do modelo: `large`, `xlarge`, `xxlarge` ou `lite` |
| `--gradient-accumulation-steps` | `1` | Passos de acúmulo de gradiente para economizar VRAM |
| `--mixed-precision` | `False` | Ativa precisão mista automática (AMP FP16/BF16) |
| `--learning-rate` | `1e-4` | Taxa de aprendizado para o otimizador AdamW |
| `--epochs` | `2` | Número total de épocas |
| `--batch-size` | `2` | Tamanho do lote por passo |
| `--checkpoint-dir` | `checkpoints` | Diretório de salvamento dos modelos `.pt` |

---

### 4. CLIs de Inspeção e Visualização

- **`scripts/inspect_image.py`**: Analisa uma imagem existente, mede métricas de estúdio e gera diagnóstico com sugestões de correção.
- **`scripts/view_image.py`**: Visualizador rápido de imagens no terminal em ANSI TrueColor com exportação opcional de relatórios HTML.

---

## 🏗️ Arquitetura e Fluxo do Sistema

```text
 Prompt do Usuário ──► build_prompt() ──► Regras de Estúdio + Paleta Settei
                                                    │
                                                    ▼
 Referências Visuais ──► VisionEncoder (ViT) ──► Atenção Cruzada
                                                    │
 Ruído / Init Img   ──► U-Net Denoiser ────────► Amostragem DDIM
                                                    │
                                                    ▼
                                          apply_studio_finish()
                                  (Lineart Marrom · Cel Shading · Retículas)
                                                    │
                                                    ▼
                                        StudioStyleCritic.analyze()
                                                    │
                                    ┌───────────────┴───────────────┐
                              Nota >= Alvo                    Nota < Alvo
                                    │                               │
                                    ▼                               ▼
                             Arte Final PNG               Refinamento img2img
                            + Visão Terminal             (Rodada de Autocorreção)
```

---

## 🧪 Testes e Verificação

A suíte de testes unitários garante o correto funcionamento de todos os componentes da U-Net, encoders de texto e visão, regras de estúdio e agendadores de ruído:

```bash
python3 -m unittest discover -s tests
```

---

## 📜 Licença

Distribuído sob a licença **MIT**. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.
