# 🎨 AI Anime & Manga Generator (Grande Porte Estúdio)

> **IA Geradora de Imagens de Alta Capacidade Exclusiva para Anime e Mangá**, treinada 100% do zero e **estritamente dedicada à estética visual japonesa de Animes e Páginas de Mangá**. Projetada para modelos de grande porte com até **1,43 Bilhão de Parâmetros**, com suporte nativo a **Flash Attention**, **Precisão Mista (AMP FP16/BF16)**, **Rastreamento EMA**, **Vision Transformer (ViT)**, **Auto-revisão por Visão Neural**, **Retículas Screentone**, **Linhas de Velocidade (Speed Lines)**, **Calhas de Painéis de Mangá** e **Folhas de Settei de Produção de Estúdio**.

![Domain](https://img.shields.io/badge/dom%C3%ADnio-Exclusivo%20Anime%20%26%20Mang%C3%A1-violet?style=for-the-badge&logo=dependabot)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Params](https://img.shields.io/badge/par%C3%A2metros-669K%20%7C%20183.5M%20%7C%20442.1M%20%7C%201.43B-ff69b4?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-2ea44f?style=for-the-badge)

---

## 🌸 Sobre o Projeto

Este repositório contém uma solução de IA generativa de **grande porte profissional**, projetada do zero **única e exclusivamente para o universo de animações japonesas e mangás**. O sistema não utiliza pesos ocidentais pré-treinados ou datasets fotorrealistas. Todo o processamento, treinamento e pós-processamento ocorrem localmente com total fidelidade às regras de produção dos estúdios japoneses (*settei*, *cel shading*, *screentones* e *lineart*).

> [!IMPORTANT]
> ### 👑 Perfis e Capacidade do Modelo
> - **Perfil Estúdio Supremo (`xxlarge`):** **1.430.660.035 (~1,43 Bilhão de Parâmetros, 1024×1024)** — Arquitetura suprema para ilustrações de altíssima definição e riqueza técnica de animações de estúdio.
> - **Perfil Ultra (`xlarge`):** **442.149.955 (~442,1 Milhões de Parâmetros, 512×512)** — Resolução intermediária de alto desempenho para GPUs com VRAM elevada.
> - **Perfil Padrão (`large`):** **183.499.395 (~183,5 Milhões de Parâmetros, 256×256)** — Perfil balanceado recomendado para rotinas de treino e geração diárias.
> - **Perfil Leve (`lite`):** **669.043 (~669 Mil Parâmetros, 32×32)** — Perfil ultra veloz para testes unitários, validação em CPU e prototipagem.

---

## 📋 Índice

- [✨ Destaques e Funcionalidades Exclusivas](#-destaques-e-funcionalidades-exclusivas)
- [📊 Capacidade e Contagem Exata de Parâmetros](#-capacidade-e-contagem-exata-de-parâmetros)
- [⚡ Início Rápido](#-início-rápido)
- [🎨 Regras de Estúdio, Manga e 19 Paletas Settei](#-regras-de-estúdio-manga-e-19-paletas-settei)
- [👁️ Visão Neural e Crítico Autocorrector](#️-visão-neural-e-crítico-autocorrector)
- [⚙️ Guia Completo de CLI e Parâmetros](#️-guia-completo-de-cli-e-parâmetros)
  - [1. Configuração da Arquitetura (`ModelConfig`)](#1-configuração-da-arquitetura-modelconfig)
  - [2. CLI de Geração (`scripts/generate.py`)](#2-cli-de-geração-scriptsgeneratepy)
  - [3. CLI de Treinamento (`scripts/train.py`)](#3-cli-de-treinamento-scriptstrainpy)
  - [4. Ferramentas de Visualização](#4-ferramentas-de-visualização)
- [🏗️ Arquitetura e Fluxo do Sistema](#️-arquitetura-e-fluxo-do-sistema)
- [🧪 Testes e Verificação](#-testes-e-verificação)
- [📜 Licença](#-licença)

---

## ✨ Destaques e Funcionalidades Exclusivas

| Funcionalidade | Descrição |
| --- | --- |
| **Domínio Estrito Anime & Mangá** | Sanitização automática de prompts (`validate_manga_anime_domain`, `enforce_manga_anime_style`), impedindo contaminação por fotorrealismo, 3D CGI ou estilos ocidentais. |
| **Linhas de Velocidade (Speed Lines)** | Geração procedural de linhas de ação radiais estilo mangá de impacto (`--speed-lines`, `--speed-line-density`). |
| **Páginas e Painéis de Mangá** | Layout automático de quadros e calhas de mangá (`--panel-layout`, `--panel-rows`, `--panel-cols`). |
| **Fichas Settei de Personagem** | Formatação de pranchas de produção de estúdio com marcação visual e paletas de cores (`--settei-sheet`). |
| **Retículas Screentone Avançadas** | Renderização monocromática de mangá com suporte a retículas halftone pontilhadas (`--manga`). |
| **Encoder Visual ViT (`VisionEncoder`)** | Vision Transformer para injeção de até 4 imagens de referência via atenção cruzada, mantendo consistência de personagens. |
| **Visão Neural & Crítico de Estúdio** | O `StudioStyleCritic` avalia o cel shading, a lineart e a composição e aciona refinamento img2img automático caso a nota fique abaixo de `--target-score`. |
| **Terminal TrueColor & Relatórios HTML** | Exibição de arte direto no terminal ANSI TrueColor (`--view`) e relatórios interativos em HTML (`--html`). |

---

## 📊 Capacidade e Contagem Exata de Parâmetros

Contagem **exata e atualizada** de parâmetros calculada na arquitetura PyTorch do projeto:

### 🔬 Tabela Detalhada por Componente

| Componente da U-Net / Encoders | `xxlarge` (1024×1024) | `xlarge` (512×512) | `large` (256×256) | `lite` (32×32) |
| --- | ---: | ---: | ---: | ---: |
| **Embedding Temporal** (`time_embedding`) | 4.721.664 | 2.099.200 | 1.181.184 | 8.320 |
| **Encoder de Prompt** (`prompt_encoder`) | 567.813.120 | 202.328.064 | 85.645.824 | 124.672 |
| **Encoder Visual ViT** (`vision_encoder`) | 568.708.608 | 152.212.480 | 57.450.240 | 113.856 |
| **Fusão de Condição** (`condition`) | 7.080.960 | 3.147.776 | 1.771.008 | 12.416 |
| **Projeção de Entrada** (`input`) | 8.960 | 5.376 | 3.584 | 448 |
| **Blocos Iniciais U-Net** (`down_blocks`) | 11.315.200 | 3.174.912 | 1.477.632 | 13.568 |
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

### 💻 Inspeção em Python

```python
from src.models.config import ModelConfig
from src.models.diffusion import DiffusionDenoiser, count_trainable_parameters

config = ModelConfig.xxlarge()
model = DiffusionDenoiser(config)
print(f"Parâmetros do perfil supremo: {count_trainable_parameters(model):,}")
# Resultado: 1,430,660,035
```

---

## ⚡ Início Rápido

### 1. Instalação do Ambiente

```bash
git clone <URL-DO-REPOSITORIO>
cd AI-Anime-and-Manga-Generator
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # Instala PyTorch 2.2+
```

### 2. Gerar Ilustrações e Páginas de Mangá

```bash
# 1. Ilustração Anime no Perfil Supremo (1024x1024, 1.43B params)
python scripts/generate.py \
  --prompt "guerreira de cabelo vermelho em uma cidade ao entardecer" \
  --preset xxlarge \
  --palette fantasia \
  --output saida/anime_xxlarge.png \
  --view

# 2. Página de Mangá com Retículas, Quadros e Speed Lines
python scripts/generate.py \
  --prompt "samurai em duelo sob chuva intensa" \
  --palette manga_shonen \
  --manga \
  --speed-lines \
  --panel-layout \
  --output saida/manga_page.png \
  --view

# 3. Ficha Settei de Produção de Personagem
python scripts/generate.py \
  --prompt "heroína de anime em traje escolar, design de estúdio" \
  --preset xlarge \
  --settei-sheet \
  --output saida/character_settei.png \
  --view
```

### 3. Treinar o Modelo Localmente

```bash
# Treino no perfil supremo (xxlarge, 1024x1024, 1.43B params)
python scripts/train.py \
  --images data/raw \
  --annotations data/annotations \
  --preset xxlarge \
  --epochs 100 \
  --batch-size 4 \
  --gradient-accumulation-steps 2 \
  --mixed-precision
```

---

## 🎨 Regras de Estúdio, Manga e 19 Paletas Settei

| Regra de Produção | Aplicação no Sistema |
| --- | --- |
| **Olhos em Camadas** | Esclerótica branca pura, pupila com profundidade, íris em gradiente e múltiplos destaques de luz (*catchlights*). |
| **Lineart Quente RGB(61,43,40)** | Linhas contornadas em tom marrom escuro quente, evitando a rigidez do preto puro `(0,0,0)`. |
| **Cel Shading Limpo** | Quantização de iluminação em blocos (Base, Sombra e Destaque) com bordas duras. |
| **Linhas de Ação de Mangá** | Traços radiais e foco de movimento procedural (`apply_speed_lines`). |
| **Calculo de Retículas (Halftone)** | Modulação de frequência de pontos de meio-tom em áreas de sombra (`apply_advanced_screentone`). |
| **19 Paletas Settei** | Opções via `--palette`: `fantasia`, `scifi`, `shonen`, `idol`, `dark`, `manga_shonen`, `manga_shojo`, `manga_seinen`, `manga_monochrome`, `anime_modern`, `anime_retro_90s`, `cyberpunk_anime`, `fantasy_isekai`, `slice_of_life` e mais. |

---

## 👁️ Visão Neural e Crítico Autocorrector

```text
 Prompt ──► build_prompt() ──► Filtro Estrito Anime/Mangá
                                         │
                                         ▼
 Referências ──► VisionEncoder (ViT) ──► U-Net Denoiser (1.43B)
                                         │
                                         ▼
                             apply_studio_finish()
        (Cel Shading · Lineart Quente · Screentones · Speed Lines)
                                         │
                                         ▼
                            StudioStyleCritic.analyze()
                                         │
                          ┌──────────────┴──────────────┐
                    Nota >= Meta                  Nota < Meta
                          │                             │
                          ▼                             ▼
                   Arte Final PNG             Refinamento img2img
                  + Terminal View            (Autocorreção de Visão)
```

---

## ⚙️ Guia Completo de CLI e Parâmetros

### 1. CLI de Geração (`scripts/generate.py`)

| Argumento | Padrão | Descrição |
| --- | --- | --- |
| `--prompt` | `"guerreira anime..."` | Prompt da ilustração ou cena de mangá |
| `--preset` | `"large"` | Perfil da arquitetura (`large`, `xlarge`, `xxlarge`, `lite`) |
| `--palette` | `None` | Ficha de cores settei (19 opções disponíveis) |
| `--manga` | `False` | Ativa modo mangá monocromático com retículas halftone |
| `--speed-lines` | `False` | Aplica linhas de velocidade e ação radiais de mangá |
| `--speed-line-density` | `0.5` | Densidade das linhas de velocidade (0.0 a 1.0) |
| `--panel-layout` | `False` | Gera bordas e calhas de painel/quadro de mangá |
| `--panel-rows`, `--panel-cols` | `2`, `2` | Divisão de quadros do painel de mangá |
| `--settei-sheet` | `False` | Formata como folha de personagem de estúdio |
| `--reference` | `[]` | Imagens de referência para o VisionEncoder (até 4) |
| `--refine-rounds` | `1` | Rodadas de autocorreção visual por visão neural |
| `--target-score` | `0.75` | Pontuação mínima almejada pelo crítico de estúdio |
| `--view`, `--show` | `False` | Exibe a arte gerada direto no terminal ANSI TrueColor |
| `--html` | `None` | Exporta relatório visual interativo em HTML |

---

## 🧪 Testes e Verificação

A suíte de testes unitários inclui verificação de todos os componentes de rede, encoders, regras de estúdio, geradores de retículas, calhas e speed lines:

```bash
python3 -m unittest discover -s tests
```

---

## 📜 Licença

Distribuído sob a licença **MIT**. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.
