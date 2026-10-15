# ✨ AI Anime & Manga Generator

> Gerador de imagens de anime e mangá por difusão, **local**, **treinado do zero**, **exclusivo para estilo de anime e mangá**, em **grande e ultra grande porte**, com **visualização de imagens no terminal/HTML**, **visão neural** e **regras de estúdio** embutidas.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)
![Params](https://img.shields.io/badge/par%C3%A2metros-182M-ff69b4)
![License](https://img.shields.io/badge/license-MIT-2ea44f)

Não há download de pesos, dataset, API externa ou telemetria. Você fornece imagens e anotações **sobre as quais tem direitos de uso**, treina localmente e gera PNGs localmente.

> [!IMPORTANT]
> O perfil padrão agora é **grande porte (182,4 milhões de parâmetros, 256×256)** e exige **GPU** para treinar em tempo razoável. Sem um checkpoint treinado, os pesos são aleatórios e a imagem gerada é ruído. Para CPU e testes use `--preset lite` (656 mil parâmetros, 32×32).

## Índice

- [Novidades desta versão](#novidades-desta-versão)
- [Capacidade do modelo](#capacidade-do-modelo)
- [Início rápido](#início-rápido)
- [Visão: a IA vê as imagens](#visão-a-ia-vê-as-imagens)
- [Regras de estúdio aplicadas](#regras-de-estúdio-aplicadas)
- [Parâmetros](#parâmetros)
- [Arquitetura e fluxo](#arquitetura-e-fluxo)
- [Limitações](#limitações)
- [Testes](#testes)

## Novidades desta versão

| Área | O que mudou |
| --- | --- |
| **Domínio Estrito** | Exclusivo para **Anime e Mangá**: validação e sanitização automática (`validate_manga_anime_domain`, `enforce_manga_anime_style`) que bloqueia/purifica termos fora do escopo. |
| **Visualização** | **Visualizador de imagens integrado**: exibe artes diretamente no terminal via ANSI TrueColor de 24 bits (`--view`) e gera relatórios interativos em galeria HTML (`--html` ou `scripts/view_image.py`). |
| **Porte e Presets** | Perfil padrão **large** (182M params, 256×256) e novo perfil **xlarge** (512×512, 16 camadas de visão). |
| **Modo Mangá Screentone** | Opção `--manga` com acabamento monocromático de alta fidelidade e padrão de retículas (screentone halftone). |
| **Texto** | Encoder Transformer de **12 a 16 camadas**, prompts de até **512 bytes** (cabem as regras do estúdio inteiras). |
| **Visão Neural** | Novo **VisionEncoder** (ViT de 8-12 camadas, patches 16×16): até **4 imagens de referência** entram na atenção cruzada junto com o prompt. |
| **Auto-revisão** | `StudioStyleCritic` analisa a imagem gerada (ou enviada) e o gerador **refaz automaticamente** o que estiver fora do padrão (img2img + prompt de correção). |
| **Correção de imagens** | `--init-image` + `--strength` corrigem/refazem uma imagem sua. `scripts/inspect_image.py` e `scripts/view_image.py` diagnosticam qualquer imagem. |
| **Prompt negativo** | Guidance com prompt negativo (padrão: lista de "defeitos de IA" dos guias). |
| **Regras dos guias** | 7 pilares do look de estúdio, 5 paletas settei com hex reais, lineart RGB(61,43,40), acabamento de cel shading. |
| **Dispositivo** | Treino em `cuda`/`mps` automático (`--device auto`). |
| **Checkpoints** | `DiffusionTrainingModel.from_checkpoint()` recria a arquitetura salva automaticamente. |

## Capacidade do modelo

### **182.447.747 parâmetros treináveis** no perfil padrão (`large`)

| Componente | `large` (padrão) | `lite` |
| --- | ---: | ---: |
| Embedding temporal (`time_embedding`) | 1.181.184 | 8.320 |
| Encoder de prompt (`prompt_encoder`, 12 camadas) | 85.645.824 | 124.672 |
| **Encoder visual (`vision_encoder`, ViT 8 camadas)** | **57.450.240** | 113.856 |
| Fusão do condicionamento (`condition`) | 1.771.008 | 12.416 |
| Blocos residuais da U-Net (down/middle/up) | 29.834.368 | 297.360 |
| Entrada, reduções e reconstruções da U-Net | 5.247.616 | 82.512 |
| Atenção cruzada texto+imagem → U-Net | 1.313.792 | 16.768 |
| Saída da U-Net | 3.715 | 467 |
| **Total — `DiffusionDenoiser`** | **182.447.747** | **656.371** |

Para medir uma configuração customizada:

```bash
python - <<'PY'
from src.models.config import ModelConfig
from src.models.diffusion import DiffusionDenoiser, count_trainable_parameters
print(f"{count_trainable_parameters(DiffusionDenoiser(ModelConfig())):,} parâmetros")
PY
```

## Início rápido

```bash
git clone <URL-DO-SEU-FORK> && cd AI-Anime-and-Manga-Generator
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt                     # torch>=2.2 (Pillow opcional p/ JPEG/WebP)
```

Organize imagens e anotações (PNG 8 bits RGB/RGBA para treino):

```text
data/raw/chapter-01/page-001.png        data/annotations/chapter-01/page-001.json (ou .txt)
```

Treinar (GPU recomendada):

```bash
python scripts/train.py --images data/raw --annotations data/annotations \
  --epochs 100 --batch-size 8 --learning-rate 1e-4 --device auto
# CPU / teste rápido:
python scripts/train.py --images data/raw --annotations data/annotations --preset lite
```

Gerar seguindo as regras do estúdio, com autocorreção:

```bash
python scripts/generate.py --checkpoint checkpoints/epoch-100.pt \
  --prompt "guerreira de cabelo vermelho em uma cidade ao entardecer" \
  --palette fantasia --output saida/cena01.png
```

Saídas: `cena01.png` + `cena01.report.json` (relatório da visão).

## Visão: a IA vê as imagens

A IA "vê" imagens de duas formas complementares:

1. **Visão neural (`VisionEncoder`)** — um ViT treinado do zero transforma até 4 imagens em tokens que a U-Net consulta pela atenção cruzada. Durante o treino, 50% dos passos usam a própria imagem como referência (`reference_dropout`), ensinando o modelo a reaproveitar personagem, paleta e composição.
2. **Visão analítica (`StudioStyleCritic`)** — mede na imagem os itens do checklist "Como identificar que a imagem parece feita por IA" e devolve pontuação, diagnóstico e o prompt de correção do guia.

```bash
# Visualizar imagens diretamente no terminal e gerar relatório HTML
python scripts/generate.py --checkpoint ck.pt --prompt "guerreira anime em uma cidade ao entardecer" \
  --view --html relatorio.html

# Gerar arte de mangá monocromático com retícula (screentone)
python scripts/generate.py --checkpoint ck.pt --prompt "samurai em duelo" --palette manga_shonen --manga --view

# Visualizador CLI dedicado para qualquer imagem
python scripts/view_image.py minha_arte.png --reference ficha_aiko.png --cols 80 --html galeria.html

# Usar imagens suas como referência (ficha de personagem, cena anterior)
python scripts/generate.py --checkpoint ck.pt --prompt "mesma heroína correndo na chuva" \
  --reference ficha_aiko.png --reference cena_01.png

# Corrigir/refazer uma imagem sua no padrão de estúdio
python scripts/generate.py --checkpoint ck.pt --prompt "heroína anime, retrato" \
  --init-image minha_arte.png --strength 0.5

# Só analisar uma imagem (gerada ou enviada) e salvar versão com acabamento
python scripts/inspect_image.py minha_arte.png --reference ficha_aiko.png --fix minha_arte_corrigida.png
```

Exemplo de relatório:

```text
Pontuação de estúdio: 60% (256×256)
  [OK] Cor da lineart: 72% — 0% das linhas em preto puro; 38% em tom quente
  [!!] Superfícies chapadas (sem granulado): 0% — ruído médio em áreas planas = 0.0341
  [!!] Composição (regra dos terços): 35% — centro visual em (0.50, 0.51)
  Prompt de correção: remove all skin texture, apply flat anime cel surface, ...
```

Depois de gerar, se a pontuação ficar abaixo de `--target-score`, a imagem gerada vira **referência visual + imagem inicial** e é refeita com o prompt de correção (`--refine-rounds`).

## Regras de estúdio aplicadas

Fonte: guias *Criar Capas e Cenas de Anime* e *Corrigir e Atualizar Imagens de Anime* (settei / color correction sheets reais). Implementado em `src/studio/`.

| Regra do guia | Onde é aplicada |
| --- | --- |
| Olhos em camadas (esclerótica branca, BL, íris em gradiente, 2 catchlights dentro da íris) | prompt (`STUDIO_PILLARS`), prompt negativo, checagem de brancos, acabamento |
| Lineart variável, cor **RGB(61,43,40)**, nunca preto puro | prompt, checagem `lineart_preto`, acabamento converte preto → marrom |
| Cel shading de borda dura, zero gradiente | prompt, checagem `sombra_gradiente`, acabamento quantiza em 3 tons (Hi/Normal/Sombra) |
| Cabelo em 3–5 mechas, highlight na curva, raiz em sombra | prompt e prompt negativo |
| Paleta por personagem (settei) | `--palette` e checagem de consistência com `--reference` |
| Pele chapada, sem poros/granulado | checagem `textura`, acabamento com filtro mediano |
| Saturação +20–25% | checagem `cores_apagadas`, acabamento +20% |
| Composição fora do centro, luz de uma fonte | checagens `composicao_centralizada` e `contraste_baixo` |
| Dedos, perspectiva, fundo 3D, dobras redondas | prompt negativo |

Paletas disponíveis (`--palette`): `fantasia` (Akatsuki no Yona), `scifi` (Aquarion EVOL), `shonen` (Diamond no Ace), `idol` (AKB0048), `dark` (seinen/guerreiro).

## Parâmetros

### `ModelConfig` — arquitetura (padrão = `large`)

| Parâmetro | `large` (padrão) | `lite` | Descrição |
| --- | ---: | ---: | --- |
| `image_resolution` | (256, 256) | (32, 32) | Resolução gerada/treinada |
| `image_channels` | 3 | 3 | RGB |
| `base_channels` | 128 | 16 | Canais do 1º estágio da U-Net (×2, ×4 nos mais profundos) |
| `unet_res_blocks` | 3 | 2 | Blocos residuais por estágio |
| `downsample_factor` | 2 | 2 | Fator de redução por estágio |
| `num_timesteps` | 1000 | 100 | Passos de ruído do treino |
| `generation_steps` | 50 | 4 | Passos de amostragem DDIM |
| `time_embedding_dim` | 768 | 64 | Dimensão do condicionamento (texto, tempo e visão) |
| `guidance_scale` | 7.5 | 3.0 | Força do classifier-free guidance |
| `condition_dropout` | 0.1 | 0.1 | Fração de legendas descartadas no treino |
| `text_vocab_size` | 256 | 256 | Vocabulário byte-level |
| `max_prompt_length` | 512 | 128 | Bytes de prompt lidos |
| `text_encoder_layers` | 12 | 2 | Camadas do encoder de texto |
| `attention_heads` | 16 | 4 | Cabeças de atenção |
| `vision_enabled` | True | True | Liga o encoder visual |
| `vision_patch_size` | 16 | 8 | Tamanho do patch do ViT |
| `vision_layers` | 8 | 2 | Camadas do ViT |
| `vision_input_resolution` | 224 | 32 | Resolução das referências |
| `max_reference_images` | 4 | 4 | Imagens de referência por geração |
| `reference_dropout` | 0.5 | 0.5 | Fração de passos de treino sem referência |

### CLI de geração — `scripts/generate.py`

| Opção | Padrão | Descrição |
| --- | --- | --- |
| `--prompt` | "guerreira anime em uma cidade ao entardecer" | Pedido |
| `--negative-prompt` | — | Extra a evitar (somado ao negativo do estúdio) |
| `--palette` | — | `fantasia`, `scifi`, `shonen`, `idol`, `dark` |
| `--checkpoint` | — | Checkpoint `.pt` (arquitetura lida dele) |
| `--preset` | `large` | `large` ou `lite` (sem checkpoint) |
| `--resolution` | do preset | Sobrescreve a resolução |
| `--steps` | 50 | Passos de amostragem |
| `--guidance-scale` | 7.5 | Força do guidance |
| `--reference` | — | Imagem que a IA vê (repetível, até 4) |
| `--init-image` | — | Imagem sua para corrigir (img2img) |
| `--strength` | 0.6 | Quanto a imagem inicial pode mudar (0–1) |
| `--refine-rounds` | 1 | Rodadas de autocorreção após ver o resultado |
| `--target-score` | 0.75 | Pontuação de estúdio para parar de corrigir |
| `--seed` | 42 | Semente |
| `--device` | `cuda` se houver | Dispositivo |
| `--no-studio-rules` | desligado | Não injeta as regras dos guias |
| `--no-finish` | desligado | Desliga o acabamento de estúdio |
| `--output` | `generated.png` | PNG de saída (+ `.report.json`) |

### CLI de análise — `scripts/inspect_image.py`

`image` (obrigatório), `--reference` (consistência de cores), `--fix` (salva versão corrigida), `--json`.

### CLI de treino — `scripts/train.py`

| Opção | Padrão |
| --- | --- |
| `--images`, `--annotations` | obrigatórios |
| `--preset` | `large` |
| `--resolution` | 256 (`large`) / 32 (`lite`) |
| `--epochs` | 2 |
| `--batch-size` | 2 |
| `--learning-rate` | 1e-4 |
| `--device` | `auto` (cuda → mps → cpu) |
| `--checkpoint-dir` | `checkpoints` |

### `FinishConfig` — acabamento de estúdio

`cel_levels=3`, `cel_strength=0.6`, `saturation_boost=0.2`, `lineart_threshold=0.2`, `white_threshold=0.86`, `denoise=True`.

### `StudioStyleCritic`

`analysis_size=256`, `threshold=0.6` (abaixo disso o item vira correção).

### YAML em `configs/`

`model.yaml`, `generation.yaml` e `training.yaml` documentam os valores do perfil `large`.

## Arquitetura e fluxo

```text
prompt ─┐                       ┌─ 7 pilares + paleta (src/studio/rules.py)
        ├─ build_prompt ────────┘
        │
        ▼
 PromptEncoder (12 camadas) ──► tokens de texto ─┐
 VisionEncoder (ViT 8 camadas) ─► tokens visuais ─┼─► atenção cruzada ─► U-Net 128/256/512
   ▲ referências do usuário / imagem gerada      │
 prompt negativo ──► ramo "incondicional" do CFG ┘
        │
        ▼
 DDIM (50 passos, ou img2img com --init-image)
        ▼
 apply_studio_finish  (lineart RGB 61,43,40 · cel shading · brancos · +20% saturação)
        ▼
 StudioStyleCritic ──► nota < meta? ──► refaz com prompt de correção vendo a própria imagem
        ▼
 PNG + relatório JSON
```

## Limitações

- O modelo é treinado do zero: a qualidade depende do seu dataset e do tempo de GPU. 182M de parâmetros a 256×256 pedem milhares de imagens anotadas e muitas horas de treino.
- A visão analítica mede propriedades de pixel (cor, bordas, ruído, composição); ela não reconhece anatomia (ex.: contar dedos). Esses itens dos guias entram pelo prompt negativo.
- Checkpoints do perfil antigo (542 mil parâmetros) não carregam no perfil novo; use `--preset lite` ou retreine.

## Testes

```bash
python -m unittest discover -s tests
```

Cobre dados, pré-processamento, difusão, treino, visão (encoder, crítico, acabamento, img2img, prompt negativo) e checkpoints.

## Licença

MIT — veja [LICENSE](LICENSE).
