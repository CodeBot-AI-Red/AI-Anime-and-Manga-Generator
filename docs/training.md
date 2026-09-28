# Treinamento local de difusão

Esta etapa adiciona um caminho de difusão com autograd real, sem baixar dados, usar APIs externas ou pesos pré-treinados. Instale as dependências locais com `python -m pip install -r requirements.txt` no ambiente de treinamento.

## Fluxo

```text
Dataset local → DatasetPreprocessor → ProcessedDataLoader
    → timestep aleatório + ruído aleatório
    → DiffusionDenoiser condicionado pela caption
    → MSE contra o ruído conhecido + AdamW
    → validação e checkpoint .pt
```

`ProcessedDataLoader` recebe `ProcessedSample` ou `ProcessedBatch` criados pelo pré-processamento existente; portanto, não cria um segundo sistema de dataset. As imagens normalizadas são convertidas de `[0, 1]` para `[-1, 1]` antes de o ruído ser aplicado. A caption ou descrição de cena é o prompt do lote.

## Execução local

Use somente suas próprias imagens e anotações locais:

```bash
python scripts/train.py --images data/raw --annotations data/annotations --resolution 32
python scripts/generate.py --checkpoint checkpoints/epoch-2.pt --prompt "anime boy, black hair, school uniform, night" --output generated.png
```

O gerador começa em ruído, percorre o scheduler reverso e grava PNG sem internet (nem Pillow é necessário). Para retomar, crie um `Trainer` com o mesmo `DiffusionTrainingModel` e chame `resume` no checkpoint `.pt`.
