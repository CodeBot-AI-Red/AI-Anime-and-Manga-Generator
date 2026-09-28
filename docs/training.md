# Treinamento local

Esta etapa adiciona um pipeline modular e pequeno para validar o caminho de treinamento sem baixar dados, usar APIs externas ou gerar imagens finais.

## Fluxo

```text
Dataset local
  ↓
Pré-processamento (`src.data.DatasetPreprocessor`)
  ↓
Modelo (`SmallReconstructionModel`, substituível)
  ↓
Treinamento
  ↓
Loss (MSE)
  ↓
Atualização dos pesos (SGD local)
  ↓
Checkpoint JSON
  ↓
Validação (MSE e MAE)
```

`ProcessedDataLoader` recebe os `ProcessedSample` ou `ProcessedBatch` criados pelo pré-processamento existente; portanto, não cria um segundo sistema de dataset. O modelo de demonstração usa apenas escala e viés treináveis para reconstruir o tensor de entrada e também chama o núcleo `AnimeMangaCore` para preservar os contratos de imagens, textos e cenas. Uma arquitetura maior poderá implementar a mesma interface (`forward`, `backward`, `state_dict` e `load_state_dict`).

## Teste local

Execute `python scripts/train.py`. O script gera tensores sintéticos somente em memória, treina por poucas épocas e cria checkpoints em `checkpoints/`, que permanecem ignorados pelo Git. Para retomar, crie um `Trainer` e chame `trainer.resume(caminho_do_checkpoint)` antes de `fit`.
