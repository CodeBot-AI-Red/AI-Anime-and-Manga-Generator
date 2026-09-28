# Arquitetura do núcleo de modelos

## Estado atual

Esta é a primeira versão do núcleo da IA. Ela é deliberadamente pequena e não
gera imagens, não baixa modelos e não treina parâmetros. O objetivo é fixar
interfaces simples para que as próximas etapas possam evoluir sem reorganizar o
projeto.

O núcleo recebe um lote de imagens no formato `(lote, canais, altura, largura)`,
uma lista de prompts e uma lista de descrições de cena. Todos precisam ter o
mesmo tamanho de lote. O resultado contém três tensores: latentes visuais,
embeddings de texto e embeddings de cena.

## Partes existentes

- `config.py`: define `ModelConfig`, com dimensões pequenas e validação de
  valores positivos.
- `tensors.py`: oferece um tensor mínimo em Python puro. Ele valida forma e
  quantidade de valores e evita exigir uma biblioteca de ML nesta fase.
- `image_network.py`: contém `ImageEncoder`. Hoje ele valida imagens BCHW e
  cria um mapa latente com resolução reduzida; futuramente será uma rede
  convolucional ou um encoder visual treinável.
- `text_conditioning.py`: contém `TextConditioner`, interface isolada para
  prompts. Seu embedding determinístico atual é apenas um placeholder para um
  tokenizer e encoder de texto.
- `scene.py`: contém `SceneDescription` e `SceneConditioner`. Os campos de
  personagem, pose, câmera, cenário e estilo já formam o contrato para os
  controles criativos futuros.
- `core.py`: contém `AnimeMangaCore`, que cria os módulos, valida o lote e
  entrega as três representações.

## Próximas etapas

O treinamento entrará atrás de `ImageEncoder`, `TextConditioner` e
`SceneConditioner`: os tensores leves poderão ser substituídos por tensores de
um framework de ML, e os componentes receberão pesos, perdas, dataloaders e um
loop de otimização no pacote `src/training/`.

O gerador final de imagens entrará depois de `AnimeMangaCore`, consumindo os
latentes e condicionamentos. Um decoder ou modelo de difusão poderá produzir a
imagem de anime ou mangá, enquanto os campos de `SceneDescription` poderão
controlar personagem, pose, enquadramento, cenário e estilo.

Para crescer, aumente `ModelConfig`, troque os placeholders por camadas
treináveis e adicione módulos especializados sem mudar a API principal. Isso
permite experimentar encoders maiores, atenção multimodal, controles de pose e
consistência de personagem de forma incremental.
